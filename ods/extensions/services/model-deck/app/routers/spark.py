"""
Spark router — remote single-slot swap control (lifecycle only, no VRAM
arbitration; see app/engines/spark.py for why). No auth, matching the rest
of the deck (ops-first single-operator box; LAN sits behind Authelia).

deck["spark"] is None on boxes with no spark configured — every endpoint
answers 503 then, so the UI can feature-detect with one GET.

A successful swap records intent for the slot (app.intent), which is what
makes the spark a reconciled resource rather than a read-only one. Engine
exceptions (GuardError/BusyError/EngineError) propagate to the app-wide
handlers (409/409/502), per the house policy in control.py.

POST /reload (Plan C2, Task 7) is design decision 5's ONE human action:
resolve the ladder for whatever profile is (or will be) serving, ship it to
the node via the node-settings configure mech, then re-swap that SAME
profile so the shipped settings actually launch. The re-swap's intent
record is what clears settings_drift — the drift flag's baseline IS the
intent's updated_ts (see app.routers._settings_drift), so re-recording it
is the entire "clearing" mechanism; nothing here touches settings_drift
directly. Two guards refuse a launch-breaking document before anything is
shipped — see spark_reload's own docstring. _swap_and_record is the tail
/swap and /reload share (swap -> intent record -> observer invalidate ->
response) so the two routes can never drift apart on what "a swap happened"
means to the rest of the deck.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.argline import POSITIONAL_KEY, render_argv
from app.compose_import import import_compose
from app.configure import apply_settings
from app.observe import SPARK_SLOT_KEY, spark_node_id
from app.routers.settings import _declared_only, _resolve, _resolve_env

router = APIRouter(prefix="/spark", tags=["spark"])


class SwapBody(BaseModel):
    profile: str
    # Skips only the busy guard; the litellm default-route guard is
    # force-proof by design (see SparkClient.swap).
    force: bool = False


class ReloadBody(BaseModel):
    profile: str | None = None
    # Passed straight through to the re-swap at the end — same semantics
    # as SwapBody.force above.
    force: bool = False


def _spark(request: Request):
    spark = request.app.state.deck.get("spark")
    if spark is None:
        raise HTTPException(status_code=503,
                            detail="spark engine is not configured")
    return spark


@router.get("/status")
def spark_status(request: Request) -> dict:
    return _spark(request).status()


def _swap_and_record(deck: dict, spark, profile: str, force: bool) -> dict:
    """The tail shared by /swap and /reload: swap, then — only on success —
    record intent and invalidate the cached observation. Moved here
    verbatim from spark_swap (Task 7) so reload's re-swap goes through the
    exact same bookkeeping, not a second, driftable copy of it."""
    out = spark.swap(profile, force=force)
    # Last, and only on success: a guard-refused swap never happened, and
    # intent is last-known-GOOD. Without this write the slot is only ever
    # read, derives 'unmanaged' forever, and the reconciler's spark restore
    # branch is unreachable in production.
    #
    # The identity recorded is the PROFILE, not the served model name:
    # mm27b serves under --served-model-name aeon, so recording the served
    # name would report permanent drift for a perfectly correct placement
    # (app.observe.translate_spark_status makes the same choice).
    deck["intent_store"].record(
        SPARK_SLOT_KEY, state="loaded", model=profile, engine="spark")
    # The swap just changed what the observation cache is holding.
    observer = deck.get("spark_observer")
    if observer is not None:
        observer.invalidate()
    return {"status": "ok", **out}


@router.post("/swap")
def spark_swap(body: SwapBody, request: Request) -> dict:
    deck = request.app.state.deck
    return _swap_and_record(deck, _spark(request), body.profile, body.force)


@router.post("/reload")
def spark_reload(body: ReloadBody, request: Request) -> dict:
    """Ship the resolved declared settings for a profile, then re-swap it.

    Two pre-ship guards exist because the helper's settings-owned branch
    TEARS DOWN every profile container BEFORE `docker compose up` reads the
    override (node-agent/swap-helper/swap-helper.sh `_launch`): a document
    compose or the engine rejects leaves the node serving nothing, and the
    Deck is the only place that can still refuse.

    * SERVICE MISMATCH — the shipped `service` key is the one adopt saw. If
      the compose service has been renamed since, the override introduces a
      second service with no image and compose config validation fails
      after teardown. Re-adopt is the remedy. ``force`` does NOT bypass
      this: a wrong service name can never launch, so there is nothing for
      force to assert.
    * SPARSE ARGV — a pre-C2 `kept` scope can hold args without the
      ``serve /model`` positionals. That argv is non-empty, so the helper
      owns the launch with it and the engine never gets its subcommand. An
      EMPTY declared set is refused by the same guard: it ships an empty
      argv, which the helper reads as "asserts nothing" and delegates to
      swap.sh, so the document's env would silently never apply either.
      ``force`` DOES bypass this one — it is the operator asserting the
      image's entrypoint supplies the subcommand, a claim only they can
      make.

    The service check costs one extra node round-trip per reload
    (``get_compose``); reload is a human click, not a loop, and the
    alternative is trusting an adopt-time snapshot with the whole slot at
    stake. A node that cannot serve the compose (EngineError -> 502) or
    serves an unparseable one (ValueError -> 422) fails the reload here
    rather than mid-teardown.
    """
    deck = request.app.state.deck
    spark = _spark(request)

    # No explicit profile -> reload whatever the node last swapped to. No
    # explicit profile AND nothing serving is a 409: there is nothing to
    # name a target from.
    profile = body.profile or (spark.status().get("swap_status") or {}).get("profile")
    if not profile:
        raise HTTPException(status_code=409,
                            detail="nothing is serving; name a profile")

    node = spark_node_id()
    entry = deck["characteristics_store"].entry(f"engine/{node}/vllm")
    identities = (entry.get("profile_identities") or {}).get("value") or {}
    if profile not in identities:
        raise HTTPException(status_code=409, detail=(
            f"profile {profile!r} has no adopted identity; "
            f"POST /api/settings/adopt/{node}/vllm first"))
    info = identities[profile]

    # Fetched FRESH, not read from the identity map: the map is an
    # adopt-time snapshot, and a stale service name is precisely what this
    # check exists to catch (see the docstring).
    service = import_compose(spark.get_compose(profile))["service"]
    if service != info["service"]:
        raise HTTPException(status_code=409, detail=(
            f"profile {profile!r} compose service is {service!r} but the "
            f"adopted identity says {info['service']!r}; "
            f"POST /api/settings/adopt/{node}/vllm to re-adopt"))

    # Declared-only (design decision 3): what ships to the node is exactly
    # what the argline would show — engine_defaults/checkpoint_recommendations
    # are the engine's own applied behavior, never re-asserted back at it.
    resolved = _resolve(deck, node, "vllm", info["identity"])
    declared = _declared_only(resolved)
    if POSITIONAL_KEY not in declared and not body.force:
        raise HTTPException(status_code=409, detail=(
            f"declared settings for {profile!r} are not launch-shaped "
            f"(no {POSITIONAL_KEY}, e.g. ['serve', '/model']); "
            f"POST /api/settings/adopt/{node}/vllm first, "
            f"or force to launch on the image entrypoint alone"))

    env = _resolve_env(deck, node, "vllm", info["identity"])
    outcome = apply_settings(
        "node-settings", engine_client=spark, resolved=declared,
        profile=profile, env=env,
        argv=render_argv({k: v["value"] for k, v in declared.items()}),
        service=info["service"])

    swap = _swap_and_record(deck, spark, profile, body.force)
    return {"shipped": outcome["applied"], "profile": profile, **swap}
