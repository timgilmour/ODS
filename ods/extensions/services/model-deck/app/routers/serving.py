"""Serving routes — per-node single-slot swap control (N1, design §6).

The canonical home of what /api/spark/* did: same handlers, node-addressed.
No auth, matching the rest of the deck. A successful swap records intent
for slot_key(node_id) (app.intent), which is what makes the slot a
reconciled resource. Engine exceptions (GuardError/BusyError/EngineError)
propagate to the app-wide handlers (409/409/502).

_swap_and_record is the tail /swap and /reload share (swap -> intent record
-> observer invalidate -> response) so the two routes can never drift apart
on what "a swap happened" means to the rest of the deck.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.argline import POSITIONAL_KEY, render_argv
from app.compose_import import import_compose
from app.configure import apply_settings
from app.observe import slot_key
from app.routers.settings import _declared_only, _resolve, _resolve_env

router = APIRouter(prefix="/nodes/{node_id}/serving", tags=["serving"])


class SwapBody(BaseModel):
    profile: str
    # Skips only the busy guard; the litellm default-route guard is
    # force-proof by design (see SparkClient.swap).
    force: bool = False


class ReloadBody(BaseModel):
    profile: str | None = None
    force: bool = False


def _client(request: Request, node_id: str):
    """The node's actuation client, or the wire refusal: 404 for an id the
    registry has never heard of, 503 for a known node that is not operable
    (control != "swap" or a prerequisite missing) — the same family as the
    old unbuilt-engine 503, message updated to say what to fix."""
    deck = request.app.state.deck
    if deck["node_store"].get(node_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
    client = deck["node_clients"].client_for(node_id)
    if client is None:
        raise HTTPException(status_code=503, detail=(
            f"node {node_id!r} is not operable — control: \"swap\" with "
            "address, serving_address and a credential required"))
    return client


def single_swap_node_id(deck: dict) -> str:
    """Resolve "the one swap node" for a caller that addresses the fleet
    without a node id in its own path — the /api/spark/* alias
    (app.routers.spark) and rename's vLLM-profile gathering
    (app.routers.rename), both pre-N1 surfaces that never took a node
    parameter. Every per-node route under this router takes `node_id`
    directly and never calls this.

    Moved here (N1 T12 review, finding 1) from app.routers.spark, which is
    DEPRECATED with a published removal target — a resolver two OTHER
    modules import cannot live in the module scheduled to go away first.

    None configured -> 503 (the legacy unbuilt-engine message, so existing
    feature-detecting callers keep working); more than one -> 409 naming the
    candidates, never guess ([[literal-declared-inputs]])."""
    ids = sorted(n["id"] for n in deck["node_store"].list()
                 if n.get("control") == "swap")
    if not ids:
        raise HTTPException(status_code=503,
                            detail="spark engine is not configured")
    if len(ids) > 1:
        raise HTTPException(status_code=409, detail=(
            "multiple swap nodes configured (" + ", ".join(ids) + "); "
            "use /api/nodes/{id}/serving/... instead"))
    return ids[0]


@router.get("/status")
def serving_status(node_id: str, request: Request) -> dict:
    return _client(request, node_id).status()


def _swap_and_record(deck: dict, node_id: str, client, profile: str,
                     force: bool) -> dict:
    """swap -> (on success only) intent record -> observer invalidate.
    Moved from app/routers/spark.py verbatim, keyed per node. The identity
    recorded is the PROFILE (see app.observe.translate_spark_status);
    the engine-class string stays "spark" (Tim's 08-12 ruling)."""
    out = client.swap(profile, force=force)
    deck["intent_store"].record(
        slot_key(node_id), state="loaded", model=profile, engine="spark")
    observers = deck.get("node_observers")
    if observers is not None:
        observers.invalidate(node_id)
    return {"status": "ok", **out}


@router.post("/swap")
def serving_swap(node_id: str, body: SwapBody, request: Request) -> dict:
    deck = request.app.state.deck
    return _swap_and_record(deck, node_id, _client(request, node_id),
                            body.profile, body.force)


@router.post("/reload")
def serving_reload(node_id: str, body: ReloadBody, request: Request) -> dict:
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
    client = _client(request, node_id)

    # No explicit profile -> reload whatever the node last swapped to. No
    # explicit profile AND nothing serving is a 409: there is nothing to
    # name a target from.
    profile = body.profile or (client.status().get("swap_status") or {}).get("profile")
    if not profile:
        raise HTTPException(status_code=409,
                            detail="nothing is serving; name a profile")

    node = node_id
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
    service = import_compose(client.get_compose(profile))["service"]
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
        "node-settings", engine_client=client, resolved=declared,
        profile=profile, env=env,
        argv=render_argv({k: v["value"] for k, v in declared.items()}),
        service=info["service"])

    swap = _swap_and_record(deck, node_id, client, profile, body.force)
    return {"shipped": outcome["applied"], "profile": profile, **swap}
