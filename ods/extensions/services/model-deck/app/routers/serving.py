"""Serving routes — per-node single-slot swap control (N1, design §6), plus
(sglang-omni Task 8) the per-node DECLARED-ENGINE verb route.

The canonical home of what /api/spark/* did: same handlers, node-addressed.
No auth, matching the rest of the deck. A successful swap records intent
for slot_key(node_id) (app.intent), which is what makes the slot a
reconciled resource. Engine exceptions (GuardError/BusyError/EngineError)
propagate to the app-wide handlers (409/409/502).

_swap_and_record is the tail /swap and /reload share (swap -> intent record
-> observer invalidate -> response) so the two routes can never drift apart
on what "a swap happened" means to the rest of the deck.

``engines_router`` (POST /api/nodes/{node_id}/engines/{resource}/{verb}) is
the same idea for an engine DECLARED on a node-agent entry: the REMOTE
counterpart of app.routers.control's /tenants/{resource}/{verb}. It is a
separate router in this file rather than a route on app.routers.nodes'
because it ACTUATES — nodes.py is the registry's CRUD plus a probe, and its
own docstring is explicit that forget "never calls the engine: no client
lookup anywhere". The wire refusals mirror ``_client``'s posture right here
in this file (404 unknown node, 503 known-but-not-operable), and the
vocabulary refusal mirrors the local dispatcher's (405 naming the kind), so
an operator meets one contract on both surfaces.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.argline import POSITIONAL_KEY, render_argv
from app.compose_import import import_compose
from app.configure import apply_settings
from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError
from app.events import log_event
from app.observe import node_key, slot_key
from app.routers.settings import _declared_only, _resolve, _resolve_env

router = APIRouter(prefix="/nodes/{node_id}/serving", tags=["serving"])
engines_router = APIRouter(prefix="/nodes/{node_id}/engines", tags=["engines"])


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
    without a node id in its own path — today only rename's vLLM-profile
    gathering (app.routers.rename), a pre-N1 surface that never took a node
    parameter. Every per-node route under this router takes `node_id`
    directly and never calls this.

    Moved here (N1 T12 review, finding 1) from the since-removed
    /api/spark/* alias router, its other caller — a resolver two OTHER
    modules import cannot live in the module scheduled to go away first.

    None configured -> 503. The detail keeps the legacy unbuilt-engine
    wording (rename_plan's client_for branch shares it, and
    test_route_503_when_spark_not_configured pins the text) — the alias
    clients it was originally preserved for are gone. More than one -> 409
    naming the candidates, never guess ([[literal-declared-inputs]])."""
    ids = sorted(n["id"] for n in deck["node_store"].list()
                 if n.get("control") == "swap")
    if not ids:
        raise HTTPException(status_code=503,
                            detail="spark engine is not configured")
    if len(ids) > 1:
        raise HTTPException(status_code=409, detail=(
            "multiple swap nodes configured (" + ", ".join(ids) + "); "
            "this operation supports exactly one"))
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


# ===========================================================================
# Declared remote engines (sglang-omni Task 8) — one verb, one node-agent
# request, one intent record.
# ===========================================================================

# Human verb -> (the intent state it asserts, the node-agent engine-channel
# call that performs it).
#
# NOT engine-kind knowledge (spec §8 keeps that in app.engine_kinds): these
# are the CHANNEL's two verbs — extensions/services/node-agent's
# /v1/node/engine/{resource}/{up,down}, which every remote engine client is
# built over (app.node_clients._adapter_remote_client). What each KIND is
# willing to be asked is read from its own `human_verbs()` below, never from
# this table; a kind whose vocabulary includes a verb this channel has no
# row for is refused by name (501), never silently mapped onto a neighbour.
_REMOTE_VERBS = {"load": ("loaded", "up"), "unload": ("unloaded", "down")}


def _declared_engine(deck: dict, node_id: str, resource: str) -> dict:
    """The engine `resource` DECLARED on `node_id` right now, read LIVE off
    the registry (never a boot-time copy — app.routers.control._declared_kind
    for the local mirror of this), or the wire refusal: 404 for a node the
    registry has never heard of, 404 for a resource that node does not
    declare. Both name what was not found."""
    entry = deck["node_store"].get(node_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id!r}")
    for engine in entry.get("engines") or []:
        if engine["resource"] == resource:
            return engine
    raise HTTPException(status_code=404,
                        detail=f"unknown engine {resource!r} on node {node_id!r}")


@engines_router.post("/{resource}/{verb}", status_code=202)
def engine_verb(node_id: str, resource: str, verb: str, request: Request) -> dict:
    """Act on a declared remote engine: record the intent, then ask the
    node's agent to bring the engine up or down.

    202, not 200. The agent answers 202 itself — it queues the request for
    the host-side swap-helper and nothing here observes the result — and a
    cold sglang-omni start takes ~4 minutes (GF4). Returning 200 with
    {"status": "ok"} would claim an outcome this route cannot know.

    WHOEVER ACTUATES, RECORDS — AND RECORDS FIRST. Same rule, same reason as
    app.routers.control's lemonade arms and app.engine_kinds' own actuators:
    the actuation runs long (minutes here, not seconds), and a reconciler
    tick landing in that window must see the operator's stated intent rather
    than deriving a death from an engine that is merely still booting
    (`_SglangOmniAdapter.warming` is the rule that consumes exactly this
    record). model=None: this surface takes no model, and a remote engine's
    declaration names none either — it serves the one checkpoint its node's
    compose file mounts.

    A POST that RAISES actuated nothing, so the speculative record is rolled
    back — as a real compare-and-swap (predicate + write in ONE critical
    section, app.intent.put_back_if), never check-then-act: an operator or
    the arbiter can record during the seconds a hung transport takes, and
    blindly putting `prior` back would silently revert THEIR action. The
    witness is the exact `updated_ts` this call stamped, which is unique to
    the record it wrote — a stronger predicate than the arbiter arms'
    actor/state pair, which had no stamp of its own to compare.

    With NO prior record there is nothing to put back and no compare-and-swap
    forget to undo the write with, so the speculative record stands — the
    same accepted gap `_LemonadeAdapter.execute_unload` documents, and the
    honest reading for a load either way: the operator did ask for the engine
    to be up.

    No `force`, no body, and no host-agent guard: those all describe THIS
    box (app.routers.control._ensure_host_agent_idle guards the local host
    agent's own lifecycle ops), and the engine being acted on is on another
    one. The node's agent runs its own single-flight guard — a request while
    one is pending is its 409, surfaced here as the EngineError the client
    raises.
    """
    deck = request.app.state.deck
    engine = _declared_engine(deck, node_id, resource)
    kind = engine["kind"]
    if verb not in ENGINE_KINDS[kind].human_verbs():
        # Same refusal, same wording as the local dispatcher's (both strings
        # are UI-catalogued verbatim, Task 11).
        raise HTTPException(status_code=405,
                            detail=f"{resource} ({kind}) does not support {verb}")
    if verb not in _REMOTE_VERBS:
        # The kind says it takes this verb, but the node-agent engine channel
        # has no call for it — a wiring gap in the deck, not a bad request.
        # Refused by name (the totality floor app.arbiter._dispatch_verb
        # takes), never mapped onto whichever channel call looks closest.
        raise HTTPException(
            status_code=501,
            detail=f"{resource} ({kind}) declares {verb} but the node-agent "
                   f"engine channel has no call for it")
    state, call = _REMOTE_VERBS[verb]

    client = deck["remote_engine_clients"].client_for(node_id, resource)
    if client is None:
        # client_for is repair-shaped, never wire-shaped (app.node_clients):
        # None is "not operable right now" — not a node-agent entry (the
        # local node lands here: /api/tenants/* is its surface), no address,
        # no stored credential, or a declared kind with no remote
        # constructor. Same 503 family as _client's above.
        raise HTTPException(status_code=503, detail=(
            f"node {node_id!r} cannot act on engine {resource!r} — a "
            "node-agent entry with an address, a stored credential and a "
            "remote-capable kind is required"))

    key = node_key(node_id, resource)
    store = deck["intent_store"]
    prior = store.get().get(key)
    # Stamped explicitly so the rollback below has an exact witness for the
    # record THIS call wrote (see the docstring); app.intent.record would
    # otherwise stamp its own and never tell us what it chose.
    stamp = datetime.now(UTC).isoformat()
    store.record(key, state=state, model=None, engine=kind, actor="operator",
                 now=stamp)
    try:
        getattr(client, call)()
    except EngineError:
        # The channel's WHOLE failure vocabulary (app.engines' docstring:
        # a non-2xx — the agent's 404/409/503 included — and any transport
        # failure both arrive as EngineError). Narrow deliberately: anything
        # else is a bug in the deck, and crashes with its traceback.
        if prior is not None and not store.put_back_if(
                key, lambda current: (current is not None
                                      and current.get("updated_ts") == stamp),
                prior):
            # Someone else's intent is newer. Leave it alone and say so — a
            # silent skip here would be the same invisibility the rollback
            # exists to end (app.engine_kinds' arms log the same event).
            log_event(deck["events_path"], "engine-verb-rollback-skipped",
                      {"node": node_id, "resource": resource, "verb": verb,
                       "reason": "intent changed during the request"})
        # Re-raised, not swallowed: the app-wide handler maps it to 502, and
        # the operator learns their click did not land.
        raise
    # The observation half is TTL-cached (app.node_clients.RemoteObserver,
    # 10 s), and a verb's whole purpose is to change what it holds — the same
    # obligation _swap_and_record discharges for a node's serving slot.
    # Absent (a deck built without one) means nothing is cached to drop.
    observer = deck.get("remote_observer")
    if observer is not None:
        observer.invalidate()
    return {"status": "accepted", "node_id": node_id, "resource": resource,
            "verb": verb}
