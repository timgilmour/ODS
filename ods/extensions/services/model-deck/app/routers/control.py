"""
Control router — direct tenant lifecycle actions (load/unload/free/park/
resume), each a thin wrapper over exactly one engine client call. No auth:
the deck runs ops-first on a single-operator box (admin gate deliberately
removed 2026-07-22; the LAN path still sits behind Authelia via ods-lan).

E1 Task 7: ONE generic ``POST /{resource}/{verb}`` route replaces the five
fixed ``/lemonade/load``, ``/lemonade/unload``, ``/comfyui/free``,
``/hipfire/park``, ``/hipfire/resume`` routes. Today's URLs keep working
unchanged (the seeded resource names still ARE lemonade/comfyui/hipfire);
what changes is that a resource is now looked up in the LIVE declaration
(``deck["node_store"].get("local")["engines"]``) on every request, and its
client resolved through ``deck["local_clients"].client_for(resource)`` on
every request too — never a boot-time per-engine deck alias, which would
silently keep acting on the OLD connection after a live declaration edit
(T10 ships those edits). An unknown resource is a 404; a verb the
resource's declared KIND doesn't support (its ``app.engine_kinds`` adapter's
``human_verbs()``) is a 405 naming the kind — both strings are
UI-catalogued verbatim (Task 11).

human_verbs() is disjoint across kinds (load/unload only ever come from a
lemonade-kind resource, free only from comfyui, park/resume only from
hipfire — enforced by the 405 check above before any handler runs), so the
verb alone is enough to pick the right handler below; no kind-name literal
is needed in the dispatcher itself (spec §8: engine-kind names live in
app.engine_kinds, not here). ``dispatch`` DOES compute ``kind`` though (the
404/405 checks above need it), and threads it into every handler below —
each records its intent with ``engine=kind`` rather than a hardcoded
``"lemonade"``/``"hipfire"`` literal (final-review item 2). Human_verbs'
disjointness is exactly why that literal happened to be correct for every
resource that could ever reach it, kind-name-matching-resource-name
coincidences of the seeded triple aside — a future non-disjoint kind, or
simply removing the coincidence, would have silently mis-recorded which
adapter owns the intent (``app.reconcile.plan_reconcile`` reads
``intent["engine"]`` straight into the restore action's adapter lookup,
``app.arbiter``:1178).

``LoadBody``/``UnloadBody`` are constructed HERE, inside ``dispatch``, not
via FastAPI's automatic body binding (there is no single pydantic model
that fits every verb's body shape at the route-decorator level, since the
verb itself decides which body class applies). Hand-rolling the
construction means a ``ValidationError`` it raises does NOT go through
FastAPI's normal path — pydantic v2's ``ValidationError`` subclasses
``ValueError``, so an uncaught one would land on ``app.main``'s bare
``ValueError`` handler instead of the REDACTING ``RequestValidationError``
one, echoing the raw request body (``input``) into the 422 (final-review
item 1). Caught and re-raised as ``RequestValidationError(exc.errors())``
below, the same idiom ``app.routers.sets.create_set`` and
``app.routers.nodes``'s ``_shape_error`` already established for the
identical hand-rolled-validation shape.

Engine exceptions (GuardError, BusyError, EngineError) are deliberately left
to propagate uncaught — ``app.main`` registers app-wide exception handlers
that map them to their HTTP status (409/409/502). A malformed engine
response that raises a bare ``KeyError`` is NOT caught here either, per the
house "let it crash" policy: a 500 with a full traceback is the correct
signal for a real bug, not a guessed-at error code.

Every route that actually changes engine state records the result as
*intent* (``app.intent``), which is what lets the reconciler tell a
deliberate park from a dead backend. Two placement rules, both load-bearing:

* **Record placement depends on how long the engine call can run.**
  hipfire/comfyui/spark record AFTER the engine call returns: their guards
  raise inside the client call itself, so putting the record last already
  means "never record a call that didn't happen". The lemonade routes are
  the exception — a lemonade load/unload can run for seconds, and a
  reconciler tick landing in that window must see the operator's stated
  intent, not stale state; a call that raises is retried under the failure
  budget, not un-recorded (2026-08-06 design ruling). Either way, guards and
  model resolution still run first: a refused (409) action records nothing.
* **``state="unloaded"`` is intent, not its absence.** A park writes a
  record; it never deletes one. Deleting would make a deliberate unload
  indistinguishable from "nobody ever asked", and the reconciler would
  restore it on the next tick.

``comfyui/free`` deliberately records nothing: it drops cached VRAM while
the server stays up and keeps observing as loaded, so an "unloaded" intent
would derive as a permanent ``unexpected``.

The routes above run synchronously in the HTTP request thread and keep their
existing coordination (intent-record-first + ``load_in_flight`` + the heal
suppressor) unchanged — folding them into ``app.actuation`` is a design pass
for another day. ``_pull_through``'s completion hook is different: it fires
on the MOVER's worker thread, minutes after its request returned "pulling",
so it holds ``app.actuation.LOCK`` (task 6, the same lock ``app.sets.apply``
and the watcher tick's actuation phase share) around its restart+load, and
checks whether a deliberately-recorded intent has superseded it before doing
either — see ``_pull_through``.
"""

import time

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from app.engine_kinds import ENGINE_KINDS
from app.observe import local_key

router = APIRouter(prefix="/tenants", tags=["control"])

# Lemonade registers store GGUFs under an "extra." namespace. The Deck select
# carries bare GGUF filenames, so a manual load must prefix them to match.
_EXTRA_PREFIX = "extra."

# Pull-through readiness gate: DockerCtl.start() (called by notify_engine's
# restart) returns as soon as Docker ACCEPTS the request, not once lemonade
# is actually serving again — a load attempted immediately after would hit
# a still-booting server and 502. Poll status() until it stops raising, or
# give up after _READY_TIMEOUT_S (the post-move hook then correctly fails
# the job rather than silently eating the error).
_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 2.0


class LoadBody(BaseModel):
    model: str


class UnloadBody(BaseModel):
    # Omitted (or explicit null) -> unload whatever is currently loaded.
    model: str | None = None


def _ensure_host_agent_idle(deck, force: bool) -> None:
    """Refuse tenant mutations while a host-agent lifecycle op owns the box.

    The agent's activation snapshots + readiness proofs assume nothing else
    mutates engine state mid-flight; a deck unload mid-activation makes its
    readiness gate fail and roll back for no reason. force=True is the
    operator override. comfyui/free is deliberately unguarded — freeing VRAM
    helps an in-flight activation.
    """
    if force:
        return
    lifecycle = deck["hostagent"].lifecycle()
    if lifecycle["active"]:
        operation = lifecycle["operation"] or "model lifecycle operation"
        target = f" ({lifecycle['target']})" if lifecycle["target"] else ""
        raise HTTPException(
            status_code=409,
            detail=f"host agent is busy: {operation}{target}; wait for it to finish or pass ?force=true",
        )


def _declared_kind(deck, resource: str) -> str | None:
    """The kind declared for `resource` right now, read LIVE off node_store
    (never a boot-time copy — same posture as app.routers.build_world_snapshot)
    — or None when the resource isn't declared at all."""
    local = deck["node_store"].get("local")
    engines = local.get("engines", []) if local is not None else []
    for entry in engines:
        if entry["resource"] == resource:
            return entry["kind"]
    return None


@router.post("/{resource}/{verb}")
def dispatch(resource: str, verb: str, request: Request,
             body: dict | None = Body(default=None),
             force: bool = False, pull: bool = False) -> dict:
    """Generic control dispatch (E1 Task 7). See module docstring for the
    404/405 contract and why the verb alone (not a kind-name literal) is
    enough to pick a handler below."""
    deck = request.app.state.deck
    kind = _declared_kind(deck, resource)
    if kind is None:
        raise HTTPException(status_code=404, detail=f"unknown resource {resource!r}")
    if verb not in ENGINE_KINDS[kind].human_verbs():
        raise HTTPException(
            status_code=405,
            detail=f"{resource} ({kind}) does not support {verb}",
        )

    if verb == "load":
        try:
            load_body = LoadBody(**(body or {}))
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        return _lemonade_load(deck, resource, kind, load_body, force, pull)
    if verb == "unload":
        try:
            unload_body = UnloadBody(**(body or {}))
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        return _lemonade_unload(deck, resource, kind, unload_body, force)
    if verb == "free":
        return _comfy_free(deck, resource)
    if verb == "park":
        return _hipfire_park(deck, resource, kind, force)
    return _hipfire_resume(deck, resource, kind, force)  # verb == "resume"


def _lemonade_load(deck, resource: str, kind: str, body: LoadBody, force: bool, pull: bool) -> dict:
    _ensure_host_agent_idle(deck, force)
    client = deck["local_clients"].client_for(resource)
    model = body.model
    if not model.startswith(_EXTRA_PREFIX):
        model = f"{_EXTRA_PREFIX}{model}"
    bare = model.removeprefix(_EXTRA_PREFIX)

    hot_files = {m["file"] for m in deck["registry"].scan()}
    if bare not in hot_files:
        cold_unit = _find_cold_gguf(deck, bare)
        if cold_unit is not None:
            return _pull_through(deck, resource, kind, bare, cold_unit, pull)

    # Hot path — unchanged semantics (see original comments for suppressor
    # rationale), plus last-used bookkeeping for the storage catalog.
    # Arm suppression for the in-flight window: while the blocking load runs,
    # lemonade still reports "unloaded" and the GPU is already filling, so an
    # un-suppressed watcher tick would infer a pending default-route load and
    # stomp this one. On failure the window simply expires.
    deck["heal_suppressor"].note_deck_unload()
    # Recorded BEFORE the call, not after: this load can run for seconds, and
    # a reconciler tick landing mid-call must see the operator's stated
    # intent, not stale state. A raise below is retried under the failure
    # budget, not un-recorded. Record the PREFIXED name — that is what
    # lemonade was told and what the observation will report back.
    deck["intent_store"].record(
        local_key(resource), state="loaded", model=model, engine=kind
    )
    client.load(model)
    # Deliberate load succeeded: clear the window (and any prior unload's).
    deck["heal_suppressor"].clear()
    deck["catalog"].note_used_gguf(bare)
    return {"status": "ok"}


def _find_cold_gguf(deck, bare: str):
    for unit in deck["catalog"].scan():
        if unit["type"] == "gguf" and unit["name"] == bare and unit["state"] == "resident":
            loc = deck["location_store"].get(unit["location"])
            if loc and loc["engine"] != "lemonade":
                return unit
    return None


def _pull_through(deck, resource: str, kind: str, bare: str, unit: dict, pull: bool) -> dict:
    from datetime import UTC, datetime

    from app.engines import EngineError, GuardError
    from app.events import log_event
    from app.notify import notify_engine
    from app.routers.storage import submit_move

    auto = deck["storage_policy_store"].get()["auto"]
    if not (auto or pull):
        raise GuardError(
            f"model {bare!r} is cold (in {unit['location']!r}) — "
            f"re-request with ?pull=true to pull it to hot storage first")

    hot_locs = [loc for loc in deck["location_store"].describe()
                if loc["engine"] == "lemonade" and loc["available"] and not loc["readonly"]]
    if not hot_locs:
        raise GuardError("no available hot lemonade location is registered")
    hot = max(hot_locs, key=lambda loc: loc["free_bytes"])   # most free space wins (spec)

    # Captured before submit_move — the copy this hook waits behind can run
    # minutes. app.intent.IntentStore.record stamps `updated_ts` with
    # datetime.now(UTC).isoformat() (app/intent.py:41-42,112) — an ISO
    # string, NOT epoch seconds — so this is captured in that exact native
    # representation; the supersession check below compares apples to
    # apples instead of coercing to a float.
    submitted_at = datetime.now(UTC).isoformat()

    def after(job: dict) -> None:
        from app import actuation
        with actuation.LOCK:
            # A deliberate OPERATOR action may have superseded this pull
            # while the copy ran (minutes): an operator's set parking
            # lemonade must not be undone by a load they asked for BEFORE
            # it. Only an operator-authored record outranks this hook —
            # the arbiter's own automatic records (idle-release,
            # contention-eviction; both stamp actor="deck", app/arbiter.py)
            # do NOT: an idle model unloading mid-copy must not silently
            # drop the operator's explicit pull-through load, leaving the
            # file moved-but-unregistered [max-review Important-1, task 6
            # follow-up]. A record with no "actor" at all (pre-upgrade
            # intent.json) reads as "operator" — conservative, preserving
            # this check's original skip behavior for those.
            entry = deck["intent_store"].get().get(local_key(resource))
            if (
                entry is not None
                and entry.get("actor", "operator") == "operator"
                and entry.get("updated_ts", "") > submitted_at
            ):
                log_event(deck["events_path"], "pull-through-superseded",
                          {"job": job["id"], "model": bare,
                           "intent_state": entry.get("state")})
                return
            warning = notify_engine(hot, deck)
            if warning is not None:
                # A model is already loaded — notify_engine deliberately
                # deferred the restart (never yanks a loaded model to
                # register a file). The move itself still succeeded; a load
                # attempt here is guaranteed to fail (the file isn't
                # registered until the next restart), so don't make one.
                # The pre-armed suppressor just expires — "on failure the
                # window simply expires".
                log_event(deck["events_path"], "storage_notify_deferred",
                          {"job": job["id"], "warning": warning, "model": bare})
                return
            # Restart happened: poll for readiness before handing lemonade a
            # load (see _READY_TIMEOUT_S/_READY_POLL_S above). Resolved
            # fresh HERE, not captured at submit time — this hook fires
            # minutes later on the mover's worker thread, and the resource's
            # declared connection may have changed in the meantime (T10).
            client = deck["local_clients"].client_for(resource)
            deadline = time.monotonic() + _READY_TIMEOUT_S
            ready = False
            while time.monotonic() < deadline:
                try:
                    client.status()
                    ready = True
                    break
                except EngineError:
                    time.sleep(_READY_POLL_S)
            if not ready:
                raise RuntimeError(f"lemonade not ready {_READY_TIMEOUT_S}s after restart")
            # The pull-through load lands here, minutes after the request
            # returned "pulling" — it is no less deliberate, so it records
            # too, and BEFORE the call for the same reason as the hot path
            # above. actor="operator" explicit (not just the default): this
            # hook is completing an operator's own earlier request, the
            # third writer class the supersession check above depends on
            # being told apart from the arbiter's automatic "deck" records.
            deck["intent_store"].record(
                local_key(resource), state="loaded", model=f"{_EXTRA_PREFIX}{bare}",
                engine=kind, actor="operator",
            )
            client.load(f"{_EXTRA_PREFIX}{bare}")
            deck["heal_suppressor"].clear()
            deck["catalog"].note_used_gguf(bare)

    # Pre-arm: the multi-minute pull must not fight the VRAM watcher's
    # pending-load inference (spec section 3). on_progress re-arms it on every
    # chunk, because a big model's copy easily outlives the 600 s window.
    deck["heal_suppressor"].note_deck_unload()
    try:
        job = submit_move(deck, unit["id"], hot["name"],
                          label="pull-through load", on_success=after,
                          on_progress=deck["heal_suppressor"].note_deck_unload)
    except Exception:
        # Refused (guard trip, unknown unit, ...): there is no pull to protect,
        # so the pre-armed window must not linger and mute contention healing.
        deck["heal_suppressor"].clear()
        raise
    return {"status": "pulling", "job": job["id"]}


def _lemonade_unload(deck, resource: str, kind: str, body: UnloadBody, force: bool) -> dict:
    _ensure_host_agent_idle(deck, force)
    client = deck["local_clients"].client_for(resource)
    model = body.model
    if model is None:
        model = client.status()["loaded"]
        if not model:
            raise HTTPException(status_code=409, detail="no model is currently loaded")
    # Recorded BEFORE the call: unload takes ~2.2s, and a 2s watcher tick
    # landing mid-call must derive this as a deliberate park, never a dead
    # backend to restore. The 409 guard above already ran, so a refused
    # unload never reaches here and records nothing.
    deck["heal_suppressor"].note_deck_unload()
    # ...and record it as intent, which is the durable half of the same
    # statement: the suppression window expires, this does not.
    deck["intent_store"].record(
        local_key(resource), state="unloaded", model=None, engine=kind
    )
    client.unload(model)
    return {"status": "ok"}


def _comfy_free(deck, resource: str) -> dict:
    deck["local_clients"].client_for(resource).free()
    return {"status": "ok"}


def _hipfire_park(deck, resource: str, kind: str, force: bool) -> dict:
    # ?force=true skips the conversation-guard, never the route guard; the
    # host-agent busy guard below shares the same flag.
    _ensure_host_agent_idle(deck, force)
    deck["local_clients"].client_for(resource).park(force=force)
    deck["intent_store"].record(
        local_key(resource), state="unloaded", model=None, engine=kind
    )
    return {"status": "ok"}


def _hipfire_resume(deck, resource: str, kind: str, force: bool) -> dict:
    _ensure_host_agent_idle(deck, force)
    deck["local_clients"].client_for(resource).resume()
    # model=None, not a name: hipfire is single-model and the Deck does not
    # choose that model (it comes from the litellm route table, app/state.py).
    # None reads as "loaded, no opinion which model"; recording a name the
    # Deck cannot observe would manufacture permanent drift.
    deck["intent_store"].record(
        local_key(resource), state="loaded", model=None, engine=kind
    )
    return {"status": "ok"}
