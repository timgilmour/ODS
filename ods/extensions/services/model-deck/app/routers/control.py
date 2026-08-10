"""
Control router — direct tenant lifecycle actions (load/unload/free/park/
resume), each a thin wrapper over exactly one engine client call. No auth:
the deck runs ops-first on a single-operator box (admin gate deliberately
removed 2026-07-22; the LAN path still sits behind Authelia via ods-lan).

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

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.observe import LOCAL_HIPFIRE_KEY, LOCAL_LEMONADE_KEY

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


class LemonadeLoadBody(BaseModel):
    model: str


class LemonadeUnloadBody(BaseModel):
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


@router.post("/lemonade/load")
def lemonade_load(body: LemonadeLoadBody, request: Request,
                  force: bool = False, pull: bool = False) -> dict:
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    model = body.model
    if not model.startswith(_EXTRA_PREFIX):
        model = f"{_EXTRA_PREFIX}{model}"
    bare = model.removeprefix(_EXTRA_PREFIX)

    hot_files = {m["file"] for m in deck["registry"].scan()}
    if bare not in hot_files:
        cold_unit = _find_cold_gguf(deck, bare)
        if cold_unit is not None:
            return _pull_through(deck, bare, cold_unit, pull)

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
        LOCAL_LEMONADE_KEY, state="loaded", model=model, engine="lemonade"
    )
    deck["lemonade"].load(model)
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


def _pull_through(deck, bare: str, unit: dict, pull: bool) -> dict:
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
            entry = deck["intent_store"].get().get(LOCAL_LEMONADE_KEY)
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
            # load (see _READY_TIMEOUT_S/_READY_POLL_S above).
            deadline = time.monotonic() + _READY_TIMEOUT_S
            ready = False
            while time.monotonic() < deadline:
                try:
                    deck["lemonade"].status()
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
                LOCAL_LEMONADE_KEY, state="loaded", model=f"{_EXTRA_PREFIX}{bare}",
                engine="lemonade", actor="operator",
            )
            deck["lemonade"].load(f"{_EXTRA_PREFIX}{bare}")
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


@router.post("/lemonade/unload")
def lemonade_unload(body: LemonadeUnloadBody, request: Request, force: bool = False) -> dict:
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    model = body.model
    if model is None:
        model = deck["lemonade"].status()["loaded"]
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
        LOCAL_LEMONADE_KEY, state="unloaded", model=None, engine="lemonade"
    )
    deck["lemonade"].unload(model)
    return {"status": "ok"}


@router.post("/comfyui/free")
def comfyui_free(request: Request) -> dict:
    request.app.state.deck["comfy"].free()
    return {"status": "ok"}


@router.post("/hipfire/park")
def hipfire_park(request: Request, force: bool = False) -> dict:
    # ?force=true skips the conversation-guard, never the route guard; the
    # host-agent busy guard below shares the same flag.
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    deck["hipfire"].park(force=force)
    deck["intent_store"].record(
        LOCAL_HIPFIRE_KEY, state="unloaded", model=None, engine="hipfire"
    )
    return {"status": "ok"}


@router.post("/hipfire/resume")
def hipfire_resume(request: Request, force: bool = False) -> dict:
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    deck["hipfire"].resume()
    # model=None, not a name: hipfire is single-model and the Deck does not
    # choose that model (it comes from the litellm route table, app/state.py).
    # None reads as "loaded, no opinion which model"; recording a name the
    # Deck cannot observe would manufacture permanent drift.
    deck["intent_store"].record(
        LOCAL_HIPFIRE_KEY, state="loaded", model=None, engine="hipfire"
    )
    return {"status": "ok"}
