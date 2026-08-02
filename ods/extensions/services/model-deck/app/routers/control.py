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
"""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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

    def after(job: dict) -> None:
        warning = notify_engine(hot, deck)
        if warning is not None:
            # A model is already loaded — notify_engine deliberately
            # deferred the restart (never yanks a loaded model to register
            # a file). The move itself still succeeded; a load attempt here
            # is guaranteed to fail (the file isn't registered until the
            # next restart), so don't make one. The pre-armed suppressor
            # just expires — "on failure the window simply expires".
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
    deck["lemonade"].unload(model)
    # Deliberate unload: arm suppression so the arbiter doesn't heal it back.
    deck["heal_suppressor"].note_deck_unload()
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
    return {"status": "ok"}


@router.post("/hipfire/resume")
def hipfire_resume(request: Request, force: bool = False) -> dict:
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    deck["hipfire"].resume()
    return {"status": "ok"}
