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

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/tenants", tags=["control"])

# Lemonade registers store GGUFs under an "extra." namespace. The Deck select
# carries bare GGUF filenames, so a manual load must prefix them to match.
_EXTRA_PREFIX = "extra."


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
def lemonade_load(body: LemonadeLoadBody, request: Request, force: bool = False) -> dict:
    deck = request.app.state.deck
    _ensure_host_agent_idle(deck, force)
    model = body.model
    if not model.startswith(_EXTRA_PREFIX):
        model = f"{_EXTRA_PREFIX}{model}"
    # Arm suppression for the in-flight window: while the blocking load runs,
    # lemonade still reports "unloaded" and the GPU is already filling, so an
    # un-suppressed watcher tick would infer a pending default-route load and
    # stomp this one. On failure the window simply expires.
    deck["heal_suppressor"].note_deck_unload()
    deck["lemonade"].load(model)
    # Deliberate load succeeded: clear the window (and any prior unload's).
    deck["heal_suppressor"].clear()
    return {"status": "ok"}


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
