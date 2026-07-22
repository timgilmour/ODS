"""
Control router — direct tenant lifecycle actions (load/unload/free/park/
resume), each a thin wrapper over exactly one engine client call. Every
route here mutates the live box, so the whole router carries
``Depends(require_admin)``.

Engine exceptions (GuardError, BusyError, EngineError) are deliberately left
to propagate uncaught — ``app.main`` registers app-wide exception handlers
that map them to their HTTP status (409/409/502). A malformed engine
response that raises a bare ``KeyError`` is NOT caught here either, per the
house "let it crash" policy: a 500 with a full traceback is the correct
signal for a real bug, not a guessed-at error code.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.security import require_admin

router = APIRouter(prefix="/tenants", tags=["control"], dependencies=[Depends(require_admin)])

# Lemonade registers store GGUFs under an "extra." namespace. The Deck select
# carries bare GGUF filenames, so a manual load must prefix them to match.
_EXTRA_PREFIX = "extra."


class LemonadeLoadBody(BaseModel):
    model: str


class LemonadeUnloadBody(BaseModel):
    # Omitted (or explicit null) -> unload whatever is currently loaded.
    model: str | None = None


@router.post("/lemonade/load")
def lemonade_load(body: LemonadeLoadBody, request: Request) -> dict:
    deck = request.app.state.deck
    model = body.model
    if not model.startswith(_EXTRA_PREFIX):
        model = f"{_EXTRA_PREFIX}{model}"
    deck["lemonade"].load(model)
    # Deliberate load: clear any suppression from a prior unload.
    deck["heal_suppressor"].clear()
    return {"status": "ok"}


@router.post("/lemonade/unload")
def lemonade_unload(body: LemonadeUnloadBody, request: Request) -> dict:
    deck = request.app.state.deck
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
    # ?force=true skips the conversation-guard, never the route guard.
    request.app.state.deck["hipfire"].park(force=force)
    return {"status": "ok"}


@router.post("/hipfire/resume")
def hipfire_resume(request: Request) -> dict:
    request.app.state.deck["hipfire"].resume()
    return {"status": "ok"}
