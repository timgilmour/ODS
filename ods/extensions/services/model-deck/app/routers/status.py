"""
Status router — read-only observability endpoints: a live world/policy/
registry snapshot (GET /state) and the arbiter's append-only audit trail
(GET /events). No auth: these expose no controls, only what's already
visible from a live box.
"""

from fastapi import APIRouter, Request

from app.events import tail_events
from app.routers import build_world_snapshot

router = APIRouter(tags=["status"])


@router.get("/state")
def get_state(request: Request) -> dict:
    deck = request.app.state.deck
    return {
        "world": build_world_snapshot(deck),
        "policy": deck["policy_store"].get(),
        "models": deck["registry"].scan(),
    }


@router.get("/events")
def get_events(request: Request, n: int = 100) -> dict:
    deck = request.app.state.deck
    return {"events": tail_events(deck["events_path"], n)}
