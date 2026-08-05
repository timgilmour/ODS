"""
Status router — read-only observability endpoints: a live world/policy/
registry/lifecycle snapshot (GET /state) and the arbiter's append-only audit
trail (GET /events). No auth: these expose no controls, only what's already
visible from a live box.
"""

from fastapi import APIRouter, Request

from app.events import tail_events
from app.routers import build_lifecycle_view, build_world_snapshot

router = APIRouter(tags=["status"])


@router.get("/state")
def get_state(request: Request) -> dict:
    deck = request.app.state.deck
    world = build_world_snapshot(deck)
    return {
        # Node identity. The deck has exactly one local node today; the key
        # matches app.observe's _LOCAL_NODE prefix so the UI adapter can join
        # this against lifecycle keys like "local/hipfire" without a mapping.
        "node": {"id": "local", "label": deck["settings"].node_label},
        "world": world,
        "policy": deck["policy_store"].get(),
        "models": deck["registry"].scan(),
        # intent x observation, derived from the SAME snapshot the world
        # block reports, so the two can never describe different moments.
        "lifecycle": build_lifecycle_view(deck, world),
    }


@router.get("/events")
def get_events(request: Request, n: int = 100) -> dict:
    deck = request.app.state.deck
    return {"events": tail_events(deck["events_path"], n)}
