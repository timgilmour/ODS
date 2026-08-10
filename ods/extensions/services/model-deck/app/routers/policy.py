"""
Policy router — read/replace the arbiter's per-tenant priority/pinned/
idle_ttl policy. No auth (admin gate deliberately removed 2026-07-22).
PUT delegates its full shape/type validation to ``PolicyStore.put`` (raises
``ValueError`` on a bad payload, mapped to 422 by ``app.main``'s app-wide
exception handler) rather than re-validating here.
"""

from fastapi import APIRouter, Request

from app.events import log_event
from app.policy import DEFAULT_POLICIES

router = APIRouter(prefix="/policy", tags=["policy"])


@router.get("")
def get_policy(request: Request) -> dict:
    return request.app.state.deck["policy_store"].get()


@router.put("")
def put_policy(body: dict, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["policy_store"]
    store.put(body)
    unknown = sorted(set(body) - set(DEFAULT_POLICIES))
    if unknown:
        # Deliberately accepted (defaults are seed data, not an allowlist —
        # policy.py:103-105); the event is the feedback that was lost when
        # the pre-1ee64611 rejection was removed: a typo'd tenant name now
        # shows up in Events instead of silently policying nothing.
        log_event(deck["events_path"], "policy-unknown-tenant", {"tenants": unknown})
    return store.get()
