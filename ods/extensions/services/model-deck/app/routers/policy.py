"""
Policy router — read/replace the arbiter's per-tenant priority/pinned/
idle_ttl policy. GET is open; PUT is admin-gated and delegates its full
shape/type validation to ``PolicyStore.put`` (raises ``ValueError`` on a bad
payload, mapped to 422 by ``app.main``'s app-wide exception handler) rather
than re-validating here.
"""

from fastapi import APIRouter, Depends, Request

from app.security import require_admin

router = APIRouter(prefix="/policy", tags=["policy"])


@router.get("")
def get_policy(request: Request) -> dict:
    return request.app.state.deck["policy_store"].get()


@router.put("", dependencies=[Depends(require_admin)])
def put_policy(body: dict, request: Request) -> dict:
    store = request.app.state.deck["policy_store"]
    store.put(body)
    return store.get()
