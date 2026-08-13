"""
Policy router — read/replace the arbiter's per-resource priority/pinned/
idle_ttl policy. No auth (admin gate deliberately removed 2026-07-22).
PUT delegates its full shape/type validation to ``PolicyStore.put`` (raises
``ValueError`` on a bad payload, mapped to 422 by ``app.main``'s app-wide
exception handler) rather than re-validating here.

PUT is refused for undeclared resources: a resource must be declared in the
node's engines[] before its policy can be set, matching the unknown-tenant
rejection idiom.
"""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/policy", tags=["policy"])


@router.get("")
def get_policy(request: Request) -> dict:
    return request.app.state.deck["policy_store"].get()


@router.put("")
def put_policy(body: dict, request: Request) -> dict:
    deck = request.app.state.deck
    store = deck["policy_store"]

    # Refuse PUTs for undeclared resources (unknown-tenant rejection idiom)
    declared = store.get()  # Returns only declared resources
    unknown = sorted(set(body) - set(declared) - {"_auto"})
    if unknown:
        raise ValueError(f"unknown resource(s): {', '.join(unknown)}")

    store.put(body)
    return store.get()
