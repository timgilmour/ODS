"""Lifecycle router — the operator escapes from automation.

``clear`` releases a quarantine (the UI's "try again" after fixing whatever
made two restores fail). ``adopt`` turns an ``unmanaged`` resource into a
managed one by recording what is ALREADY running as the intent — it changes
bookkeeping only and must never restart, load, or unload anything.
``expect-absence`` is the converse of adopt for the OTHER direction: an
actuator outside the deck (host-agent's dashboard activate path) announcing
that a resource is about to go away on purpose, so the reconciler does not
read a deliberate teardown as a death. It too actuates nothing.

Adopting an UNREACHABLE resource is refused (409) rather than recorded: an
observation we failed to make is not evidence of anything, and the record it
would write ("unloaded") is a park nobody asked for — the reconciler would
then correctly refuse to restore it forever.

No auth, matching the rest of the deck. Keys contain a slash
(``local/hipfire``), hence the ``:path`` converters.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StrictBool, StrictInt

from app.holds import DEFAULT_HOLD_TTL_S, MAX_HOLD_TTL_S
from app.observe import engine_for
from app.routers import build_observations, build_world_snapshot

router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])


class AutoBody(BaseModel):
    # StrictBool, not bool: pydantic's lax mode coerces "yes"/"on"/1 to True,
    # and a safety brake should refuse an ambiguous value rather than guess
    # which way the operator meant it. Matches policy.py's strict validation.
    enabled: StrictBool


class ExpectAbsenceBody(BaseModel):
    # StrictInt for the same reason AutoBody uses StrictBool: pydantic's lax
    # mode would turn "60" or True into a duration, and a window during which
    # the reconciler stands down is not something to guess at. Bounded at
    # both ends — a hold is an announcement, not an off switch; that is what
    # POST /lifecycle/auto is for.
    ttl_s: StrictInt = Field(default=int(DEFAULT_HOLD_TTL_S),
                             gt=0, le=int(MAX_HOLD_TTL_S))


@router.get("/auto")
def get_auto(request: Request) -> dict:
    """Whether the reconciler may act."""
    return {"enabled": request.app.state.deck["policy_store"].auto_enabled()}


@router.post("/auto")
def set_auto(body: AutoBody, request: Request) -> dict:
    """Turn lifecycle automation on or off.

    The brake. Without it the reconciler's only off switch is hand-editing
    ``policy.json`` and restarting the container — an unacceptable answer for
    a component that starts and stops real engines, and the reason this route
    exists at all. ``PUT /api/policy`` deliberately rejects the reserved
    ``_auto`` key, so the toggle needs its own surface rather than riding on
    the tenant policy payload.

    Off does NOT unload anything: it stops the Deck acting on its own, and
    leaves every resource exactly as it is.
    """
    request.app.state.deck["policy_store"].set_auto(body.enabled)
    return {"enabled": body.enabled}


@router.post("/quarantine/{key:path}/clear")
def clear_quarantine(key: str, request: Request) -> dict:
    """Release `key`'s quarantine so the reconciler will try again."""
    store = request.app.state.deck["intent_store"]
    if key not in store.get():
        raise HTTPException(status_code=404, detail=f"no intent for {key!r}")
    store.clear_failures(key)
    return {"key": key, "quarantined": False}


@router.post("/expect-absence/{key:path}")
def expect_absence(key: str, body: ExpectAbsenceBody, request: Request) -> dict:
    """Announce that `key` is about to go away deliberately.

    Suppresses restore for `key` until the TTL expires or someone releases
    it. Records nothing and actuates nothing: this is a statement about the
    near future, not about what the resource IS.

    Unlike ``adopt`` this does NOT validate that the key is known — the whole
    point is to be called while the resource is mid-teardown, when an
    observation would be unreliable or absent. An unknown key simply holds
    nothing the reconciler was going to act on.
    """
    request.app.state.deck["hold_store"].hold(key, float(body.ttl_s))
    return {"key": key, "held": True, "ttl_s": body.ttl_s}


@router.delete("/expect-absence/{key:path}")
def clear_expect_absence(key: str, request: Request) -> dict:
    """End `key`'s announced absence early.

    Idempotent: releasing a key that is not held is a success, because the
    caller's goal — "the reconciler may act on this again" — is already true.
    That matters because the bracket's rollback path calls this exactly when
    it cannot know whether the hold survived.
    """
    released = request.app.state.deck["hold_store"].release(key)
    return {"key": key, "held": False, "released": released}


@router.post("/adopt/{key:path}")
def adopt(key: str, request: Request) -> dict:
    """Record the observed state of `key` as its intent. Actuates nothing."""
    deck = request.app.state.deck
    observed = build_observations(deck, build_world_snapshot(deck))

    record = observed.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown resource {key!r}")

    swap_ids = frozenset(
        n["id"] for n in deck["node_store"].list()
        if n.get("control") == "swap")
    # E1 T13 fix: thread the LIVE local declaration through so a resource
    # not literally named lemonade/comfyui/hipfire can adopt too — mirrors
    # app.routers.control._declared_kind / app.routers.sets._declared_kinds
    # (same "read node_store fresh, never a boot-time copy" posture) and
    # closes the local-side half of the resource==kind-name-coincidence bug
    # test_adopt_a_swap_nodes_slot (N1 T12) already fixed for swap nodes —
    # see app.observe._LOCAL_ENGINE_BY_KEY's own docstring.
    local = deck["node_store"].get("local")
    local_kinds = {e["resource"]: e["kind"] for e in (local.get("engines", []) if local else [])}
    engine = engine_for(key, swap_ids, local_kinds)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"no engine owns {key!r}")

    if not record["reachable"]:
        raise HTTPException(
            status_code=409,
            detail=f"{key!r} did not answer; adopting would record a state "
                   "nobody observed")

    deck["intent_store"].record(
        key,
        state="loaded" if record["loaded"] else "unloaded",
        model=record["model"],
        engine=engine,
    )
    # Adoption is the announced window closing: the deck now knows what is
    # actually running, so there is nothing left to protect from the
    # reconciler. Deliberately AFTER record() and after the 409 above — a
    # refused adopt concluded nothing and must leave the hold standing.
    deck["hold_store"].release(key)
    return {"key": key, "adopted": record}
