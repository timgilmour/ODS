"""Lifecycle router — the two operator escapes from automation.

``clear`` releases a quarantine (the UI's "try again" after fixing whatever
made two restores fail). ``adopt`` turns an ``unmanaged`` resource into a
managed one by recording what is ALREADY running as the intent — it changes
bookkeeping only and must never restart, load, or unload anything. Adoption
that actuates would make "start managing this" a dangerous button, and
nobody would press it.

Adopting an UNREACHABLE resource is refused (409) rather than recorded: an
observation we failed to make is not evidence of anything, and the record it
would write ("unloaded") is a park nobody asked for — the reconciler would
then correctly refuse to restore it forever.

No auth, matching the rest of the deck. Keys contain a slash
(``local/hipfire``), hence the ``:path`` converters.
"""

from fastapi import APIRouter, HTTPException, Request

from app.observe import engine_for
from app.routers import build_observations, build_world_snapshot

router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])


@router.post("/quarantine/{key:path}/clear")
def clear_quarantine(key: str, request: Request) -> dict:
    """Release `key`'s quarantine so the reconciler will try again."""
    store = request.app.state.deck["intent_store"]
    if key not in store.get():
        raise HTTPException(status_code=404, detail=f"no intent for {key!r}")
    store.clear_failures(key)
    return {"key": key, "quarantined": False}


@router.post("/adopt/{key:path}")
def adopt(key: str, request: Request) -> dict:
    """Record the observed state of `key` as its intent. Actuates nothing."""
    deck = request.app.state.deck
    observed = build_observations(deck, build_world_snapshot(deck))

    record = observed.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown resource {key!r}")

    engine = engine_for(key)
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
    return {"key": key, "adopted": record}
