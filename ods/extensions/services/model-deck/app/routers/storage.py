"""
Storage router — locations, catalog, moves, pins, tiering policy.

Same conventions as the sibling routers: deps from request.app.state.deck,
GuardError/ValueError left to the app-wide handlers (409/422), no auth.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.engines import GuardError
from app.notify import notify_engine
from app.routers import build_world_snapshot
from app.events import log_event
from app.storage import plan_move

router = APIRouter(prefix="/storage", tags=["storage"])

_TERMINAL = frozenset({"done", "failed", "cancelled"})


class MoveBody(BaseModel):
    unit_id: str
    dest: str


class PinBody(BaseModel):
    pinned: bool


class LocationPatch(BaseModel):
    role: str | None = None
    watermark_gb: float | None = None
    archive_to: str | None = None
    readonly: bool | None = None


@router.get("/state")
def storage_state(request: Request) -> dict:
    deck = request.app.state.deck
    # A READ, never a disk walk: the UI polls this on a timer, and scanning
    # every location per poll re-stats every model file on every drive (and
    # rewrites catalog.json). Rescans belong to the storage watcher's tick,
    # POST /rescan, and the cold-model lookup in routers/control.py.
    return {"locations": deck["location_store"].describe(),
            "units": deck["catalog"].units(),
            "jobs": deck["job_queue"].jobs(),
            "policy": deck["storage_policy_store"].get()}


@router.post("/locations")
def register_location(spec: dict, request: Request) -> dict:
    return request.app.state.deck["location_store"].register(spec)


@router.put("/locations/{name}")
def update_location(name: str, patch: LocationPatch, request: Request) -> dict:
    # exclude_unset (not "not None"): watermark_gb/archive_to are nullable
    # fields the UI must be able to explicitly CLEAR ("empty = disabled").
    # Filtering None out would make an explicit null indistinguishable from
    # an omitted field and silently drop the clear.
    fields = patch.model_dump(exclude_unset=True)
    return request.app.state.deck["location_store"].update(name, fields)


@router.delete("/locations/{name}")
def deregister_location(name: str, request: Request) -> dict:
    request.app.state.deck["location_store"].deregister(name)
    return {"status": "ok"}


def submit_move(deck, unit_id: str, dest_name: str, label: str, on_success=None) -> dict:
    """Shared by the manual-move route and control.py's pull-through."""
    unit = deck["catalog"].get(unit_id)
    if unit is None:
        raise ValueError(f"unknown unit {unit_id!r}")
    described = {loc["name"]: loc for loc in deck["location_store"].describe()}
    dest = described.get(dest_name)
    if dest is None:
        raise ValueError(f"unknown location {dest_name!r}")
    # Plan/UX half of the destination-collision guard (the worker enforces the
    # other half at its choke point): refuse up front so the operator gets a
    # 409 instead of a job that fails minutes later, and so a same-named
    # archived copy is never a move away from being overwritten.
    if (Path(dest["path"]) / unit["relpath"]).exists():
        raise GuardError(
            f"destination {dest_name!r} already exists: {unit['relpath']!r} "
            "is already there — move or remove it first")
    world = build_world_snapshot(deck)
    active = frozenset(j["unit_id"] for j in deck["job_queue"].jobs()
                       if j["state"] not in _TERMINAL)
    plan = plan_move(unit, dest, world, active, dest["free_bytes"],
                     deck["settings"].storage_slack_bytes)
    return deck["job_queue"].submit(plan, label=label, on_success=on_success)


@router.post("/moves")
def create_move(body: MoveBody, request: Request) -> dict:
    deck = request.app.state.deck
    described = {loc["name"]: loc for loc in deck["location_store"].describe()}
    dest = described.get(body.dest)

    def after(job: dict) -> None:
        if dest and dest["engine"] != "none":
            warning = notify_engine(dest, deck)
            if warning:
                log_event(deck["events_path"], "storage_notify_deferred",
                          {"job": job["id"], "warning": warning})

    job = submit_move(deck, body.unit_id, body.dest, label="manual move", on_success=after)
    return {"job": job}


@router.get("/moves/{job_id}")
def get_move(job_id: str, request: Request) -> dict:
    job = request.app.state.deck["job_queue"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return job


@router.delete("/moves/{job_id}")
def cancel_move(job_id: str, request: Request) -> dict:
    queue = request.app.state.deck["job_queue"]
    if queue.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return {"cancelled": queue.cancel(job_id)}


@router.put("/units/{unit_id:path}")
def pin_unit(unit_id: str, body: PinBody, request: Request) -> dict:
    return request.app.state.deck["catalog"].set_pinned(unit_id, body.pinned)


@router.get("/policy")
def get_storage_policy(request: Request) -> dict:
    return request.app.state.deck["storage_policy_store"].get()


@router.put("/policy")
def put_storage_policy(policy: dict, request: Request) -> dict:
    store = request.app.state.deck["storage_policy_store"]
    store.put(policy)
    return store.get()


@router.post("/rescan")
def rescan(request: Request) -> dict:
    return {"units": len(request.app.state.deck["catalog"].scan())}
