"""
Sets router — CRUD over saved config sets, plus preview (pure ``plan_apply``,
no execution) and apply (executes against the live box, serialized under
``app.sets``' module lock). No auth on any route (admin gate deliberately
removed 2026-07-22 — ops-first; LAN path still behind Authelia).

Slug lookups that miss return 404. The reserved ``_previous`` revert slot
(written only by ``apply()``) is readable through GET like any other set,
but can never be created (``SetStore.save`` rejects it — surfaces as 422 via
the app-wide ``ValueError`` handler) or deleted (explicit 403 below) through
this API.

``apply`` builds its ``World`` snapshot *inside* this endpoint, immediately
before calling ``sets.apply()`` — never reusing one from ``/api/state`` or
``preview`` — because reality may have moved between requests and applying
against a stale snapshot could evict/load the wrong thing. It's a sync
``def`` on purpose: FastAPI runs sync endpoints in a threadpool, which is
what lets a slow/blocked apply (serialized under ``app.sets``' lock) not
stall the event loop.
"""

from fastapi import APIRouter, HTTPException, Request

from app.routers import build_world_snapshot
from app.sets import PREVIOUS_NAME, RESERVED_SLUG, ConfigSet, plan_apply, slugify
from app.sets import apply as sets_apply

router = APIRouter(prefix="/sets", tags=["sets"])

# preview()'s duration estimate, seconds per step kind. Unlisted step kinds
# (unload_lemonade, free_comfyui, park_hipfire, load_lemonade, policy_patch)
# fall back to _DEFAULT_DURATION.
_STEP_DURATIONS = {"activate": 120, "resume_hipfire": 180, "warn": 0}
_DEFAULT_DURATION = 5


@router.get("")
def list_sets(request: Request) -> dict:
    all_sets = request.app.state.deck["set_store"].list()
    previous = next((cfgset for cfgset in all_sets if cfgset.name == PREVIOUS_NAME), None)
    user_sets = [cfgset for cfgset in all_sets if cfgset.name != PREVIOUS_NAME]
    return {"sets": user_sets, "previous": previous}


@router.post("")
def create_set(cfgset: ConfigSet, request: Request, overwrite: bool = False) -> dict:
    store = request.app.state.deck["set_store"]
    slug = slugify(cfgset.name)
    if not overwrite and store.get(slug) is not None:
        raise HTTPException(status_code=409, detail=f"set {slug!r} already exists")
    saved_slug = store.save(cfgset)
    return {"slug": saved_slug}


@router.get("/{slug}")
def get_set(slug: str, request: Request) -> ConfigSet:
    cfgset = request.app.state.deck["set_store"].get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")
    return cfgset


@router.delete("/{slug}")
def delete_set(slug: str, request: Request) -> dict:
    if slug == RESERVED_SLUG:
        raise HTTPException(status_code=403, detail="cannot delete the reserved revert snapshot")
    store = request.app.state.deck["set_store"]
    if store.get(slug) is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")
    store.delete(slug)
    return {"status": "ok"}


@router.post("/{slug}/preview")
def preview_set(slug: str, request: Request) -> dict:
    deck = request.app.state.deck
    cfgset = deck["set_store"].get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")

    world = build_world_snapshot(deck)
    steps = plan_apply(cfgset, world)
    estimate_s = sum(_STEP_DURATIONS.get(step["step"], _DEFAULT_DURATION) for step in steps)
    return {"steps": steps, "estimate_s": estimate_s}


@router.post("/{slug}/apply")
def apply_set(slug: str, request: Request, force: bool = False) -> dict:
    # ?force=true skips the hipfire conversation-guard (an operator
    # overriding an abandoned conversation); mirrors create_set's
    # ?overwrite= convention.
    deck = request.app.state.deck
    cfgset = deck["set_store"].get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")

    world = build_world_snapshot(deck)
    return sets_apply(
        cfgset,
        world=world,
        force=force,
        lemonade=deck["lemonade"],
        comfy=deck["comfy"],
        hipfire=deck["hipfire"],
        hostagent=deck["hostagent"],
        policy_store=deck["policy_store"],
        store=deck["set_store"],
        events_path=deck["events_path"],
        heal_suppressor=deck["heal_suppressor"],
        catalog=deck["catalog"],
    )
