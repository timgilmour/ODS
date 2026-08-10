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

``apply`` also records the resulting intent (``_record_intent`` below) for
each step that actually completed — a set apply is as deliberate as a button
press on the control routes, and the reconciler must see it the same way.

Task 9 (2026-08-07): every saved set now carries the ENTIRE settings store,
captured at save time — ``create_set`` always re-stamps ``settings_snapshot``
from the live store, ignoring whatever the client sent (save means "snapshot
NOW"). ``settings_diff``/``adopt_set`` below let a human inspect and
reconcile drift between a saved snapshot and the live store without a full
re-apply; ``preview``/``apply`` both now read the live store fresh (same
"never reuse a stale snapshot" posture as ``World`` above) so a
``restore_settings`` step shows up wherever it's actually going to fire.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from app.routers import build_world_snapshot
from app.sets import (
    PREVIOUS_NAME,
    RESERVED_SLUG,
    ConfigSet,
    adopt_selective,
    diff_snapshot,
    plan_apply,
    slugify,
)
from app.sets import apply as sets_apply

router = APIRouter(prefix="/sets", tags=["sets"])

_ADOPT_MODES = frozenset({"current", "selective"})


class AdoptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    # Each entry is the {"scope", "key"} shape diff_snapshot emits — plain
    # dicts, not a stricter sub-model, so a malformed entry fails loudly via
    # adopt_selective's own ValueError (-> 422) rather than FastAPI's
    # differently-shaped request-validation error.
    keys: list[dict] = []

# preview()'s duration estimate, seconds per step kind. Unlisted step kinds
# (unload_lemonade, free_comfyui, park_hipfire, load_lemonade, policy_patch)
# fall back to _DEFAULT_DURATION.
_STEP_DURATIONS = {"activate": 120, "resume_hipfire": 180, "warn": 0}
_DEFAULT_DURATION = 5


@router.get("")
def list_sets(request: Request) -> dict:
    store = request.app.state.deck["set_store"]
    all_sets = store.list()
    previous = next((cfgset for cfgset in all_sets if cfgset.name == PREVIOUS_NAME), None)
    user_sets = [cfgset for cfgset in all_sets if cfgset.name != PREVIOUS_NAME]
    # unreadable() [c44]: one bad file must not down this route (list()
    # already skips it) — additive field, surfaced so a recovery UI can
    # offer to delete it; unknown fields are ignored by older clients.
    return {"sets": user_sets, "previous": previous, "unreadable": store.unreadable()}


@router.post("")
def create_set(cfgset: ConfigSet, request: Request, overwrite: bool = False) -> dict:
    deck = request.app.state.deck
    store = deck["set_store"]
    slug = slugify(cfgset.name)
    if not overwrite and store.get(slug) is not None:
        raise HTTPException(status_code=409, detail=f"set {slug!r} already exists")
    # Save = "snapshot NOW" (Task 9, design decision 6): the live settings
    # store, ALWAYS — never whatever the client sent, even a set that
    # round-tripped through the UI carrying its own (now-stale) snapshot.
    # A saved set is a whole-store recipe captured at save time, not
    # something a client gets to author directly.
    cfgset = cfgset.model_copy(update={"settings_snapshot": deck["settings_store"].get()})
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
    try:
        missing = store.get(slug) is None
    except ValueError:
        # Present but corrupt [c44]: exactly the file this recovery path
        # exists to remove — DELETE must not itself fail on the invalid
        # JSON it's here to clear out.
        missing = False
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")
    store.delete(slug)
    return {"status": "ok"}


@router.get("/{slug}/settings-diff")
def settings_diff(slug: str, request: Request) -> dict:
    """Diff a saved set's captured settings_snapshot against the live
    settings store. ``has_snapshot`` lives HERE, not inside diff_snapshot
    (Task 9): an old set (no snapshot) diffs as empty for the same reason a
    set that IS identical to the live store does, so the caller needs the
    extra bit to tell "nothing to compare" from "compared, no drift"."""
    deck = request.app.state.deck
    cfgset = deck["set_store"].get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")

    diff = diff_snapshot(cfgset.settings_snapshot, deck["settings_store"].get())
    return {**diff, "has_snapshot": cfgset.settings_snapshot is not None}


@router.post("/{slug}/adopt")
def adopt_set(slug: str, body: AdoptRequest, request: Request) -> ConfigSet:
    """Update the SAVED set's settings_snapshot without a full re-apply.
    ``mode="current"`` is a full re-stamp (identical to what saving does);
    ``mode="selective"`` takes only the named diff entries (``body.keys``,
    the {"scope", "key"} shape settings-diff emits) from the live store."""
    deck = request.app.state.deck
    store = deck["set_store"]
    cfgset = store.get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")
    if body.mode not in _ADOPT_MODES:
        raise ValueError(f"unknown adopt mode {body.mode!r}; expected one of {sorted(_ADOPT_MODES)}")

    current = deck["settings_store"].get()
    new_snapshot = (
        current if body.mode == "current"
        else adopt_selective(cfgset.settings_snapshot, current, body.keys)
    )
    updated = cfgset.model_copy(update={"settings_snapshot": new_snapshot})
    # replace(), not save(): must write back to the slug it read FROM.
    # save() derives the slug from the NAME, and slugify('· previous') ==
    # 'previous' collides with the reserved slug save() refuses [c50] — the
    # one set adopt is most useful on (reconciling · previous's settings
    # snapshot) would otherwise 422.
    store.replace(slug, updated)
    return updated


@router.post("/{slug}/preview")
def preview_set(slug: str, request: Request) -> dict:
    deck = request.app.state.deck
    cfgset = deck["set_store"].get(slug)
    if cfgset is None:
        raise HTTPException(status_code=404, detail=f"unknown set {slug!r}")

    world = build_world_snapshot(deck)
    steps = plan_apply(cfgset, world, settings_now=deck["settings_store"].get())
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
    report = sets_apply(
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
        settings_now=deck["settings_store"].get(),
        settings_store=deck["settings_store"],
    )
    _record_intent(deck, report)
    return report


# How a completed apply step translates to an intent record. A set apply is
# as deliberate as a button press, so it must leave the same last-known-good
# record the control routes do — otherwise "load it via a set" would be
# invisible to the reconciler.
#
# Only ``report["completed"]`` is walked: a failed or never-reached step did
# not happen, and intent is last-known-GOOD. free_comfyui and policy_patch
# are deliberately absent — /free leaves the server up and still observing
# as loaded, so an "unloaded" intent would derive as permanent
# ``unexpected``; a policy patch touches no engine state at all.
_STEP_INTENT = {
    "load_lemonade": ("local/lemonade", "lemonade", "loaded"),
    "unload_lemonade": ("local/lemonade", "lemonade", "unloaded"),
    "resume_hipfire": ("local/hipfire", "hipfire", "loaded"),
    "park_hipfire": ("local/hipfire", "hipfire", "unloaded"),
}


def _record_intent(deck: dict, report: dict) -> None:
    """Record every completed step of an apply as intent, in plan order."""
    store = deck["intent_store"]
    for step in report["completed"]:
        mapping = _STEP_INTENT.get(step["step"])
        if mapping is None:
            continue
        key, engine, state = mapping
        # Only a load names a model. hipfire is single-model and the Deck
        # does not choose its model, so it records None ("loaded, no opinion
        # which model") rather than a name it cannot observe.
        model = step["model"] if state == "loaded" and "model" in step else None
        store.record(key, state=state, model=model, engine=engine)
