"""S13–S17: set CRUD, reserved-slot protection, preview purity, no-op apply,
and the apply→revert round-trip (ephemeral-only; asserts the plan is free of
hipfire/activate steps BEFORE applying — safe-tier guarantee)."""

import pytest

from clients import EXTRA
from test_safe_lemonade import load_drill

pytestmark = pytest.mark.safe

# E1 T13 fix (ledgered obligation, Task 8's review): plan_apply's step
# vocabulary went verb-generic, resource-keyed ("park_hipfire"/
# "resume_hipfire" compound names are gone — app/sets.py:56-69's own
# docstring) — a step is now {"step": <verb>, "resource": <name>, ...}
# (app/sets.py:693 for "unload", the same shape every other verb uses).
# The old {"park_hipfire", "resume_hipfire", "activate"} literal set could
# therefore never match anything post-migration ("park_hipfire" never
# appears as a bare `s["step"]` value again) — HALF-BLIND, silently always
# passing regardless of what the plan actually contained. Fixed to the bare
# VERBS plan_apply can actually emit for park/resume: today those are
# hipfire-kind's ONLY human_verbs() (app/engine_kinds.py's _HipfireAdapter),
# so on THIS deployment "park"/"resume" appearing in a plan still means
# "touches hipfire" exactly as before — just checked as the real string that
# shows up now, not one that can't. "activate" (app/sets.py:651, box-wide,
# no resource) is unchanged; it was never a compound name.
#
# RE-CHECKED at the fourth kind (sglang-omni Task 7 — the first kind added
# since this guard was written): it declares human_verbs {"load","unload"}
# and nothing else, so park/resume are STILL hipfire's alone and this set's
# meaning is unwidened. A future kind declaring park or resume would
# silently make this guard mean "touches hipfire OR that kind" — check this
# comment when adding one.
FORBIDDEN_STEPS = {"park", "resume", "activate"}


@pytest.fixture
def probe_set(deck):
    """A throwaway set slug, deleted afterwards."""
    name = "deck-drill probe"
    yield name, "deck-drill-probe"
    deck.delete("/api/sets/deck-drill-probe")


def _current_ephemeral(deck) -> dict:
    """A genuine no-op ephemeral body against CURRENT state, in the E1
    schema (app/sets.py:150-157: ``Ephemeral = {resources: {resource:
    ResourceDesired}}``, ``desired`` in {"loaded","unloaded","parked",
    "freed"} — app/sets.py:129-134's ``_DESIRED_VERBS`` table — never the
    pre-Task-8 top-level lemonade/comfyui/hipfire sub-sections with a
    "state" field). comfyui is OMITTED rather than given a "leave" value:
    under the new schema omission itself IS "leave" (same module,
    app/sets.py:247's comment on the legacy-upgrade table) — there is no
    "leave" desired value to spell.

    "loaded" works uniformly for both lemonade (verb "load") and hipfire
    (verb "resume") — app/sets.py:129-130 maps "loaded" to
    {"load","resume"} — so hipfire currently running needs no different
    desired value than lemonade currently loaded; only a currently-PARKED
    hipfire needs "parked" instead, so plan_apply's park/resume checks
    (app/sets.py:712 and 748-749) see a real no-op either way.
    """
    world = deck.get("/api/state").json()["world"]
    lem = world["tenants"]["lemonade"]["state"]
    hip = world["tenants"]["hipfire"]["state"]
    return {
        "resources": {
            "lemonade": {"desired": "loaded" if lem == "loaded" else "unloaded"},
            "hipfire": {"desired": "loaded" if hip in ("running", "loading") else "parked"},
        }
    }


def test_s13_set_crud(deck, probe_set):
    name, slug = probe_set
    body = {"name": name, "ephemeral": {"resources": {}}}
    assert deck.post("/api/sets", json=body).json()["slug"] == slug
    assert deck.post("/api/sets", json=body).status_code == 409          # duplicate
    assert deck.post("/api/sets", params={"overwrite": "true"}, json=body).status_code == 200
    assert deck.get(f"/api/sets/{slug}").json()["name"] == name
    assert deck.delete(f"/api/sets/{slug}").status_code == 200
    assert deck.get(f"/api/sets/{slug}").status_code == 404


def test_s14_reserved_revert_slot(deck):
    resp = deck.post("/api/sets", json={"name": "previous"})
    assert resp.status_code == 422, resp.text
    assert deck.delete("/api/sets/_previous").status_code in (403, 404)  # 404 = never applied


def test_s15_preview_is_pure(deck, lemonade_direct, lemonade_guard, drill_model, probe_set):
    name, slug = probe_set
    load_drill(deck, drill_model)
    deck.post("/api/sets", params={"overwrite": "true"}, json={
        "name": name,
        "ephemeral": {"resources": {"lemonade": {"desired": "unloaded"}}}})

    before = lemonade_direct.loaded()
    preview = deck.post(f"/api/sets/{slug}/preview").json()
    steps = [s["step"] for s in preview["steps"]]
    # Verb-generic since E1 Task 8 — was "unload_lemonade" (app/sets.py:693
    # builds {"step": "unload", "resource": "lemonade", ...} now).
    assert steps == ["unload"]
    assert preview["estimate_s"] > 0
    assert lemonade_direct.loaded() == before, "preview mutated the box"


def test_s16_noop_apply_still_captures_previous(deck, probe_set):
    name, slug = probe_set
    deck.post("/api/sets", params={"overwrite": "true"},
              json={"name": name, "ephemeral": _current_ephemeral(deck)})
    report = deck.post(f"/api/sets/{slug}/apply").json()
    assert report["failed"] is None and report["completed"] == []
    previous = deck.get("/api/sets/_previous")
    assert previous.status_code == 200
    assert previous.json()["name"] == "· previous"


def test_s17_apply_and_previous_revert_roundtrip(deck, lemonade_direct, lemonade_guard,
                                                 drill_model, probe_set):
    name, slug = probe_set
    load_drill(deck, drill_model)
    deck.post("/api/sets", params={"overwrite": "true"}, json={
        "name": name,
        "ephemeral": {"resources": {"lemonade": {"desired": "unloaded"}}}})

    plan = deck.post(f"/api/sets/{slug}/preview").json()["steps"]
    assert FORBIDDEN_STEPS.isdisjoint({s["step"] for s in plan}), \
        f"safe-tier set plan contains disruptive steps: {plan}"

    report = deck.post(f"/api/sets/{slug}/apply").json()
    assert report["failed"] is None
    # app/sets.py:1132's apply() appends the step dict itself to "completed"
    # — same verb-generic shape as preview's steps above.
    assert [s["step"] for s in report["completed"]] == ["unload"]
    assert lemonade_direct.loaded() is None

    # _previous records the model that was ACTUALLY resident (max-review #11,
    # `LemonadeEphemeral.model` — plan_apply's precedence is declared-explicit
    # over durable over route), so the revert reloads the DRILL model even
    # when it is off-route. The pre-fix contract asserted here — "revert
    # reloads the ROUTE model" — was the honest-revert bug itself.
    revert = deck.post("/api/sets/_previous/apply", timeout=240.0).json()
    assert revert["failed"] is None
    assert lemonade_direct.loaded() == f"{EXTRA}{drill_model}", \
        f"revert did not reload the actually-resident model; report={revert}"
