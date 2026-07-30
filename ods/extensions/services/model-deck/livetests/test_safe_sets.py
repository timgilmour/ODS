"""S13–S17: set CRUD, reserved-slot protection, preview purity, no-op apply,
and the apply→revert round-trip (ephemeral-only; asserts the plan is free of
hipfire/activate steps BEFORE applying — safe-tier guarantee)."""

import pytest

from clients import EXTRA
from test_safe_lemonade import load_drill

pytestmark = pytest.mark.safe

FORBIDDEN_STEPS = {"park_hipfire", "resume_hipfire", "activate"}


@pytest.fixture
def probe_set(deck):
    """A throwaway set slug, deleted afterwards."""
    name = "deck-drill probe"
    yield name, "deck-drill-probe"
    deck.delete("/api/sets/deck-drill-probe")


def _current_ephemeral(deck) -> dict:
    world = deck.get("/api/state").json()["world"]
    lem = world["tenants"]["lemonade"]["state"]
    hip = world["tenants"]["hipfire"]["state"]
    return {
        "lemonade": {"state": "loaded" if lem == "loaded" else "unloaded"},
        "comfyui": {"state": "leave"},
        "hipfire": {"state": "running" if hip in ("running", "loading") else "parked"},
    }


def test_s13_set_crud(deck, probe_set):
    name, slug = probe_set
    body = {"name": name, "ephemeral": {"comfyui": {"state": "leave"}}}
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
        "name": name, "ephemeral": {"lemonade": {"state": "unloaded"}}})

    before = lemonade_direct.loaded()
    preview = deck.post(f"/api/sets/{slug}/preview").json()
    steps = [s["step"] for s in preview["steps"]]
    assert steps == ["unload_lemonade"]
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
        "name": name, "ephemeral": {"lemonade": {"state": "unloaded"}}})

    plan = deck.post(f"/api/sets/{slug}/preview").json()["steps"]
    assert FORBIDDEN_STEPS.isdisjoint({s["step"] for s in plan}), \
        f"safe-tier set plan contains disruptive steps: {plan}"

    report = deck.post(f"/api/sets/{slug}/apply").json()
    assert report["failed"] is None
    assert [s["step"] for s in report["completed"]] == ["unload_lemonade"]
    assert lemonade_direct.loaded() is None

    # _previous records load STATE, not model identity: its load step resolves
    # to the durable route model (sets cannot express "an off-route model was
    # resident"). So the revert reloads the ROUTE model, or the drill model
    # only when no route is set.
    route = deck.get("/api/state").json()["world"]["default_route"]
    revert = deck.post("/api/sets/_previous/apply", timeout=240.0).json()
    assert revert["failed"] is None
    expected = route or f"{EXTRA}{drill_model}"
    assert lemonade_direct.loaded() == expected, \
        f"revert did not reload the route model; report={revert}"
