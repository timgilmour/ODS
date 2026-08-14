"""D2 + D4: durable route flips via host-agent activate, and the halt report.
Catalog ids are auto-discovered from dashboard-api; DECK_DRILL_ACTIVATE_IDS
("<current>,<target>") overrides."""

import os

import pytest

from clients import EXTRA
from test_disruptive_hipfire import hipfire_window  # noqa: F401 — fixture reuse

pytestmark = pytest.mark.disruptive


@pytest.fixture
def route_ids(deck, catalog, drill_model):
    """(current_route_model, current_id, target_id) — skip if unresolvable."""
    override = os.environ.get("DECK_DRILL_ACTIVATE_IDS", "")
    route = deck.get("/api/state").json()["world"]["default_route"]
    if not route:
        pytest.skip("no default route configured")
    if override:
        current_id, target_id = override.split(",")
        return route, current_id, target_id
    current_id = catalog.id_for_gguf(route.removeprefix(EXTRA))
    target_id = catalog.id_for_gguf(drill_model)
    if not current_id or not target_id:
        pytest.skip(f"catalog ids unresolved (current={current_id}, target={target_id}); "
                    "pass DECK_DRILL_ACTIVATE_IDS=<current>,<target>")
    return route, current_id, target_id


def _flip_set(deck, name: str, model_file: str, model_id: str) -> dict:
    deck.post("/api/sets", params={"overwrite": "true"}, json={
        "name": name,
        "durable": {"default_route_model": f"{EXTRA}{model_file}",
                    "activate_model_id": model_id},
        # E1 (app/sets.py:150-157): Ephemeral is {resources: {resource:
        # {desired, model?}}} — the pre-Task-8 top-level lemonade/comfyui/
        # hipfire shape with a "state" field 422s under ConfigSet/Ephemeral
        # (both extra="forbid", no legacy upgrade on the create_set POST
        # path — upgrade_legacy_set only runs on LOAD of an old on-disk file).
        "ephemeral": {"resources": {"lemonade": {"desired": "loaded"}}},
    }).raise_for_status()
    # force=true: the D1 probe warmed hipfire's tracker; the veto is expected
    # and deliberately overridden inside the window.
    slug = name.lower().replace(" ", "-")
    report = deck.post(f"/api/sets/{slug}/apply", params={"force": "true"},
                       timeout=700.0).json()
    deck.delete(f"/api/sets/{slug}")
    return report


def test_d2_route_flip_roundtrip(deck, litellm_direct, lemonade_direct,
                                 drill_model, route_ids, hipfire_window):
    route, current_id, target_id = route_ids

    report = _flip_set(deck, "deck drill flip", drill_model, target_id)
    assert report["failed"] is None, report
    assert litellm_direct.route_table()["default"].endswith(f"{EXTRA}{drill_model}")
    assert litellm_direct.completion("default")
    assert lemonade_direct.loaded() == f"{EXTRA}{drill_model}"

    restore = _flip_set(deck, "deck drill restore", route.removeprefix(EXTRA), current_id)
    assert restore["failed"] is None, restore
    assert litellm_direct.route_table()["default"].endswith(route)
    assert litellm_direct.completion("default")


def test_d4_apply_halts_with_exact_report_on_bad_activate(deck, events, hipfire_window):
    deck.post("/api/sets", params={"overwrite": "true"}, json={
        "name": "deck drill bogus",
        "durable": {"default_route_model": "extra.__deck_drill_no_such__.gguf",
                    "activate_model_id": "deck-drill-bogus-id"},
    }).raise_for_status()
    try:
        report = deck.post("/api/sets/deck-drill-bogus/apply",
                           params={"force": "true"}, timeout=700.0).json()
    finally:
        deck.delete("/api/sets/deck-drill-bogus")

    assert report["failed"] is not None and report["failed"]["step"] == "activate"
    assert report["error"], report
    assert report["completed"] == []  # plan was [activate] only; nothing mutated
    events.expect("apply-end", timeout=10)
