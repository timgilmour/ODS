"""Tests for the alias -> identity rename migration PLANNER (Plan C2, Task
10).

Two layers: ``app.rename.plan_rename`` is a pure function (no I/O, never
mutates its inputs) — the first block of tests below drives it directly.
``POST /api/rename/plan`` is the thin, read-only route that gathers real
vLLM profiles (adopt-sweep machinery) and real litellm routes, then hands
them to ``plan_rename`` unchanged — the second block proves the route
never writes anywhere (a second identical call returns the same plan) and
handles the no-spark-configured case the same way every other spark-backed
route does.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.rename import plan_rename
from tests.test_api import make_app, wire_swap_node

# ===========================================================================
# plan_rename — pure function
# ===========================================================================


def test_multi_alias_profile_collapses_to_one_identity():
    profiles = {"mm27b": {"served_model_name": ["aeon", "aeon-fast", "aeon-deep"],
                          "identity": "Qwen3.5-27B-mm-modelopt"}}
    plan = plan_rename(profiles, routes={}, client_pins={})
    assert plan["renames"] == [{
        "profile": "mm27b",
        "from": ["aeon", "aeon-fast", "aeon-deep"],
        "to": "Qwen3.5-27B-mm-modelopt",
        "proposed_tags": ["fast", "deep"],
    }]


def test_role_flavoured_aliases_are_proposed_as_tags():
    profiles = {"mm27b": {"served_model_name": ["aeon", "aeon-fast"],
                          "identity": "Qwen3.5-27B-mm-modelopt"}}
    plan = plan_rename(profiles, routes={}, client_pins={})
    assert "fast" in plan["renames"][0]["proposed_tags"]


def test_already_correct_profile_produces_no_rename():
    profiles = {"heretic": {"served_model_name": ["Qwen3.6-35B-A3B-heretic-NVFP4"],
                            "identity": "Qwen3.6-35B-A3B-heretic-NVFP4"}}
    assert plan_rename(profiles, routes={}, client_pins={})["renames"] == []


def test_two_profiles_resolving_to_one_identity_is_a_collision():
    profiles = {
        "a": {"served_model_name": ["x"], "identity": "SameModel"},
        "b": {"served_model_name": ["y"], "identity": "SameModel"},
    }
    plan = plan_rename(profiles, routes={}, client_pins={})
    assert plan["collisions"]
    assert plan["renames"] == []


def test_affected_client_pins_are_listed():
    profiles = {"heretic": {"served_model_name": ["heretic"],
                            "identity": "Qwen3.6-35B-A3B-heretic-NVFP4"}}
    pins = {"spark-heretic": ["omp modelRoles.default"]}
    plan = plan_rename(profiles, routes={}, client_pins=pins)
    assert "omp modelRoles.default" in str(plan["client_pins"])


def test_planner_never_mutates_its_inputs():
    profiles = {"mm27b": {"served_model_name": ["aeon"], "identity": "M"}}
    before = str(profiles)
    plan_rename(profiles, routes={}, client_pins={})
    assert str(profiles) == before


def test_planner_never_mutates_routes_or_client_pins():
    """Same guarantee as the input-profiles test above, extended to the
    other two args — 'pure, no I/O' means none of the three, not just the
    one the 08-04 plan happened to assert on."""
    routes = {"spark-aeon": "openai/aeon"}
    pins = {"spark-aeon": ["omp modelRoles.default"]}
    routes_before, pins_before = str(routes), str(pins)

    profiles = {"mm27b": {"served_model_name": ["aeon"], "identity": "M"}}
    plan_rename(profiles, routes, pins)

    assert str(routes) == routes_before
    assert str(pins) == pins_before


def test_duplicate_role_suffix_across_aliases_is_not_repeated_in_tags():
    """mm27b's real compose carries two independent -ultimate aliases
    (aeon-ultimate, qwen36-ultimate) -- proposed_tags is a TAG SET to apply
    to the renamed identity, not one entry per alias, so a suffix shared by
    more than one alias still appears exactly once (first-seen order)."""
    profiles = {"mm27b": {"served_model_name": ["aeon", "aeon-ultimate", "qwen36-ultimate"],
                          "identity": "Qwen3.5-27B-mm-modelopt"}}
    plan = plan_rename(profiles, routes={}, client_pins={})
    assert plan["renames"][0]["proposed_tags"] == ["ultimate"]


def test_route_targeting_a_renamed_alias_is_reported():
    """A litellm route whose litellm_params.model resolves to an alias being
    renamed appears in the plan with its route name, so the operator knows
    which routes to regenerate. Join on the resolved model (facts.py's
    rule, routers/facts.py:120-123: NEVER model_name — that's the alias)."""
    profiles = {"mm27b": {"served_model_name": ["aeon", "aeon-fast"],
                          "identity": "Qwen3.5-27B-mm-modelopt"}}
    routes = {"spark-aeon": "openai/aeon"}

    plan = plan_rename(profiles, routes, client_pins={})

    assert any(r["route"] == "spark-aeon" for r in plan["client_pins"])


def test_a_route_targeting_an_already_correct_alias_is_not_reported():
    """The flip side of the test above: nothing is changing for this
    profile, so its route needs no regeneration and must not show up."""
    profiles = {"heretic": {"served_model_name": ["Qwen3.6-35B-A3B-heretic-NVFP4"],
                            "identity": "Qwen3.6-35B-A3B-heretic-NVFP4"}}
    routes = {"spark-heretic": "openai/Qwen3.6-35B-A3B-heretic-NVFP4"}

    plan = plan_rename(profiles, routes, client_pins={})

    assert plan["client_pins"] == []


# ===========================================================================
# POST /api/rename/plan — the route
# ===========================================================================

_FIXTURES = Path(__file__).parent / "fixtures" / "spark-profiles"
_HERETIC_COMPOSE = (_FIXTURES / "compose-heretic.yaml").read_text()
_MM27B_COMPOSE = (_FIXTURES / "compose-mm27b.yaml").read_text()
_DS4_COMPOSE = (_FIXTURES / "compose-ds4.yaml").read_text()


class FakeSparkForRename:
    """Mirrors test_api.py's FakeSparkForAdopt (Task 5): real compose
    fixtures for two real vLLM profiles — mm27b (a genuine multi-alias
    collapse, six served names) and heretic (a single stale alias) — plus
    one non-vllm profile (ds4) the route must skip, same as adopt does."""

    def __init__(self):
        self.compose = {"heretic": _HERETIC_COMPOSE, "mm27b": _MM27B_COMPOSE,
                        "ds4": _DS4_COMPOSE}

    def status(self):
        return {"profiles": [
            {"name": "heretic", "engine": "vllm", "health_url": None, "container": None},
            {"name": "mm27b", "engine": "vllm", "health_url": None, "container": None},
            {"name": "ds4", "engine": "ds4",
             "health_url": "http://127.0.0.1:8000/metrics", "container": "spark-ds4"},
        ], "swap_status": None, "serving": None}

    def get_compose(self, profile):
        return self.compose[profile]


class FakeLiteLLMRoutes:
    """Just route_table() — the only litellm surface this route touches."""

    def __init__(self, routes=None):
        self._routes = dict(routes or {})

    def route_table(self):
        return dict(self._routes)


def _rename_app(tmp_path, monkeypatch, spark="default", routes=None):
    from app.characteristics import CharacteristicsStore
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    if spark is not None:
        wire_swap_node(deck, "boxa",
                       FakeSparkForRename() if spark == "default" else spark,
                       label="Box Alpha")
    deck["litellm"] = FakeLiteLLMRoutes(routes)
    # Same reasoning as test_api.py's _adopt_app: the default deck's stores
    # point at the container's /data, which doesn't exist under test, AND
    # the no-mutation tests below need a real, readable store to prove
    # nothing landed in it.
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    deck["characteristics_store"] = CharacteristicsStore(tmp_path / "c.json")
    deck["intent_store"] = IntentStore(tmp_path / "intent.json")
    return app, deck


def test_route_plans_mm27b_style_renames(tmp_path, monkeypatch):
    app, _ = _rename_app(
        tmp_path, monkeypatch, routes={"spark-heretic": "openai/heretic"})

    resp = TestClient(app).post("/api/rename/plan", json={})

    assert resp.status_code == 200
    body = resp.json()

    mm27b = next(r for r in body["renames"] if r["profile"] == "mm27b")
    assert mm27b["to"] == "Qwen3.6-27B-AEON-MM-MTP"
    assert "aeon" in mm27b["from"]
    # Real fixture: aeon, aeon-fast, aeon-deep, aeon-ultimate,
    # qwen36-ultimate, aeon-ultimate-xs -- two aliases share the "ultimate"
    # suffix, so this also locks in the dedup (one "ultimate", not two).
    assert mm27b["proposed_tags"] == ["fast", "deep", "ultimate", "xs"]

    heretic = next(r for r in body["renames"] if r["profile"] == "heretic")
    assert heretic["from"] == ["heretic"]
    assert heretic["to"] == "Qwen3.6-35B-A3B-heretic-NVFP4"

    assert any(p["route"] == "spark-heretic" for p in body["client_pins"])


def test_route_passes_through_caller_supplied_client_pins(tmp_path, monkeypatch):
    app, _ = _rename_app(tmp_path, monkeypatch)

    resp = TestClient(app).post(
        "/api/rename/plan",
        json={"client_pins": {"spark-heretic": ["omp modelRoles.default"]}},
    )

    assert resp.status_code == 200
    assert "omp modelRoles.default" in str(resp.json()["client_pins"])


def test_route_accepts_a_body_with_no_client_pins_key(tmp_path, monkeypatch):
    """body defaults to {} — an empty JSON object must still work."""
    app, _ = _rename_app(tmp_path, monkeypatch)

    assert TestClient(app).post("/api/rename/plan", json={}).status_code == 200


def test_route_accepts_no_body_at_all(tmp_path, monkeypatch):
    """client_pins' own default is {} per the task brief — posting with no
    body whatsoever must not 422."""
    app, _ = _rename_app(tmp_path, monkeypatch)

    assert TestClient(app).post("/api/rename/plan").status_code == 200


def test_route_503_when_spark_not_configured(tmp_path, monkeypatch):
    """Matches app.routers.serving.single_swap_node_id's guard exactly.
    The detail is pinned too: with the /api/spark/* alias tests gone, this
    is the only assertion anywhere on the message's text, which
    single_swap_node_id and rename_plan's client_for branch share."""
    app, _ = _rename_app(tmp_path, monkeypatch, spark=None)

    resp = TestClient(app).post("/api/rename/plan", json={})

    assert resp.status_code == 503
    assert resp.json()["detail"] == "spark engine is not configured"


def test_route_409s_with_two_swap_nodes(tmp_path, monkeypatch):
    """N1 T12 review, finding 1: single_swap_node_id moved to
    app.routers.serving, imported here rather than duplicated — this
    exercises the resolver through the rename route, and since the
    /api/spark/* alias (its other caller) was removed, it is the resolver's
    ONLY 409 coverage. Never guess between candidates
    ([[literal-declared-inputs]])."""
    app, deck = _rename_app(tmp_path, monkeypatch, spark=None)
    wire_swap_node(deck, "boxa", FakeSparkForRename(), label="Box Alpha")
    wire_swap_node(deck, "boxb", FakeSparkForRename(), label="Box Beta")

    resp = TestClient(app).post("/api/rename/plan", json={})

    assert resp.status_code == 409
    assert "boxa" in resp.json()["detail"] and "boxb" in resp.json()["detail"]


def test_route_is_read_only_second_call_matches_and_nothing_is_written(
    tmp_path, monkeypatch
):
    """The whole point of a PLANNER: a second identical call returns the
    identical plan, and none of adopt/intent/settings — the three stores a
    real rename EXECUTION would eventually touch — moved at all."""
    app, deck = _rename_app(
        tmp_path, monkeypatch, routes={"spark-heretic": "openai/heretic"})
    client = TestClient(app)

    settings_before = deck["settings_store"].get()
    intent_before = deck["intent_store"].get()
    characteristics_before = deck["characteristics_store"].get()

    first = client.post("/api/rename/plan", json={}).json()
    second = client.post("/api/rename/plan", json={}).json()

    assert first == second
    assert deck["settings_store"].get() == settings_before
    assert deck["intent_store"].get() == intent_before
    assert deck["characteristics_store"].get() == characteristics_before
