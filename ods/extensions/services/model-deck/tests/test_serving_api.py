"""Per-node serving routes (/api/nodes/{id}/serving/*) — N1 T9.

Two swap nodes throughout (boxa/boxb, ids ≠ "sparky") so a handler that
secretly resolves "the" spark node cannot pass ([[defaults-that-hide-bugs]]).
"""

from fastapi.testclient import TestClient

from app.engines import GuardError
from tests.test_api import HERETIC_COMPOSE, make_app, wire_swap_node
from tests.test_spark_api import FakeSpark


def _two_node_app(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    a, b = FakeSpark(), FakeSpark()
    wire_swap_node(deck, "boxa", a, label="Box Alpha")
    wire_swap_node(deck, "boxb", b, label="Box Beta")
    return app, deck, a, b


def test_status_routes_to_the_named_node(tmp_path, monkeypatch):
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get("/api/nodes/boxb/serving/status")
    assert r.status_code == 200
    assert b.status_calls == 1 and a.status_calls == 0


def test_swap_records_intent_under_the_nodes_slot(tmp_path, monkeypatch):
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.post("/api/nodes/boxa/serving/swap", json={"profile": "laguna"})
    assert r.status_code == 200
    assert ("swap", "laguna", False) in a.calls and not b.calls
    intent = deck["intent_store"].get()["boxa/slot0"]
    assert intent["model"] == "laguna" and intent["engine"] == "spark"


def test_swap_unknown_node_404(tmp_path, monkeypatch):
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/nodes/ghost/serving/swap",
                             json={"profile": "laguna"})
    assert r.status_code == 404
    assert not a.calls and not b.calls


def test_swap_not_operable_503(tmp_path, monkeypatch):
    # control:"none" node WITH serving_address: declared-not-operable.
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    deck["node_store"].add(
        {"id": "boxc", "label": "Box Gamma", "agent_kind": "node-agent",
         "address": "http://boxc:7720", "serving_address": "http://boxc:8000",
         "control": "none"}, credential="key-boxc")

    r = TestClient(app).post("/api/nodes/boxc/serving/swap",
                             json={"profile": "laguna"})

    assert r.status_code == 503
    assert not a.calls and not b.calls


def test_swap_guard_409_records_no_intent(tmp_path, monkeypatch):
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    a.fail = GuardError("busy")

    r = TestClient(app).post("/api/nodes/boxa/serving/swap",
                             json={"profile": "laguna"})

    assert r.status_code == 409
    assert deck["intent_store"].get().get("boxa/slot0") is None


def test_force_passes_through(tmp_path, monkeypatch):
    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/nodes/boxa/serving/swap",
                             json={"profile": "laguna", "force": True})
    assert r.status_code == 200
    assert ("swap", "laguna", True) in a.calls
    assert not b.calls


# ===========================================================================
# POST /api/nodes/{id}/serving/reload — ported from test_spark_api.py's
# reload suite, per-node. settings_store and characteristics_store need
# real, tmp_path-backed instances (same reason as test_api.py's _adopt_app):
# the default deck's copies point at the container's /data, which does not
# exist under test.
# ===========================================================================

_IDENTITY = "Qwen3.6-35B-A3B-heretic-NVFP4"
_HERETIC_IDENTITIES = {"heretic": {"identity": _IDENTITY, "service": "aeon-vllm",
                                   "container_name": "aeon-vllm"}}


def _reload_node_app(tmp_path, monkeypatch, identities=None):
    from app.characteristics import CharacteristicsStore
    from app.settings_store import SettingsStore

    app, deck, a, b = _two_node_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    if identities is not None:
        characteristics.put_fields("engine/boxa/vllm", {
            "profile_identities": {"value": identities, "source": "compose import",
                                   "derived_ts": "t"},
        })
    deck["characteristics_store"] = characteristics
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    return app, deck, a, b


def _reload_ready(tmp_path, monkeypatch, args=None, identities=None):
    """A deck whose boxa/heretic reload would succeed: identity map, serving
    profile, and launch-shaped declared args unless a test says otherwise."""
    app, deck, a, b = _reload_node_app(
        tmp_path, monkeypatch, identities=identities or _HERETIC_IDENTITIES)
    a._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}
    deck["settings_store"].put(
        "engine_models", f"boxa/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]}
        if args is None else args)
    return app, deck, a, b


def test_reload_ships_then_reswaps_same_node(tmp_path, monkeypatch):
    """The reload happy path (Plan C2 Decision 5), per-node: resolve the
    ladder, ship it, re-swap, re-record intent — clearing settings_drift
    with zero extra machinery. boxb is wired throughout and never touched,
    proving the route addresses boxa specifically rather than resolving
    "the" spark node."""
    from app.observe import slot_key
    from app.routers import _settings_drift

    app, deck, a, b = _reload_node_app(tmp_path, monkeypatch,
                                       identities=_HERETIC_IDENTITIES)
    a.compose = {"heretic": HERETIC_COMPOSE}
    a._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}
    deck["settings_store"].put(
        "engine_models", f"boxa/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]})

    key = slot_key("boxa")
    deck["intent_store"].record(key, state="loaded", model="heretic",
                                engine="spark", now="2020-01-01T00:00:00+00:00")
    stale_intent = deck["intent_store"].get()[key]
    assert _settings_drift(deck["settings_store"].get(), key, stale_intent,
                           identity_map=_HERETIC_IDENTITIES) is not None

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["shipped"] is True
    assert body["profile"] == "heretic"
    assert body["id"] == "u1"

    sent_profile, document = a.settings_sent
    assert sent_profile == "heretic"
    assert document["args"] == {"max-model-len": "131072",
                                "_positional": ["serve", "/model"]}
    assert document["argv"] == ["serve", "/model", "--max-model-len", "131072"]
    assert document["service"] == "aeon-vllm"

    assert a.calls == [("swap", "heretic", False)]
    assert not b.calls

    fresh_intent = deck["intent_store"].get()[key]
    assert fresh_intent["model"] == "heretic"
    assert fresh_intent["updated_ts"] > stale_intent["updated_ts"]
    assert _settings_drift(deck["settings_store"].get(), key, fresh_intent,
                           identity_map=_HERETIC_IDENTITIES) is None


def test_reload_service_mismatch_409(tmp_path, monkeypatch):
    """A compose service renamed since adopt makes the identity map's
    service name stale; refuse before shipping, name re-adopt."""
    app, deck, a, b = _reload_ready(tmp_path, monkeypatch)
    a.compose = {
        "heretic": HERETIC_COMPOSE.replace("aeon-vllm:", "aeon-vllm-v2:", 1)}

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "aeon-vllm-v2" in detail and "aeon-vllm" in detail
    assert "adopt" in detail.lower()
    assert a.settings_sent is None
    assert a.calls == []
    assert not b.calls


def test_reload_no_profile_serving_409(tmp_path, monkeypatch):
    """No serving profile (swap_status None) and no explicit profile in the
    body -> 409; there is nothing to name a reload target from."""
    app, deck, a, b = _reload_node_app(tmp_path, monkeypatch, identities={})

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 409
    assert a.settings_sent is None
    assert a.calls == []
