"""Per-node serving routes (/api/nodes/{id}/serving/*) — N1 T9.

Two swap nodes throughout (boxa/boxb, ids ≠ "sparky") so a handler that
secretly resolves "the" spark node cannot pass ([[defaults-that-hide-bugs]]).
"""

from fastapi.testclient import TestClient

from app.engines import GuardError
from tests.test_api import FakeSpark, HERETIC_COMPOSE, make_app, wire_swap_node


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


# ===========================================================================
# Reload-guard permutations — recovered per N1 T10 review (controller
# ruling): T10 deleted these six from test_spark_api.py without a named
# replacement here; this is that replacement. Ported from the pre-T10
# tests/test_spark_api.py (git show f36efd8c:.../tests/test_spark_api.py),
# adapted exactly as T9 adapted the core reload suite above: route ->
# /api/nodes/boxa/serving/reload, wiring -> wire_swap_node via
# _reload_node_app/_reload_ready, scope key -> boxa/vllm|<identity>, and a
# `not b.calls` assertion added throughout to keep the two-node isolation
# discipline this file establishes.
# ===========================================================================


def test_reload_ships_env_most_specific_wins(tmp_path, monkeypatch):
    """Review fix round 1, IMPORTANT 1: _resolve_env had zero coverage
    anywhere in the suite — every other reload test seeds args only, and
    test_configure.py's env test passes env= straight into the mech (pins
    the mech, not the resolution). A copy-paste slip in _resolve_env (the
    wrong namespace, or a wrong scope key) would ship the wrong environment
    to a live vLLM launch with the rest of the suite green. Seeds a
    conflicting key at two scopes plus one key unique to each, and asserts
    document['env'] is exactly the per-key, most-specific-wins merge — the
    same ladder app.ladder.resolve_settings gives args, reused for env with
    both derived layers empty (app.routers.settings._resolve_env)."""
    app, deck, a, b = _reload_node_app(tmp_path, monkeypatch,
                                       identities=_HERETIC_IDENTITIES)
    a._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}

    # Launch-shaped args are a reload PRECONDITION (the positional guard
    # below): an env-only document ships an empty argv, which the helper
    # treats as "asserts nothing" and falls back to swap.sh — the env would
    # silently never apply.
    deck["settings_store"].put(
        "engine_models", f"boxa/vllm|{_IDENTITY}", "args",
        {"_positional": ["serve", "/model"]})
    deck["settings_store"].put("engines", "boxa/vllm", "env", {
        "VLLM_USE_FLASHINFER_SAMPLER": "1",   # unique to 'engines'
        "VLLM_LOGGING_LEVEL": "engine-level",  # overridden by engine_models
    })
    deck["settings_store"].put(
        "engine_models", f"boxa/vllm|{_IDENTITY}", "env", {
            "VLLM_LOGGING_LEVEL": "engine-model-level",  # most specific wins
            "CUDA_VISIBLE_DEVICES": "0",  # unique to 'engine_models'
        })

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 200
    _, document = a.settings_sent
    assert document["env"] == {
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_LOGGING_LEVEL": "engine-model-level",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert not b.calls


def test_reload_unadopted_profile_409(tmp_path, monkeypatch):
    """No identity-map entry for the requested profile -> 409 telling the
    operator to adopt first; put_settings must not have been called."""
    app, deck, a, b = _reload_node_app(tmp_path, monkeypatch,
                                       identities=_HERETIC_IDENTITIES)

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload",
                                json={"profile": "ghost"})

    assert resp.status_code == 409
    assert "adopt" in resp.json()["detail"].lower()
    assert a.settings_sent is None
    assert a.calls == []
    assert not b.calls


def test_reload_explicit_profile_overrides_serving(tmp_path, monkeypatch):
    """An explicit body profile wins over whatever swap_status reports as
    currently serving."""
    app, deck, a, b = _reload_node_app(tmp_path, monkeypatch,
                                       identities=_HERETIC_IDENTITIES)
    a._status["swap_status"] = {
        "state": "done", "profile": "mm27b", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}
    deck["settings_store"].put(
        "engine_models", f"boxa/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]})

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload",
                                json={"profile": "heretic"})

    assert resp.status_code == 200
    assert resp.json()["profile"] == "heretic"
    assert a.settings_sent[0] == "heretic"
    assert a.calls == [("swap", "heretic", False)]
    assert not b.calls


def test_reload_no_positionals_409(tmp_path, monkeypatch):
    """A pre-C2 'kept' scope holds args but no `serve /model` positionals.
    That argv is non-empty, so the helper OWNS the launch with it — and the
    engine never gets its subcommand. Refuse, naming adopt."""
    app, deck, a, b = _reload_ready(tmp_path, monkeypatch,
                                    args={"max-model-len": "131072"})

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 409
    assert "adopt" in resp.json()["detail"].lower()
    assert a.settings_sent is None
    assert a.calls == []
    assert not b.calls


def test_force_bypasses_the_positional_guard(tmp_path, monkeypatch):
    """Documented in serving_reload's docstring: force is the operator
    saying the entrypoint supplies the subcommand. It ships and swaps."""
    app, deck, a, b = _reload_ready(tmp_path, monkeypatch,
                                    args={"max-model-len": "131072"})

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload",
                                json={"profile": "heretic", "force": True})

    assert resp.status_code == 200
    assert a.settings_sent[0] == "heretic"
    assert a.calls == [("swap", "heretic", True)]
    assert not b.calls


def test_reload_empty_declared_args_409(tmp_path, monkeypatch):
    """An empty declared set ships an empty argv, which the helper reads as
    'asserts nothing' and delegates to swap.sh — the env in the same
    document then silently never applies. Same guard, same remedy."""
    app, deck, a, b = _reload_ready(tmp_path, monkeypatch, args={})
    deck["settings_store"].put("engines", "boxa/vllm", "env", {"K": "v"})

    resp = TestClient(app).post("/api/nodes/boxa/serving/reload", json={})

    assert resp.status_code == 409
    assert a.settings_sent is None
    assert a.calls == []
    assert not b.calls


def test_spark_alias_is_gone(tmp_path, monkeypatch):
    """/api/spark/* was a one-deploy-cycle alias for these routes (N1,
    design §6); its pre-registered removal happened after the first healthy
    cycle on the canonical routes.

    ONE node wired, unlike the rest of this file: this is the resurrection
    scenario that would actually answer — with a single swap node a revived
    forwarder returns 200 on status and records a swap call (with two it
    409s before touching any client, which would make the call assertions
    vacuous). The POSTs accept 405 as well as 404: on a checkout with
    ui/dist built, main.py's StaticFiles catch-all at "/" claims every
    unrouted path and refuses non-GET methods with 405. The route-table
    assertion carries the exactness either way."""
    app, deck = make_app(tmp_path, monkeypatch)
    fake = FakeSpark()
    wire_swap_node(deck, "boxa", fake, label="Box Alpha")
    assert not any(getattr(r, "path", "").startswith("/api/spark")
                   for r in app.routes)
    client = TestClient(app)
    assert client.get("/api/spark/status").status_code == 404
    assert client.post(
        "/api/spark/swap", json={"profile": "laguna"}).status_code in (404, 405)
    assert client.post("/api/spark/reload", json={}).status_code in (404, 405)
    assert fake.calls == []
