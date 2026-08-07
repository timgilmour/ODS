"""HTTP API tests for the spark endpoints (/api/spark/*).

Same construction as test_api.py: create_app() with the watcher off, then
the deck's "spark" entry swapped for a recording fake. The engine itself is
covered by test_spark_engine.py; these tests cover the router contract —
including the disabled state (deck["spark"] is None => 503), which is the
default on boxes with no spark configured.
"""

import json

from fastapi.testclient import TestClient

from app.engines import BusyError, EngineError, GuardError
from tests.test_api import HERETIC_COMPOSE, make_app


class FakeSpark:
    def __init__(self):
        self.calls = []  # mutating only: ("swap", profile, force)
        self.status_calls = 0
        self.fail = None
        self.settings_sent = None  # (profile, document), last put_settings call
        self.settings_fail = None
        # Reload re-fetches the profile's compose before shipping (final
        # branch review: a stale service name in the identity map would
        # introduce an imageless service AFTER teardown killed everything),
        # so every reload test needs real compose text behind get_compose.
        self.compose = {}          # {profile: text}; default = the fixture
        self.compose_fail = None
        self._status = {
            "profiles": [
                {"name": "laguna", "engine": "vllm", "health_url": None,
                 "container": None},
                {"name": "mm27b", "engine": "vllm", "health_url": None,
                 "container": None},
            ],
            "swap_status": None,
            "serving": {"model": "aeon", "endpoint_ok": True,
                        "container_status": None},
        }

    def status(self):
        self.status_calls += 1
        return self._status

    def swap(self, profile, force=False):
        self.calls.append(("swap", profile, force))
        if self.fail:
            raise self.fail
        return {"id": "u1", "profile": profile}

    def put_settings(self, profile, document):
        if self.settings_fail:
            raise self.settings_fail
        self.settings_sent = (profile, document)

    def get_compose(self, profile):
        if self.compose_fail:
            raise self.compose_fail
        return self.compose.get(profile, HERETIC_COMPOSE)


def _spark_app(tmp_path, monkeypatch, spark="default"):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["spark"] = FakeSpark() if spark == "default" else spark
    return app, deck


def test_status_503_when_spark_not_configured(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch, spark=None)
    r = TestClient(app).get("/api/spark/status")
    assert r.status_code == 503


def test_swap_503_when_spark_not_configured(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch, spark=None)
    r = TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    assert r.status_code == 503


def test_status_passthrough(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    r = TestClient(app).get("/api/spark/status")
    assert r.status_code == 200
    assert r.json()["profiles"] == [
        {"name": "laguna", "engine": "vllm", "health_url": None,
         "container": None},
        {"name": "mm27b", "engine": "vllm", "health_url": None,
         "container": None},
    ]
    assert r.json()["serving"]["model"] == "aeon"


def test_swap_calls_engine_and_returns_id(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "id": "u1", "profile": "laguna"}
    assert deck["spark"].calls == [("swap", "laguna", False)]


def test_swap_passes_force(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    TestClient(app).post("/api/spark/swap",
                         json={"profile": "laguna", "force": True})
    assert deck["spark"].calls == [("swap", "laguna", True)]


def test_swap_guard_maps_to_409(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    deck["spark"].fail = GuardError("busy")
    r = TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    assert r.status_code == 409


def test_swap_busy_maps_to_409(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    deck["spark"].fail = BusyError("mid-swap")
    r = TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    assert r.status_code == 409


def test_swap_engine_error_maps_to_502(tmp_path, monkeypatch):
    app, deck = _spark_app(tmp_path, monkeypatch)
    deck["spark"].fail = EngineError("node down")
    r = TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    assert r.status_code == 502


# --- intent (Task 9, correction 0b) ------------------------------------------


def test_swap_records_intent_for_the_spark_slot(tmp_path, monkeypatch):
    """Without this the spark slot is only ever READ: it derives 'unmanaged'
    forever and the reconciler's spark restore branch is unreachable."""
    from app.observe import SPARK_SLOT_KEY

    app, deck = _spark_app(tmp_path, monkeypatch)

    TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})

    record = deck["intent_store"].get()[SPARK_SLOT_KEY]
    assert record["state"] == "loaded"
    assert record["engine"] == "spark"


def test_swap_records_the_profile_not_the_served_model(tmp_path, monkeypatch):
    """mm27b serves under --served-model-name aeon. observe_spark reports the
    PROFILE, so recording the served name would be permanent false drift."""
    from app.observe import SPARK_SLOT_KEY

    app, deck = _spark_app(tmp_path, monkeypatch)

    TestClient(app).post("/api/spark/swap", json={"profile": "mm27b"})

    assert deck["intent_store"].get()[SPARK_SLOT_KEY]["model"] == "mm27b"


def test_failed_swap_records_no_intent(tmp_path, monkeypatch):
    """Intent is last-known-GOOD: a guard-refused swap never happened."""
    app, deck = _spark_app(tmp_path, monkeypatch)
    deck["spark"].fail = GuardError("busy")

    TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})

    assert deck["intent_store"].get() == {}


def test_swap_invalidates_the_cached_spark_observation(tmp_path, monkeypatch):
    """A swap just changed the thing the TTL cache is holding; the next read
    must not report the outgoing profile."""
    app, deck = _spark_app(tmp_path, monkeypatch)
    observer = deck["spark_observer"]
    observer.status()
    before = deck["spark"].status_calls

    TestClient(app).post("/api/spark/swap", json={"profile": "laguna"})
    observer.status()

    assert deck["spark"].status_calls > before


# ===========================================================================
# POST /api/spark/reload (Plan C2, Task 7) — resolve -> ship -> re-swap the
# serving profile, ONE human action (design decision 5). settings_store and
# characteristics_store need real, tmp_path-backed instances (same reason
# as test_api.py's _adopt_app): the default deck's copies point at the
# container's /data, which does not exist under test.
# ===========================================================================

_IDENTITY = "Qwen3.6-35B-A3B-heretic-NVFP4"


def _reload_app(tmp_path, monkeypatch, identities=None, spark="default"):
    from app.characteristics import CharacteristicsStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["spark"] = FakeSpark() if spark == "default" else spark

    characteristics = CharacteristicsStore(tmp_path / "c.json")
    if identities is not None:
        characteristics.put_fields("engine/sparky/vllm", {
            "profile_identities": {"value": identities, "source": "compose import",
                                   "derived_ts": "t"},
        })
    deck["characteristics_store"] = characteristics
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    return app, deck


def test_reload_resolves_ships_and_swaps_the_serving_profile(tmp_path, monkeypatch):
    """The whole point of Decision 5: reload is ONE human action that
    resolves the ladder, ships it, re-swaps, and re-records intent — which
    clears settings_drift with zero extra machinery (the drift baseline IS
    the intent's updated_ts)."""
    from app.harvest import parse_probe_output
    from app.observe import SPARK_SLOT_KEY
    from app.routers import _settings_drift

    identities = {"heretic": {"identity": _IDENTITY, "service": "aeon-vllm",
                              "container_name": "aeon-vllm"}}
    app, deck = _reload_app(tmp_path, monkeypatch, identities=identities)
    deck["spark"]._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}

    # A harvested engine default (derived layer) that WOULD render as a
    # flag if the argline were not declared-only.
    probe_output = json.dumps({"options": [
        {"flags": ["--tokenizer-mode"], "type": "str",
         "choices": ["auto", "slow", "mistral", "custom"],
         "default": repr("auto"), "nargs": None, "cls": "_StoreAction",
         "help": "Tokenizer mode."},
    ]})
    deck["characteristics_store"].put_fields("engine/sparky/vllm", {
        "option_catalog": parse_probe_output(probe_output, engine_version="test", now="t"),
    })

    deck["settings_store"].put(
        "engine_models", f"sparky/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]})

    # A stale intent baseline (T0), well before the settings write above —
    # sets up the drift this reload must clear.
    deck["intent_store"].record(SPARK_SLOT_KEY, state="loaded", model="heretic",
                                engine="spark", now="2020-01-01T00:00:00+00:00")
    stale_intent = deck["intent_store"].get()[SPARK_SLOT_KEY]
    assert _settings_drift(deck["settings_store"].get(), SPARK_SLOT_KEY, stale_intent,
                           identity_map=identities) is not None

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["shipped"] is True
    assert body["profile"] == "heretic"
    assert body["id"] == "u1"

    sent_profile, document = deck["spark"].settings_sent
    assert sent_profile == "heretic"
    assert document["args"] == {"max-model-len": "131072",
                                "_positional": ["serve", "/model"]}
    assert document["argv"] == ["serve", "/model", "--max-model-len", "131072"]
    assert document["service"] == "aeon-vllm"
    # Declared-only: the harvested engine default never leaked into what
    # was shipped, even though it's part of the full resolution.
    assert "tokenizer-mode" not in document["args"]
    assert "--tokenizer-mode" not in document["argv"]

    assert deck["spark"].calls == [("swap", "heretic", False)]

    fresh_intent = deck["intent_store"].get()[SPARK_SLOT_KEY]
    assert fresh_intent["model"] == "heretic"
    assert fresh_intent["updated_ts"] > stale_intent["updated_ts"]
    assert _settings_drift(deck["settings_store"].get(), SPARK_SLOT_KEY, fresh_intent,
                           identity_map=identities) is None


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
    identities = {"heretic": {"identity": _IDENTITY, "service": "aeon-vllm",
                              "container_name": "aeon-vllm"}}
    app, deck = _reload_app(tmp_path, monkeypatch, identities=identities)
    deck["spark"]._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}

    # Launch-shaped args are a reload PRECONDITION (the positional guard
    # below): an env-only document ships an empty argv, which the helper
    # treats as "asserts nothing" and falls back to swap.sh — the env would
    # silently never apply.
    deck["settings_store"].put(
        "engine_models", f"sparky/vllm|{_IDENTITY}", "args",
        {"_positional": ["serve", "/model"]})
    deck["settings_store"].put("engines", "sparky/vllm", "env", {
        "VLLM_USE_FLASHINFER_SAMPLER": "1",   # unique to 'engines'
        "VLLM_LOGGING_LEVEL": "engine-level",  # overridden by engine_models
    })
    deck["settings_store"].put(
        "engine_models", f"sparky/vllm|{_IDENTITY}", "env", {
            "VLLM_LOGGING_LEVEL": "engine-model-level",  # most specific wins
            "CUDA_VISIBLE_DEVICES": "0",  # unique to 'engine_models'
        })

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 200
    _, document = deck["spark"].settings_sent
    assert document["env"] == {
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_LOGGING_LEVEL": "engine-model-level",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def test_reload_on_unadopted_profile_is_409(tmp_path, monkeypatch):
    """No identity-map entry for the requested profile -> 409 telling the
    operator to adopt first; put_settings must not have been called."""
    app, deck = _reload_app(tmp_path, monkeypatch,
                            identities={"heretic": {"identity": _IDENTITY,
                                                    "service": "aeon-vllm",
                                                    "container_name": "aeon-vllm"}})

    resp = TestClient(app).post("/api/spark/reload", json={"profile": "ghost"})

    assert resp.status_code == 409
    assert "adopt" in resp.json()["detail"].lower()
    assert deck["spark"].settings_sent is None
    assert deck["spark"].calls == []


def test_reload_with_nothing_serving_and_no_profile_is_409(tmp_path, monkeypatch):
    """No serving profile (swap_status None) and no explicit profile in the
    body -> 409; there is nothing to name a reload target from."""
    app, deck = _reload_app(tmp_path, monkeypatch, identities={})

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 409
    assert deck["spark"].settings_sent is None
    assert deck["spark"].calls == []


def test_reload_explicit_profile_overrides_serving(tmp_path, monkeypatch):
    """An explicit body profile wins over whatever swap_status reports as
    currently serving."""
    identities = {"heretic": {"identity": _IDENTITY, "service": "aeon-vllm",
                              "container_name": "aeon-vllm"}}
    app, deck = _reload_app(tmp_path, monkeypatch, identities=identities)
    deck["spark"]._status["swap_status"] = {
        "state": "done", "profile": "mm27b", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}
    deck["settings_store"].put(
        "engine_models", f"sparky/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]})

    resp = TestClient(app).post("/api/spark/reload", json={"profile": "heretic"})

    assert resp.status_code == 200
    assert resp.json()["profile"] == "heretic"
    assert deck["spark"].settings_sent[0] == "heretic"
    assert deck["spark"].calls == [("swap", "heretic", False)]


# --- pre-ship guards on the shipped document (final branch review) ---------

_HERETIC_IDENTITIES = {"heretic": {"identity": _IDENTITY, "service": "aeon-vllm",
                                   "container_name": "aeon-vllm"}}


def _reload_ready(tmp_path, monkeypatch, args=None, identities=None):
    """A deck whose heretic reload would succeed: identity map, serving
    profile, and launch-shaped declared args unless a test says otherwise."""
    app, deck = _reload_app(tmp_path, monkeypatch,
                            identities=identities or _HERETIC_IDENTITIES)
    deck["spark"]._status["swap_status"] = {
        "state": "done", "profile": "heretic", "id": "u0",
        "message": "swap launched", "ts": "2020-01-01T00:00:00Z"}
    deck["settings_store"].put(
        "engine_models", f"sparky/vllm|{_IDENTITY}", "args",
        {"max-model-len": "131072", "_positional": ["serve", "/model"]}
        if args is None else args)
    return app, deck


def test_reload_409s_when_the_compose_service_was_renamed(tmp_path, monkeypatch):
    """A compose service renamed since adopt makes the identity map's
    service name stale. Shipping it introduces a service with no image, so
    compose config validation fails AFTER the helper's _teardown_all has
    already killed everything — the node is left serving nothing. Refuse
    before shipping, and name re-adopt as the remedy."""
    app, deck = _reload_ready(tmp_path, monkeypatch)
    deck["spark"].compose = {
        "heretic": HERETIC_COMPOSE.replace("aeon-vllm:", "aeon-vllm-v2:", 1)}

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "aeon-vllm-v2" in detail and "aeon-vllm" in detail
    assert "adopt" in detail.lower()
    assert deck["spark"].settings_sent is None
    assert deck["spark"].calls == []


def test_force_does_not_bypass_the_service_mismatch_guard(tmp_path, monkeypatch):
    """A wrong service name can never launch, so there is nothing for force
    to mean here — unlike the positional guard below."""
    app, deck = _reload_ready(tmp_path, monkeypatch)
    deck["spark"].compose = {
        "heretic": HERETIC_COMPOSE.replace("aeon-vllm:", "aeon-vllm-v2:", 1)}

    resp = TestClient(app).post("/api/spark/reload",
                                json={"profile": "heretic", "force": True})

    assert resp.status_code == 409
    assert deck["spark"].settings_sent is None
    assert deck["spark"].calls == []


def test_reload_409s_when_declared_args_have_no_positionals(tmp_path, monkeypatch):
    """A pre-C2 'kept' scope holds args but no `serve /model` positionals.
    That argv is non-empty, so the helper OWNS the launch with it — and the
    engine never gets its subcommand. Refuse, naming adopt."""
    app, deck = _reload_ready(tmp_path, monkeypatch,
                              args={"max-model-len": "131072"})

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 409
    assert "adopt" in resp.json()["detail"].lower()
    assert deck["spark"].settings_sent is None
    assert deck["spark"].calls == []


def test_force_bypasses_the_positional_guard(tmp_path, monkeypatch):
    """Documented in spark_reload's docstring: force is the operator saying
    the entrypoint supplies the subcommand. It ships and swaps."""
    app, deck = _reload_ready(tmp_path, monkeypatch,
                              args={"max-model-len": "131072"})

    resp = TestClient(app).post("/api/spark/reload",
                                json={"profile": "heretic", "force": True})

    assert resp.status_code == 200
    assert deck["spark"].settings_sent[0] == "heretic"
    assert deck["spark"].calls == [("swap", "heretic", True)]


def test_reload_with_no_declared_args_at_all_is_also_409(tmp_path, monkeypatch):
    """An empty declared set ships an empty argv, which the helper reads as
    'asserts nothing' and delegates to swap.sh — the env in the same
    document then silently never applies. Same guard, same remedy."""
    app, deck = _reload_ready(tmp_path, monkeypatch, args={})
    deck["settings_store"].put("engines", "sparky/vllm", "env", {"K": "v"})

    resp = TestClient(app).post("/api/spark/reload", json={})

    assert resp.status_code == 409
    assert deck["spark"].settings_sent is None
