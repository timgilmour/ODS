"""Tests for the Model Deck HTTP API — app.routers.* (no auth: the admin
gate was deliberately removed 2026-07-22; every endpoint is open).

TestClient against app.main.create_app(), with individual app.state.deck
entries swapped for recording fakes AFTER construction — no env vars beyond
MODEL_DECK_NO_WATCHER=1 (so no background thread starts). No real sockets
are ever touched.

Fakes track only *mutating* calls in `.calls` (load/unload/free/park/resume/
activate/policy put) — read-only calls a World snapshot legitimately makes
(status/queue_len/route_table) are NOT mutations and are deliberately left
untracked, so "assert fakes record zero calls" (preview) means exactly what
it says: no action was executed, even though preview must still read live
state to diff against.
"""

from fastapi.testclient import TestClient

from app.engines import EngineError, GuardError
from app.main import create_app
from app.policy import DEFAULT_POLICIES, PolicyStore
from app.sets import PREVIOUS_NAME, RESERVED_SLUG, ConfigSet, SetStore


# ===========================================================================
# Fakes
# ===========================================================================


class FakeLemonade:
    def __init__(self, loaded=None):
        self.calls = []  # mutating only: ("load", model) / ("unload", model)
        self.fail = None
        self._loaded = loaded

    def status(self):
        return {"loaded": self._loaded}

    def activity(self):
        return None

    def load(self, model):
        self.calls.append(("load", model))
        if self.fail:
            raise self.fail
        self._loaded = model

    def unload(self, model):
        self.calls.append(("unload", model))
        if self.fail:
            raise self.fail
        self._loaded = None


class FakeComfy:
    def __init__(self, queue=0):
        self.calls = []  # mutating only: "free"
        self.fail = None
        self._queue = queue

    def queue_len(self):
        return self._queue

    def free(self):
        self.calls.append("free")
        if self.fail:
            raise self.fail


class FakeHipfire:
    def __init__(self, state="running", busy_error=None):
        self.calls = []  # mutating only: "park" / "resume"
        self.fail = None
        self.busy_error = busy_error  # raised by ensure_not_busy / unforced park
        self.busy_checks = []
        self.park_forces = []
        self._state = state

    def status(self):
        return self._state

    def stats(self):
        return {"queue_depth": 0, "requests_served": 0}

    def ensure_not_busy(self, action):
        self.busy_checks.append(action)
        if self.busy_error:
            raise self.busy_error

    def park(self, force=False):
        self.calls.append("park")
        self.park_forces.append(force)
        if self.fail:
            raise self.fail
        if not force and self.busy_error:
            raise self.busy_error
        self._state = "parked"

    def resume(self):
        self.calls.append("resume")
        if self.fail:
            raise self.fail
        self._state = "running"


class FakeLiteLLM:
    def __init__(self, default=None, hipfire=None):
        self._routes = {}
        if default is not None:
            self._routes["default"] = default
        if hipfire is not None:
            self._routes["hipfire"] = hipfire

    def route_table(self):
        return dict(self._routes)


class FakeHostAgent:
    def __init__(self):
        self.calls = []  # mutating only: ("activate", model_id)
        self.fail = None

    def activate(self, model_id):
        self.calls.append(("activate", model_id))
        if self.fail:
            raise self.fail
        return {"activated": model_id}


class FakeRegistry:
    def __init__(self, footprints=None, models=None):
        self._footprints = footprints or {}
        self._models = models or []

    def footprint(self, key):
        if key not in self._footprints:
            raise FileNotFoundError(key)
        return self._footprints[key]

    def scan(self):
        return list(self._models)


class FakeReadGpus:
    """Recording stand-in for app.gpu.read_gpus; ignores the roots passed
    in and returns a fixed two-GPU list, but records every call."""

    def __init__(self, gpus=None):
        self.calls = []
        self._gpus = (
            gpus
            if gpus is not None
            else [
                {"index": 0, "vram_total": 40_000_000_000, "vram_used": 5_000_000_000, "pids": {}},
                {"index": 1, "vram_total": 24_000_000_000, "vram_used": 1_000_000_000, "pids": {}},
            ]
        )

    def __call__(self, drm_root, kfd_root):
        self.calls.append((drm_root, kfd_root))
        return self._gpus


# ===========================================================================
# App builder
# ===========================================================================


def make_app(tmp_path, monkeypatch):
    """create_app() with MODEL_DECK_NO_WATCHER=1 and every engine client /
    read_gpus swapped for a fake; policy_store/set_store point at tmp_path
    (real, not faked — their own test files already cover their behavior).
    No auth setup: the admin gate was deliberately removed 2026-07-22."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")

    app = create_app()
    deck = app.state.deck
    deck.update(
        {
            "lemonade": FakeLemonade(),
            "comfy": FakeComfy(),
            "hipfire": FakeHipfire(),
            "hostagent": FakeHostAgent(),
            "litellm": FakeLiteLLM(default="extra.model.gguf"),
            "registry": FakeRegistry(),
            "read_gpus": FakeReadGpus(),
            "policy_store": PolicyStore(tmp_path / "policy.json"),
            "set_store": SetStore(tmp_path / "sets"),
            "events_path": tmp_path / "events.jsonl",
        }
    )
    return app, deck


# ===========================================================================


def test_gets_are_open_without_auth(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    assert client.get("/api/state").status_code == 200
    assert client.get("/api/events").status_code == 200
    assert client.get("/api/sets").status_code == 200
    assert client.get("/api/policy").status_code == 200


# ===========================================================================
# /api/state, /api/events
# ===========================================================================


def test_api_state_shape(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["registry"] = FakeRegistry(models=[{"file": "m.gguf", "size": 1, "footprint": 2}])
    resp = TestClient(app).get("/api/state")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"world", "policy", "models"}
    assert body["policy"] == DEFAULT_POLICIES
    assert body["models"] == [{"file": "m.gguf", "size": 1, "footprint": 2}]
    assert body["world"]["default_route"] == "extra.model.gguf"
    assert body["world"]["tenants"]["hipfire"]["state"] == "running"
    assert body["world"]["placement"]["hipfire"] == 0


def test_api_state_calls_read_gpus_fresh_each_request(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    client.get("/api/state")
    client.get("/api/state")

    assert len(deck["read_gpus"].calls) == 2


def test_api_events_returns_tail(tmp_path, monkeypatch):
    from app.events import log_event

    app, deck = make_app(tmp_path, monkeypatch)
    log_event(deck["events_path"], "test-event", {"a": 1})

    resp = TestClient(app).get("/api/events?n=100")

    assert resp.status_code == 200
    events = resp.json()["events"]
    assert events[-1]["kind"] == "test-event"
    assert events[-1]["detail"] == {"a": 1}


# ===========================================================================
# Control: lemonade / comfyui / hipfire
# ===========================================================================


def test_lemonade_load(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.new.gguf")]


def test_lemonade_load_bare_name_is_extra_prefixed(tmp_path, monkeypatch):
    """The Deck select carries bare GGUF filenames; the load route prefixes
    them with 'extra.' before handing them to Lemonade (I2)."""
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "Qwen3.5-27B.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.Qwen3.5-27B.gguf")]


def test_lemonade_load_clears_heal_suppressor(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["heal_suppressor"].note_deck_unload()
    assert deck["heal_suppressor"].suppressed() is True
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert deck["heal_suppressor"].suppressed() is False


def test_lemonade_load_suppresses_healing_while_load_is_in_flight(tmp_path, monkeypatch):
    # While the blocking engine load runs, lemonade still reports "unloaded"
    # and the target GPU's VRAM is already climbing, so an un-suppressed
    # watcher tick infers a pending default-route load and stomps the manual
    # one. The route must arm suppression BEFORE the engine call; the
    # clear-after-success behavior stays.
    app, deck = make_app(tmp_path, monkeypatch)
    fake = deck["lemonade"]
    seen = {}
    original_load = fake.load

    def observing_load(model):
        seen["suppressed_during_load"] = deck["heal_suppressor"].suppressed()
        return original_load(model)

    fake.load = observing_load
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert seen["suppressed_during_load"] is True
    assert deck["heal_suppressor"].suppressed() is False


def test_lemonade_unload_engages_heal_suppressor(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    assert deck["heal_suppressor"].suppressed() is False
    resp = TestClient(app).post(
        "/api/tenants/lemonade/unload", json={"model": "extra.m.gguf"}
    )
    assert resp.status_code == 200
    assert deck["heal_suppressor"].suppressed() is True


def test_lemonade_unload_explicit_model(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post(
        "/api/tenants/lemonade/unload", json={"model": "extra.m.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_omitted_model_uses_currently_loaded(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={})
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_no_model_loaded_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={})
    assert resp.status_code == 409
    assert deck["lemonade"].calls == []


def test_comfy_free_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = GuardError("ComfyUI queue is not empty")
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "ComfyUI queue is not empty"


def test_comfy_free_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = EngineError("comfyui unreachable")
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "comfyui unreachable"


def test_hipfire_park_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = GuardError("litellm default route targets hipfire")
    resp = TestClient(app).post("/api/tenants/hipfire/park")
    assert resp.status_code == 409


def test_hipfire_resume_success(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/tenants/hipfire/resume")
    assert resp.status_code == 200
    assert deck["hipfire"].calls == ["resume"]


def test_hipfire_resume_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = EngineError("dockerctl unreachable")
    resp = TestClient(app).post("/api/tenants/hipfire/resume")
    assert resp.status_code == 502


# ===========================================================================
# Sets CRUD
# ===========================================================================


def test_sets_create_get_roundtrip(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/api/sets", json={"name": "Chat mode", "notes": "n"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"slug": "chat-mode"}

    got = client.get("/api/sets/chat-mode")
    assert got.status_code == 200
    assert got.json()["name"] == "Chat mode"
    assert got.json()["notes"] == "n"


def test_sets_create_bad_payload_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets", json={"name": ""})
    assert resp.status_code == 422


def test_sets_create_duplicate_without_overwrite_409(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    body = {"name": "Chat mode"}
    client.post("/api/sets", json=body)

    resp = client.post("/api/sets", json=body)
    assert resp.status_code == 409


def test_sets_create_duplicate_with_overwrite_true_succeeds(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Chat mode", "notes": "v1"})

    resp = client.post(
        "/api/sets?overwrite=true", json={"name": "Chat mode", "notes": "v2"}
    )
    assert resp.status_code == 200

    got = client.get("/api/sets/chat-mode")
    assert got.json()["notes"] == "v2"


def test_sets_get_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/sets/nope")
    assert resp.status_code == 404


def test_sets_delete_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).delete("/api/sets/nope")
    assert resp.status_code == 404


def test_sets_delete_reserved_403(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save_previous(ConfigSet(name=PREVIOUS_NAME))

    resp = TestClient(app).delete(f"/api/sets/{RESERVED_SLUG}")

    assert resp.status_code == 403
    assert deck["set_store"].get(RESERVED_SLUG) is not None


def test_sets_delete_success(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Temp"})

    resp = client.delete("/api/sets/temp")

    assert resp.status_code == 200
    assert client.get("/api/sets/temp").status_code == 404


def test_sets_list_separates_previous_and_filters_it_out_of_sets(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="Chat"))
    deck["set_store"].save_previous(ConfigSet(name=PREVIOUS_NAME))

    resp = TestClient(app).get("/api/sets")

    body = resp.json()
    assert [s["name"] for s in body["sets"]] == ["Chat"]
    assert body["previous"]["name"] == PREVIOUS_NAME


def test_sets_list_previous_null_when_absent(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/sets")
    assert resp.json() == {"sets": [], "previous": None}


# ===========================================================================
# Preview — pure, no execution
# ===========================================================================


def test_preview_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/preview")
    assert resp.status_code == 404


def test_preview_no_exec_and_estimate_arithmetic(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["comfy"] = FakeComfy(queue=0)
    deck["hipfire"] = FakeHipfire(state="parked")
    deck["litellm"] = FakeLiteLLM(default="extra.old.gguf")
    client = TestClient(app)

    client.post(
        "/api/sets",
        json={
            "name": "Full Switch",
            "durable": {"default_route_model": "extra.new.gguf", "activate_model_id": "cat-1"},
            "ephemeral": {
                "lemonade": {"state": "loaded"},
                "comfyui": {"state": "free"},
                "hipfire": {"state": "running"},
            },
            "policy_overrides": {"lemonade": {"priority": 10, "pinned": False, "idle_ttl": 60}},
        },
       
    )

    resp = client.post("/api/sets/full-switch/preview")

    assert resp.status_code == 200
    body = resp.json()
    step_names = [step["step"] for step in body["steps"]]
    assert step_names == [
        "free_comfyui",
        "activate",
        "resume_hipfire",
        "load_lemonade",
        "policy_patch",
    ]
    # 5 (free_comfyui, default) + 120 (activate) + 180 (resume_hipfire)
    # + 5 (load_lemonade, default) + 5 (policy_patch, default) == 315
    assert body["estimate_s"] == 315

    # Preview must not execute anything.
    assert deck["lemonade"].calls == []
    assert deck["comfy"].calls == []
    assert deck["hipfire"].calls == []
    assert deck["hostagent"].calls == []


def test_preview_warn_step_costs_zero(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"] = FakeComfy(queue=3)  # busy -> comfyui "free" desired emits a warn
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Free Comfy", "ephemeral": {"comfyui": {"state": "free"}}},
       
    )

    resp = client.post("/api/sets/free-comfy/preview")

    body = resp.json()
    assert body["steps"] == [{"step": "warn", "reason": "comfyui-busy-skipped"}]
    assert body["estimate_s"] == 0


# ===========================================================================
# Apply — executes, fresh snapshot per call
# ===========================================================================


def test_apply_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/apply")
    assert resp.status_code == 404


def test_apply_executes_and_uses_a_fresh_snapshot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Load It", "ephemeral": {"lemonade": {"state": "loaded"}}},
       
    )

    client.get("/api/state")  # already calls read_gpus once
    calls_before = len(deck["read_gpus"].calls)

    resp = client.post("/api/sets/load-it/apply")

    assert resp.status_code == 200
    body = resp.json()
    assert body["completed"] == [{"step": "load_lemonade", "model": "extra.model.gguf"}]
    assert body["failed"] is None
    assert deck["lemonade"].calls == [("load", "extra.model.gguf")]

    # The apply endpoint itself must call read_gpus exactly once more — a
    # fresh snapshot, never one reused from /api/state.
    assert len(deck["read_gpus"].calls) == calls_before + 1


def test_apply_captures_previous_snapshot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Noop", "ephemeral": {"hipfire": {"state": "running"}}},
       
    )

    resp = client.post("/api/sets/noop/apply")

    assert resp.status_code == 200
    previous = deck["set_store"].get(RESERVED_SLUG)
    assert previous is not None
    assert previous.name == PREVIOUS_NAME


def test_apply_hipfire_busy_veto_409_and_no_mutation(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        state="running",
        busy_error=GuardError("hipfire request in flight (queue_depth=1)"),
    )
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Park It", "ephemeral": {"hipfire": {"state": "parked"}}},
       
    )

    resp = client.post("/api/sets/park-it/apply")

    assert resp.status_code == 409
    assert "in flight" in resp.json()["detail"]
    assert deck["hipfire"].calls == []
    assert deck["set_store"].get(RESERVED_SLUG) is None


def test_apply_force_skips_hipfire_busy_veto(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        state="running",
        busy_error=GuardError("hipfire request in flight (queue_depth=1)"),
    )
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Park It", "ephemeral": {"hipfire": {"state": "parked"}}},
       
    )

    resp = client.post("/api/sets/park-it/apply?force=true")

    assert resp.status_code == 200
    assert resp.json()["failed"] is None
    assert deck["hipfire"].calls == ["park"]
    assert deck["hipfire"].park_forces == [True]


def test_hipfire_park_busy_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        busy_error=GuardError(
            "hipfire served a request 12s ago (activity window 600s; "
            "pass force=true to override)"
        )
    )

    resp = TestClient(app).post("/api/tenants/hipfire/park")

    assert resp.status_code == 409
    assert "activity window" in resp.json()["detail"]
    assert deck["hipfire"]._state == "running"


def test_hipfire_park_force_query_param(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(busy_error=GuardError("busy"))

    resp = TestClient(app).post("/api/tenants/hipfire/park?force=true")

    assert resp.status_code == 200
    assert deck["hipfire"].park_forces == [True]
    assert deck["hipfire"]._state == "parked"


# ===========================================================================
# Policy
# ===========================================================================


def test_policy_get_default(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/policy")
    assert resp.status_code == 200
    assert resp.json() == DEFAULT_POLICIES


def test_policy_put_roundtrip_partial_by_tenant(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.put(
        "/api/policy",
        json={"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}},
       
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["lemonade"] == {"priority": 5, "pinned": True, "idle_ttl": 0}
    assert body["comfyui"] == DEFAULT_POLICIES["comfyui"]  # untouched

    again = client.get("/api/policy")
    assert again.json() == body


def test_policy_put_unknown_tenant_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).put(
        "/api/policy",
        json={"bogus": {"priority": 5, "pinned": True, "idle_ttl": 0}},
       
    )
    assert resp.status_code == 422


def test_policy_put_bad_field_type_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).put(
        "/api/policy",
        json={"lemonade": {"priority": "high", "pinned": True, "idle_ttl": 0}},
       
    )
    assert resp.status_code == 422


# ===========================================================================
# Open access — no admin token required (auth deliberately removed)
# ===========================================================================


def test_mutating_endpoints_open_without_any_auth(tmp_path, monkeypatch):
    """Ops-first mode: every endpoint works with zero auth headers. The
    admin-token/proxy-key gate was removed 2026-07-22 (Tim: config and
    operations now, security later)."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.post("/api/tenants/comfyui/free")
    assert resp.status_code == 200
    assert deck["comfy"].calls == ["free"]

    resp = client.post("/api/sets", json={"name": "Open"})
    assert resp.status_code == 200

    resp = client.put(
        "/api/policy",
        json={
            "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
            "comfyui": {"priority": 60, "pinned": False, "idle_ttl": 300},
            "hipfire": {"priority": 90, "pinned": True, "idle_ttl": 0},
        },
    )
    assert resp.status_code == 200
