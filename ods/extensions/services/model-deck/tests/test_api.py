"""Tests for the Model Deck HTTP API — app.security + app.routers.*.

TestClient against app.main.create_app(), with individual app.state.deck
entries swapped for recording fakes AFTER construction — no env vars beyond
MODEL_DECK_NO_WATCHER=1 (so no background thread starts) and, per test,
MODEL_DECK_ADMIN_TOKEN (read once at Settings() construction time, before
create_app() returns). No real sockets are ever touched.

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

ADMIN_TOKEN = "s3cr3t"
AUTH = {"X-Deck-Token": ADMIN_TOKEN}
PROXY_KEY = "pr0xy-s3cr3t"


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
    def __init__(self, state="running"):
        self.calls = []  # mutating only: "park" / "resume"
        self.fail = None
        self._state = state

    def status(self):
        return self._state

    def park(self):
        self.calls.append("park")
        if self.fail:
            raise self.fail
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


def make_app(tmp_path, monkeypatch, *, admin_token=ADMIN_TOKEN, proxy_key=None):
    """create_app() with MODEL_DECK_NO_WATCHER=1 and every engine client /
    read_gpus swapped for a fake; policy_store/set_store point at tmp_path
    (real, not faked — their own test files already cover their behavior).

    proxy_key defaults to unset (empty), which disables the Remote-Groups
    auth branch entirely — tests that exercise it must pass proxy_key
    explicitly, same as admin_token."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    if admin_token is not None:
        monkeypatch.setenv("MODEL_DECK_ADMIN_TOKEN", admin_token)
    else:
        monkeypatch.delenv("MODEL_DECK_ADMIN_TOKEN", raising=False)
    if proxy_key is not None:
        monkeypatch.setenv("MODEL_DECK_PROXY_KEY", proxy_key)
    else:
        monkeypatch.delenv("MODEL_DECK_PROXY_KEY", raising=False)

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
# Auth
# ===========================================================================


def test_mutating_endpoint_401_without_token(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 401


def test_mutating_endpoint_200_with_deck_token(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/tenants/comfyui/free", headers=AUTH)
    assert resp.status_code == 200
    assert deck["comfy"].calls == ["free"]


def test_mutating_endpoint_200_with_remote_groups_admins(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch, proxy_key=PROXY_KEY)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free",
        headers={"Remote-Groups": "users, admins", "X-Deck-Proxy-Key": PROXY_KEY},
    )
    assert resp.status_code == 200
    assert deck["comfy"].calls == ["free"]


def test_mutating_endpoint_401_wrong_token(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free", headers={"X-Deck-Token": "wrong"}
    )
    assert resp.status_code == 401


def test_mutating_endpoint_401_remote_groups_without_admins(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch, proxy_key=PROXY_KEY)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free",
        headers={"Remote-Groups": "users, editors", "X-Deck-Proxy-Key": PROXY_KEY},
    )
    assert resp.status_code == 401


def test_mutating_endpoint_401_remote_groups_no_proxy_key_configured(tmp_path, monkeypatch):
    # Server has no MODEL_DECK_PROXY_KEY set at all — the Remote-Groups
    # branch must be fully disabled, even though the header alone is
    # forgeable by any sibling compose container.
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free", headers={"Remote-Groups": "admins"}
    )
    assert resp.status_code == 401


def test_mutating_endpoint_401_remote_groups_wrong_proxy_key(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch, proxy_key=PROXY_KEY)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free",
        headers={"Remote-Groups": "admins", "X-Deck-Proxy-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_empty_admin_token_disables_token_auth_even_with_empty_header(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch, admin_token=None)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free", headers={"X-Deck-Token": ""}
    )
    assert resp.status_code == 401


def test_remote_groups_still_works_when_admin_token_empty(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch, admin_token=None, proxy_key=PROXY_KEY)
    resp = TestClient(app).post(
        "/api/tenants/comfyui/free",
        headers={"Remote-Groups": "admins", "X-Deck-Proxy-Key": PROXY_KEY},
    )
    assert resp.status_code == 200


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
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}, headers=AUTH
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.new.gguf")]


def test_lemonade_unload_explicit_model(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post(
        "/api/tenants/lemonade/unload", json={"model": "extra.m.gguf"}, headers=AUTH
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_omitted_model_uses_currently_loaded(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={}, headers=AUTH)
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_no_model_loaded_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={}, headers=AUTH)
    assert resp.status_code == 409
    assert deck["lemonade"].calls == []


def test_comfy_free_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = GuardError("ComfyUI queue is not empty")
    resp = TestClient(app).post("/api/tenants/comfyui/free", headers=AUTH)
    assert resp.status_code == 409
    assert resp.json()["detail"] == "ComfyUI queue is not empty"


def test_comfy_free_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = EngineError("comfyui unreachable")
    resp = TestClient(app).post("/api/tenants/comfyui/free", headers=AUTH)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "comfyui unreachable"


def test_hipfire_park_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = GuardError("litellm default route targets hipfire")
    resp = TestClient(app).post("/api/tenants/hipfire/park", headers=AUTH)
    assert resp.status_code == 409


def test_hipfire_resume_success(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/tenants/hipfire/resume", headers=AUTH)
    assert resp.status_code == 200
    assert deck["hipfire"].calls == ["resume"]


def test_hipfire_resume_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = EngineError("dockerctl unreachable")
    resp = TestClient(app).post("/api/tenants/hipfire/resume", headers=AUTH)
    assert resp.status_code == 502


# ===========================================================================
# Sets CRUD
# ===========================================================================


def test_sets_create_get_roundtrip(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/api/sets", json={"name": "Chat mode", "notes": "n"}, headers=AUTH
    )
    assert resp.status_code == 200
    assert resp.json() == {"slug": "chat-mode"}

    got = client.get("/api/sets/chat-mode")
    assert got.status_code == 200
    assert got.json()["name"] == "Chat mode"
    assert got.json()["notes"] == "n"


def test_sets_create_without_auth_401(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets", json={"name": "x"})
    assert resp.status_code == 401


def test_sets_create_bad_payload_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets", json={"name": ""}, headers=AUTH)
    assert resp.status_code == 422


def test_sets_create_duplicate_without_overwrite_409(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    body = {"name": "Chat mode"}
    client.post("/api/sets", json=body, headers=AUTH)

    resp = client.post("/api/sets", json=body, headers=AUTH)
    assert resp.status_code == 409


def test_sets_create_duplicate_with_overwrite_true_succeeds(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Chat mode", "notes": "v1"}, headers=AUTH)

    resp = client.post(
        "/api/sets?overwrite=true", json={"name": "Chat mode", "notes": "v2"}, headers=AUTH
    )
    assert resp.status_code == 200

    got = client.get("/api/sets/chat-mode")
    assert got.json()["notes"] == "v2"


def test_sets_get_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/sets/nope")
    assert resp.status_code == 404


def test_sets_delete_requires_auth_401(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="X"))
    resp = TestClient(app).delete("/api/sets/x")
    assert resp.status_code == 401


def test_sets_delete_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).delete("/api/sets/nope", headers=AUTH)
    assert resp.status_code == 404


def test_sets_delete_reserved_403(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save_previous(ConfigSet(name=PREVIOUS_NAME))

    resp = TestClient(app).delete(f"/api/sets/{RESERVED_SLUG}", headers=AUTH)

    assert resp.status_code == 403
    assert deck["set_store"].get(RESERVED_SLUG) is not None


def test_sets_delete_success(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Temp"}, headers=AUTH)

    resp = client.delete("/api/sets/temp", headers=AUTH)

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


def test_preview_requires_auth_401(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="X"))
    resp = TestClient(app).post("/api/sets/x/preview")
    assert resp.status_code == 401


def test_preview_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/preview", headers=AUTH)
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
        headers=AUTH,
    )

    resp = client.post("/api/sets/full-switch/preview", headers=AUTH)

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
        headers=AUTH,
    )

    resp = client.post("/api/sets/free-comfy/preview", headers=AUTH)

    body = resp.json()
    assert body["steps"] == [{"step": "warn", "reason": "comfyui-busy-skipped"}]
    assert body["estimate_s"] == 0


# ===========================================================================
# Apply — executes, fresh snapshot per call
# ===========================================================================


def test_apply_requires_auth_401(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="X"))
    resp = TestClient(app).post("/api/sets/x/apply")
    assert resp.status_code == 401


def test_apply_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/apply", headers=AUTH)
    assert resp.status_code == 404


def test_apply_executes_and_uses_a_fresh_snapshot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Load It", "ephemeral": {"lemonade": {"state": "loaded"}}},
        headers=AUTH,
    )

    client.get("/api/state")  # already calls read_gpus once
    calls_before = len(deck["read_gpus"].calls)

    resp = client.post("/api/sets/load-it/apply", headers=AUTH)

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
        headers=AUTH,
    )

    resp = client.post("/api/sets/noop/apply", headers=AUTH)

    assert resp.status_code == 200
    previous = deck["set_store"].get(RESERVED_SLUG)
    assert previous is not None
    assert previous.name == PREVIOUS_NAME


# ===========================================================================
# Policy
# ===========================================================================


def test_policy_get_default(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/policy")
    assert resp.status_code == 200
    assert resp.json() == DEFAULT_POLICIES


def test_policy_put_requires_auth_401(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).put(
        "/api/policy", json={"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}}
    )
    assert resp.status_code == 401


def test_policy_put_roundtrip_partial_by_tenant(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.put(
        "/api/policy",
        json={"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}},
        headers=AUTH,
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
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_policy_put_bad_field_type_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).put(
        "/api/policy",
        json={"lemonade": {"priority": "high", "pinned": True, "idle_ttl": 0}},
        headers=AUTH,
    )
    assert resp.status_code == 422
