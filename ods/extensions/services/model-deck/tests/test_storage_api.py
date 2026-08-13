"""Tests for the /api/storage HTTP surface (app.routers.storage) and the
pull-through cold-load path on app.routers.control.lemonade_load.

Same idiom as tests/test_api.py: TestClient over create_app(), with the
engine clients swapped for recording fakes after construction. Unlike
test_api.py, MODEL_DECK_DATA_DIR is pointed at tmp_path so location_store/
catalog/storage_policy_store/job_queue are REAL instances over real files —
that's the thing under test here (register a location, physically move a
file between two tmp roots, drive the queue synchronously).

No worker thread is ever started (MODEL_DECK_NO_WATCHER=1, and TestClient is
never used as a context manager, so the FastAPI lifespan — which is what
calls job_queue.start() — never runs). Jobs are driven synchronously with
deck["job_queue"]._process(deck["job_queue"]._pending.pop(0)).
"""

from fastapi.testclient import TestClient

import app.routers.control as control_module
from app.engines import EngineError
from app.events import tail_events
from app.main import create_app

# ===========================================================================
# Fakes (copied from tests/test_api.py's shapes, not imported — see that
# file's own note on why a cross-module import isn't worth it here)
# ===========================================================================


class FakeLemonade:
    def __init__(self, loaded=None, raise_on_load=None):
        self.calls = []  # mutating only: ("load", model) / ("unload", model)
        self._loaded = loaded
        # Fails the pull-through restart branch's load() specifically (see
        # test_pull_through_on_load_failure_still_records_intent) —
        # additive, existing call sites are unaffected.
        self.raise_on_load = raise_on_load

    def load_in_flight(self):
        return False

    def status(self):
        return {"loaded": self._loaded}

    def activity(self):
        return None

    def load(self, model):
        self.calls.append(("load", model))
        if self.raise_on_load:
            raise self.raise_on_load
        self._loaded = model

    def unload(self, model):
        self.calls.append(("unload", model))
        self._loaded = None


class _NeverReadyLemonade:
    """status() succeeds exactly once — satisfying notify_engine's own
    pre-restart "is something loaded" check, so the restart actually
    proceeds — then raises EngineError on every call after, simulating a
    lemonade container that accepted the restart but never becomes healthy.
    Exercises _pull_through's readiness-poll timeout path specifically
    (as opposed to notify_engine's own unrelated status() call failing)."""

    def __init__(self):
        self.calls = []
        self._first = True

    def load_in_flight(self):
        return False

    def status(self):
        if self._first:
            self._first = False
            return {"loaded": None}
        raise EngineError("still booting")

    def activity(self):
        return None

    def load(self, model):
        self.calls.append(("load", model))


class FakeComfy:
    def __init__(self, queue=0):
        self.calls = []
        self._queue = queue

    def queue_len(self):
        return self._queue

    def free(self):
        self.calls.append("free")


class FakeHipfire:
    def __init__(self, state="running"):
        self.calls = []
        self._state = state

    def status(self):
        return self._state

    def stats(self):
        return {"queue_depth": 0, "requests_served": 0}


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
    def lifecycle(self):
        return {"active": False, "operation": None, "target": None}


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


class FakeDockerCtl:
    def __init__(self):
        self.calls = []

    def stop(self, name):
        self.calls.append(("stop", name))

    def start(self, name):
        self.calls.append(("start", name))


# ===========================================================================
# App builder
# ===========================================================================


# Same posture as tests/test_api.py's _ENGINES/FakeLocalClients (copied, not
# imported — see this file's own header note on why): a fresh data dir has
# no presence proof, so World.snapshot would see no declared resources at
# all, and app.storage.unit_in_use (this file's own subject, plus
# app.observe.observe_local via the lifecycle block) still assumes the
# pre-E1 fixed triple. make_app seeds the declaration + wires
# World.snapshot's clients onto the fakes below unconditionally.
_ENGINES = [
    {"resource": "hipfire", "kind": "hipfire",
     "connection": {"container": "ods-hipfire"}, "gpu_index": 0,
     "policy_defaults": {"priority": 100, "pinned": True, "idle_ttl": 0}},
    {"resource": "lemonade", "kind": "lemonade",
     "connection": {"url": "http://llama-server:8080",
                    "metrics_url": "http://llama-server:8001/metrics",
                    "container": "ods-llama-server"},
     "gpu_index": 1,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
    {"resource": "comfyui", "kind": "comfyui",
     "connection": {"url": "http://comfyui:8188"}, "gpu_index": 1,
     "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
]


class FakeLocalClients:
    """See tests/test_api.py's class of the same name — live dict-key
    lookup off `deck` so a test's post-make_app `deck["lemonade"] = ...`
    reassignment is still what World.snapshot observes."""

    _DECK_KEY = {"lemonade": "lemonade", "comfyui": "comfy", "hipfire": "hipfire"}

    def __init__(self, deck: dict) -> None:
        self._deck = deck

    def client_for(self, resource: str):
        return self._deck.get(self._DECK_KEY.get(resource, resource))

    def retire_absent(self, keep_resources) -> None:
        pass


def make_app(tmp_path, monkeypatch, **deck_overrides):
    """create_app() with a real, tmp_path-backed data dir (so
    location_store/catalog/storage_policy_store/job_queue are real), no
    watcher, and every network-talking engine client swapped for a fake.
    ``deck_overrides`` replaces individual deck entries after construction,
    same as tests/test_api.py's make_app."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    # A real free-space check against the actual dev-box filesystem would
    # otherwise require every move destination to have >=2GB free (the
    # production default slack). Zero it so tiny test fixtures are enough.
    monkeypatch.setenv("MODEL_DECK_STORAGE_SLACK_BYTES", "0")

    app = create_app()
    deck = app.state.deck
    deck.update(
        {
            "lemonade": FakeLemonade(),
            "comfy": FakeComfy(),
            "hipfire": FakeHipfire(),
            "hostagent": FakeHostAgent(),
            "litellm": FakeLiteLLM(),
            "registry": FakeRegistry(),
            "read_gpus": lambda drm, kfd: [],
            "dockerctl": FakeDockerCtl(),
        }
    )
    # E1 Task 3: declare the coexistence triple + wire World.snapshot's
    # clients onto the fakes above (see _ENGINES/FakeLocalClients docstrings).
    deck["node_store"].update("local", {"engines": _ENGINES})
    deck["local_clients"] = FakeLocalClients(deck)
    deck.update(deck_overrides)
    return app, deck


def _spec(tmp_path, name, **overrides):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    spec = {
        "name": name,
        "path": str(root),
        "role": "hot",
        "store_type": "gguf",
        "engine": "none",
        "watermark_gb": None,
        "archive_to": None,
        "readonly": False,
    }
    spec.update(overrides)
    return root, spec


def _register(client, tmp_path, name, **overrides):
    root, spec = _spec(tmp_path, name, **overrides)
    resp = client.post("/api/storage/locations", json=spec)
    assert resp.status_code == 200, resp.text
    return root


def _write_gguf(root, name, content=b"x" * 1024):
    (root / name).write_bytes(content)
    return root / name


# ===========================================================================
# /api/storage/state
# ===========================================================================


def test_storage_state_shape(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/storage/state")
    assert resp.status_code == 200
    assert set(resp.json().keys()) == {"locations", "units", "jobs", "policy"}


# ===========================================================================
# Locations CRUD
# ===========================================================================


def test_register_location_via_api(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    _register(client, tmp_path, "hot")

    state = client.get("/api/storage/state").json()
    locs = {loc["name"]: loc for loc in state["locations"]}
    assert locs["hot"]["available"] is True


def test_register_bad_spec_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/storage/locations", json={"name": "bad"})
    assert resp.status_code == 422


def test_register_unmounted_path_409(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    _, spec = _spec(tmp_path, "ghost", path=str(tmp_path / "does-not-exist"))
    resp = TestClient(app).post("/api/storage/locations", json=spec)
    assert resp.status_code == 409


def test_update_location_clears_watermark_with_explicit_null(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    _register(client, tmp_path, "hot", watermark_gb=100.0)

    resp = client.put("/api/storage/locations/hot", json={"watermark_gb": None})
    assert resp.status_code == 200
    assert resp.json()["watermark_gb"] is None

    state = client.get("/api/storage/state").json()
    loc = next(loc for loc in state["locations"] if loc["name"] == "hot")
    assert loc["watermark_gb"] is None


def test_update_location_empty_patch_is_noop(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    _register(client, tmp_path, "hot", watermark_gb=100.0, readonly=True)

    resp = client.put("/api/storage/locations/hot", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["watermark_gb"] == 100.0
    assert body["readonly"] is True


# ===========================================================================
# Moves
# ===========================================================================


def test_move_endpoint_runs_job_to_done(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    dst = _register(client, tmp_path, "dst")
    _write_gguf(src, "model.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/storage/moves", json={"unit_id": "src:model.gguf", "dest": "dst"})
    assert resp.status_code == 200, resp.text
    job = resp.json()["job"]
    assert job["state"] == "queued"

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job["id"])
    assert done["state"] == "done"
    assert not (src / "model.gguf").exists()
    assert (dst / "model.gguf").exists()


def test_move_of_loaded_model_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch, lemonade=FakeLemonade(loaded="extra.a.gguf"))
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"})
    assert resp.status_code == 409


def test_move_default_route_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch, litellm=FakeLiteLLM(default="openai/extra.a.gguf"))
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"})
    assert resp.status_code == 409


def test_get_move_endpoint(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    job_id = client.post(
        "/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"}
    ).json()["job"]["id"]

    resp = client.get(f"/api/storage/moves/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id

    assert client.get("/api/storage/moves/nope").status_code == 404


def test_cancel_endpoint(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    job_id = client.post(
        "/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"}
    ).json()["job"]["id"]

    resp = client.delete(f"/api/storage/moves/{job_id}")
    assert resp.status_code == 200
    assert resp.json() == {"cancelled": True}

    resp = client.delete("/api/storage/moves/nope")
    assert resp.status_code == 404


# ===========================================================================
# Pins
# ===========================================================================


def test_pin_endpoint_roundtrip(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    resp = client.put("/api/storage/units/src:a.gguf", json={"pinned": True})
    assert resp.status_code == 200
    assert resp.json()["pinned"] is True

    state = client.get("/api/storage/state").json()
    unit = next(u for u in state["units"] if u["id"] == "src:a.gguf")
    assert unit["pinned"] is True


# ===========================================================================
# Policy
# ===========================================================================


def test_policy_endpoints_roundtrip(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    assert client.get("/api/storage/policy").json() == {"auto": False}

    resp = client.put("/api/storage/policy", json={"auto": True})
    assert resp.status_code == 200
    assert resp.json() == {"auto": True}

    assert client.get("/api/storage/policy").json() == {"auto": True}


# ===========================================================================
# Rescan
# ===========================================================================


def test_rescan_endpoint(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _write_gguf(src, "a.gguf")
    _write_gguf(src, "b.gguf")

    resp = client.post("/api/storage/rescan")
    assert resp.status_code == 200
    assert resp.json() == {"units": 2}


# ===========================================================================
# Pull-through cold load (app.routers.control.lemonade_load)
# ===========================================================================


def test_pull_through_cold_load_auto_off_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load", json={"model": "a.gguf"})
    assert resp.status_code == 409
    assert "pull=true" in resp.json()["detail"]
    assert deck["lemonade"].calls == []


def test_pull_through_with_flag_submits_job(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pulling"
    assert "job" in body
    assert deck["heal_suppressor"].suppressed() is True
    # Not executed synchronously — the queue is only ever driven by the test.
    assert deck["lemonade"].calls == []


def test_pull_through_on_success_loads_and_notes(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "done"
    assert ("load", "extra.a.gguf") in deck["lemonade"].calls
    assert deck["heal_suppressor"].suppressed() is False

    units = deck["catalog"].units()
    moved = next(u for u in units if u["type"] == "gguf" and u["name"] == "a.gguf")
    assert moved["last_used"] is not None

    # The restart branch's load is exactly as deliberate as the hot path's —
    # same intent-first ordering (task-3-brief), so it must record too.
    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.a.gguf"


def test_pull_through_on_load_failure_still_records_intent(tmp_path, monkeypatch):
    """2026-08-06 design ruling: the restart branch's intent record sits
    BEFORE its load() call (mirroring the hot path) — a load that then
    raises must still leave intent=loaded, so the reconciler retries it
    under the failure budget instead of losing track of it. The job itself
    fails through the same post-move path as a readiness timeout."""
    app, deck = make_app(
        tmp_path, monkeypatch,
        lemonade=FakeLemonade(raise_on_load=EngineError("engine boom")),
    )
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "failed"
    assert "post-move" in done["error"]

    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.a.gguf"


def test_pull_through_notify_deferred_completes_without_load(tmp_path, monkeypatch):
    """notify_engine defers the restart when a model is already loaded — the
    move must still complete (job "done"); no load is attempted (the file
    isn't registered until the next restart) and a storage_notify_deferred
    event records why."""
    app, deck = make_app(tmp_path, monkeypatch, lemonade=FakeLemonade(loaded="extra.other.gguf"))
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "done"
    assert deck["lemonade"].calls == []

    events = tail_events(deck["events_path"])
    deferred = [e for e in events if e["kind"] == "storage_notify_deferred"]
    assert len(deferred) == 1
    assert deferred[0]["detail"]["job"] == job_id
    assert deferred[0]["detail"]["model"] == "a.gguf"


def test_pull_through_readiness_timeout_fails_job(tmp_path, monkeypatch):
    """DockerCtl.start() isn't a readiness gate — if lemonade never comes
    back healthy within the poll window, the job must fail post-move
    (not silently succeed with a phantom load, and not hang)."""
    monkeypatch.setattr(control_module, "_READY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(control_module, "_READY_POLL_S", 0.01)
    app, deck = make_app(tmp_path, monkeypatch, lemonade=_NeverReadyLemonade())
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "failed"
    assert "post-move" in done["error"]
    assert deck["lemonade"].calls == []


def test_pull_through_hook_skips_load_when_superseded(tmp_path, monkeypatch):
    """[c41] semantic half, task 6: the completion hook is a third actuator
    (mover thread, minutes after the request returned "pulling") that used
    to coordinate with nobody. Submit the pull at t0; while the copy is "in
    flight" (synchronous here, but the intent record below stands in for
    whatever happened during the real minutes-long window), the operator's
    own set parks lemonade — recorded at t1 > t0. The completion hook must
    see that deliberate action outranks the pull it would otherwise finish
    with a load: it must NOT restart/load, must log
    'pull-through-superseded', and must leave the superseding intent exactly
    as the operator recorded it (no re-stamp, no silent overwrite).

    actor="operator" explicit here (max-review Important-1, task 6 follow-
    up): only an OPERATOR-authored record may supersede the hook — see
    test_pull_through_hook_loads_despite_deck_authored_record_after_submission
    for the sibling case this rule exists to fix."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    # t1: recorded strictly after submission — a far-future ISO stamp so the
    # native-representation string comparison in control.py's after() is
    # unambiguous regardless of how fast this test runs (app/intent.py:41-42
    # stamps updated_ts with datetime.now(UTC).isoformat(); the hook compares
    # in that exact representation, never coerced to epoch seconds).
    deck["intent_store"].record(
        "local/lemonade", state="unloaded", model=None, engine="lemonade",
        now="2999-01-01T00:00:00+00:00", actor="operator",
    )

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "done"  # the move itself still succeeded
    assert deck["lemonade"].calls == []  # but no restart-driven load happened

    events = tail_events(deck["events_path"])
    superseded = [e for e in events if e["kind"] == "pull-through-superseded"]
    assert len(superseded) == 1
    assert superseded[0]["detail"]["job"] == job_id
    assert superseded[0]["detail"]["model"] == "a.gguf"
    assert superseded[0]["detail"]["intent_state"] == "unloaded"

    record = deck["intent_store"].get()["local/lemonade"]
    assert record["actor"] == "operator"
    assert record["state"] == "unloaded"
    assert record["updated_ts"] == "2999-01-01T00:00:00+00:00"


def test_pull_through_hook_loads_despite_deck_authored_record_after_submission(tmp_path, monkeypatch):
    """[max-review Important-1] The predicate this test guards against: "ANY
    intent recorded after submission supersedes the hook" is WRONG — the
    arbiter records intent on its own automatic actions too (idle-release at
    arbiter.py's unload arm, pending-load retrigger), and the heal suppressor
    does not stop idle rules. Live scenario this reproduces: an idle model Y
    unloads mid-copy (a deck-authored 'unloaded' record, actor="deck") while
    this operator's pull-through is still running. That must NOT silently
    drop the operator's explicit load — only an OPERATOR-authored record may
    supersede. A deck-authored record after submission must be invisible to
    the check: the load still happens."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    # Deck-authored, strictly after submission — exactly the shape
    # app.arbiter's idle-release/contention-eviction unload writes.
    deck["intent_store"].record(
        "local/lemonade", state="unloaded", model=None, engine="lemonade",
        now="2999-01-01T00:00:00+00:00", actor="deck",
    )

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "done"
    assert ("load", "extra.a.gguf") in deck["lemonade"].calls  # the load DID happen

    events = tail_events(deck["events_path"])
    assert [e for e in events if e["kind"] == "pull-through-superseded"] == []

    # The hook's own completion record wins last (it ran after the deck's
    # stale-mid-copy unload) and is itself operator-authored.
    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["actor"] == "operator"


def test_pull_through_hook_loads_when_intent_predates_submission(tmp_path, monkeypatch):
    """[max-review Important-2] The comparison direction that a green suite
    doesn't otherwise exercise: every OTHER pull-through test starts with NO
    pre-existing intent record (entry is None), so a mutant that weakens the
    predicate from "postdates submission" down to merely "entry is not None"
    would still pass every one of them. Seed an OLD operator-authored record
    (well before submission) and prove the load still happens — this is the
    one test that specifically requires the timestamp comparison, not just
    presence, to be checked."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    deck["catalog"].scan()

    # An old record, well BEFORE submission — e.g. yesterday's manual park,
    # long since irrelevant to this pull.
    deck["intent_store"].record(
        "local/lemonade", state="unloaded", model=None, engine="lemonade",
        now="2020-01-01T00:00:00+00:00", actor="operator",
    )

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})
    job_id = resp.json()["job"]

    pending = deck["job_queue"]._pending.pop(0)
    deck["job_queue"]._process(pending)

    done = deck["job_queue"].get(job_id)
    assert done["state"] == "done"
    assert ("load", "extra.a.gguf") in deck["lemonade"].calls

    events = tail_events(deck["events_path"])
    assert [e for e in events if e["kind"] == "pull-through-superseded"] == []


def test_hot_load_notes_last_used(tmp_path, monkeypatch):
    app, deck = make_app(
        tmp_path, monkeypatch,
        registry=FakeRegistry(models=[{"file": "a.gguf", "size": 1, "footprint": 2}]),
    )
    client = TestClient(app)
    hot = _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(hot, "a.gguf")
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load", json={"model": "a.gguf"})
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.a.gguf")]

    units = deck["catalog"].units()
    hit = next(u for u in units if u["type"] == "gguf" and u["name"] == "a.gguf")
    assert hit["last_used"] is not None


# ===========================================================================
# Destination-collision guard (C1b: plan/UX layer)
# ===========================================================================


def test_move_onto_existing_destination_file_409(tmp_path, monkeypatch):
    """A move whose destination already holds a same-named file is refused at
    plan time (409) — the operator sees it immediately instead of the worker
    discovering it, and the archived copy is never at risk."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    dst = _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    _write_gguf(dst, "a.gguf", content=b"older archived version")
    deck["catalog"].scan()

    resp = client.post("/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]
    assert (dst / "a.gguf").read_bytes() == b"older archived version"
    assert (src / "a.gguf").exists()


# ===========================================================================
# Execution-start guard re-check (I2 — spec section 2: guards are evaluated at
# plan time AND re-checked at execution start)
# ===========================================================================


def test_queued_move_refused_when_model_loaded_before_execution(tmp_path, monkeypatch):
    """A move can sit queued for a long time behind another job. If the model
    gets loaded in the meantime, the worker must refuse rather than yank the
    file out from under lemonade — the plan-time guard alone is not enough."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    dst = _register(client, tmp_path, "dst")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    job_id = client.post(
        "/api/storage/moves", json={"unit_id": "src:a.gguf", "dest": "dst"}
    ).json()["job"]["id"]

    # Reality moves between plan time and execution start.
    deck["lemonade"] = FakeLemonade(loaded="extra.a.gguf")

    deck["job_queue"]._process(deck["job_queue"]._pending.pop(0))

    failed = deck["job_queue"].get(job_id)
    assert failed["state"] == "failed"
    assert "currently loaded" in failed["error"]
    assert (src / "a.gguf").exists() and not (dst / "a.gguf").exists()
    assert deck["catalog"].get("src:a.gguf")["state"] == "resident"


# ===========================================================================
# /state is a read, not a disk walk (I3)
# ===========================================================================


def test_state_reads_the_catalog_without_rescanning_disk(tmp_path, monkeypatch):
    """GET /state used to walk every location on every poll (the UI polls it
    on a timer). It now reads the persisted catalog; rescans happen on the
    watcher tick, POST /rescan, and the cold-model lookup."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    src = _register(client, tmp_path, "src")
    _write_gguf(src, "a.gguf")
    deck["catalog"].scan()

    assert len(client.get("/api/storage/state").json()["units"]) == 1

    _write_gguf(src, "b.gguf")          # dropped in behind the deck's back
    assert len(client.get("/api/storage/state").json()["units"]) == 1

    assert client.post("/api/storage/rescan").json() == {"units": 2}
    assert len(client.get("/api/storage/state").json()["units"]) == 2


# ===========================================================================
# last_used observation through the set-apply route (I5 wiring)
# ===========================================================================


def test_set_apply_load_notes_last_used_through_the_router(tmp_path, monkeypatch):
    """End-to-end for the router half of the set-apply observation wiring."""
    app, deck = make_app(
        tmp_path, monkeypatch, litellm=FakeLiteLLM(default="openai/extra.a.gguf")
    )
    client = TestClient(app)
    hot = _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(hot, "a.gguf")
    deck["catalog"].scan()

    assert client.post("/api/sets", json={
        "name": "chat", "ephemeral": {"lemonade": {"state": "loaded"}},
    }).status_code == 200
    resp = client.post("/api/sets/chat/apply")
    assert resp.status_code == 200, resp.text
    assert resp.json()["failed"] is None
    assert deck["lemonade"].calls == [("load", "extra.a.gguf")]

    unit = next(u for u in deck["catalog"].units() if u["name"] == "a.gguf")
    assert unit["last_used"] is not None


# ===========================================================================
# Heal-suppression window hygiene around pull-through (I6)
# ===========================================================================


def test_refused_pull_through_does_not_leave_suppression_armed(tmp_path, monkeypatch):
    """The suppressor is pre-armed BEFORE the move is submitted (a multi-minute
    pull must not fight the VRAM watcher). If the submit is then refused, the
    window must not stay armed for its full duration with no pull to protect —
    that would silently disable contention healing."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    cold = _register(client, tmp_path, "cold", engine="none")
    hot = _register(client, tmp_path, "hot", engine="lemonade")
    _write_gguf(cold, "a.gguf")
    _write_gguf(hot, "a.gguf")          # destination collision -> submit refused
    deck["catalog"].scan()

    resp = client.post("/api/tenants/lemonade/load?pull=true", json={"model": "a.gguf"})

    assert resp.status_code == 409
    assert deck["heal_suppressor"].suppressed() is False
    assert deck["job_queue"].jobs() == []
