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
    def __init__(self, loaded=None):
        self.calls = []  # mutating only: ("load", model) / ("unload", model)
        self._loaded = loaded

    def status(self):
        return {"loaded": self._loaded}

    def activity(self):
        return None

    def load(self, model):
        self.calls.append(("load", model))
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
