"""HTTP API tests for the spark endpoints (/api/spark/*).

Same construction as test_api.py: create_app() with the watcher off, then
the deck's "spark" entry swapped for a recording fake. The engine itself is
covered by test_spark_engine.py; these tests cover the router contract —
including the disabled state (deck["spark"] is None => 503), which is the
default on boxes with no spark configured.
"""

from fastapi.testclient import TestClient

from app.engines import BusyError, EngineError, GuardError
from tests.test_api import make_app


class FakeSpark:
    def __init__(self):
        self.calls = []  # mutating only: ("swap", profile, force)
        self.status_calls = 0
        self.fail = None
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
