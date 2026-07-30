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
        self.fail = None
        self._status = {
            "profiles": ["laguna", "mm27b"],
            "swap_status": None,
            "serving": {"model": "aeon", "endpoint_ok": True,
                        "container_status": None},
        }

    def status(self):
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
    assert r.json()["profiles"] == ["laguna", "mm27b"]
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
