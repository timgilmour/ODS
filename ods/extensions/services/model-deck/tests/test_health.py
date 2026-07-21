from fastapi.testclient import TestClient

import app.main as main
from app.main import create_app


def test_health():
    assert TestClient(create_app()).get("/health").json() == {"status": "ok"}


class _FakeWatcher:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def test_lifespan_starts_and_stops_watcher(monkeypatch):
    """When MODEL_DECK_NO_WATCHER is not set, the watcher starts on startup
    and stops on shutdown."""
    monkeypatch.delenv("MODEL_DECK_NO_WATCHER", raising=False)
    fake = _FakeWatcher()
    monkeypatch.setattr(main, "_build_watcher", lambda settings: fake)

    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert fake.started == 1
        assert fake.stopped == 0

    assert fake.stopped == 1


def test_lifespan_skips_watcher_when_env_set(monkeypatch):
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")

    def _boom(settings):
        raise AssertionError("watcher must not be built when MODEL_DECK_NO_WATCHER=1")

    monkeypatch.setattr(main, "_build_watcher", _boom)

    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_build_watcher_constructs_real_deps(monkeypatch):
    """_build_watcher wires real clients/stores without opening any sockets."""
    monkeypatch.setenv("MODEL_DECK_ADMIN_TOKEN", "x")
    from app.arbiter import Watcher
    from app.settings import Settings

    watcher = main._build_watcher(Settings())

    assert isinstance(watcher, Watcher)
