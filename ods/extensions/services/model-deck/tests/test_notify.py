"""Tests for app.notify.notify_engine — post-move engine registration hooks."""
import pytest

from app.engines import EngineError
from app.notify import notify_engine


class _Lemonade:
    def __init__(self, loaded=None):
        self._loaded = loaded
    def status(self):
        return {"loaded": self._loaded}


class _DockerCtl:
    def __init__(self, stop_exc=None, start_exc=None):
        self.calls = []
        self._stop_exc = stop_exc
        self._start_exc = start_exc
    def stop(self, name):
        self.calls.append(("stop", name))
        if self._stop_exc is not None:
            raise self._stop_exc
    def start(self, name):
        self.calls.append(("start", name))
        if self._start_exc is not None:
            raise self._start_exc


class _Settings:
    lemonade_container = "ods-llama-server"


def _deck(loaded=None, dockerctl=None):
    return {"lemonade": _Lemonade(loaded), "dockerctl": dockerctl or _DockerCtl(), "settings": _Settings()}


def _loc(engine):
    return {"name": "x", "engine": engine}


def test_lemonade_idle_restarts_container():
    deck = _deck(loaded=None)
    assert notify_engine(_loc("lemonade"), deck) is None
    assert deck["dockerctl"].calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]


def test_lemonade_loaded_defers_with_warning():
    deck = _deck(loaded="extra.a.gguf")
    warning = notify_engine(_loc("lemonade"), deck)
    assert warning and "restart deferred" in warning
    assert deck["dockerctl"].calls == []


def test_comfyui_and_none_are_noops():
    deck = _deck()
    assert notify_engine(_loc("comfyui"), deck) is None
    assert notify_engine(_loc("none"), deck) is None
    assert deck["dockerctl"].calls == []


def test_lemonade_stop_timeout_then_start_succeeds_returns_none(monkeypatch):
    slept = []
    monkeypatch.setattr("app.notify.time.sleep", lambda s: slept.append(s))
    dockerctl = _DockerCtl(stop_exc=EngineError("timed out"))
    deck = _deck(loaded=None, dockerctl=dockerctl)

    result = notify_engine(_loc("lemonade"), deck)

    assert result is None
    assert dockerctl.calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]
    assert slept == [10]


def test_lemonade_stop_timeout_then_start_also_fails_raises_start_error(monkeypatch):
    monkeypatch.setattr("app.notify.time.sleep", lambda s: None)
    dockerctl = _DockerCtl(
        stop_exc=EngineError("stop timed out"),
        start_exc=EngineError("start timed out"),
    )
    deck = _deck(loaded=None, dockerctl=dockerctl)

    with pytest.raises(EngineError, match="start timed out"):
        notify_engine(_loc("lemonade"), deck)

    assert dockerctl.calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]
