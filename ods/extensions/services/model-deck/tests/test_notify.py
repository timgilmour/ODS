"""Tests for app.notify.notify_engine — post-move engine registration hooks."""
from app.notify import notify_engine


class _Lemonade:
    def __init__(self, loaded=None):
        self._loaded = loaded
    def status(self):
        return {"loaded": self._loaded}


class _DockerCtl:
    def __init__(self):
        self.calls = []
    def stop(self, name):
        self.calls.append(("stop", name))
    def start(self, name):
        self.calls.append(("start", name))


class _Settings:
    lemonade_container = "ods-llama-server"


def _deck(loaded=None):
    return {"lemonade": _Lemonade(loaded), "dockerctl": _DockerCtl(), "settings": _Settings()}


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
