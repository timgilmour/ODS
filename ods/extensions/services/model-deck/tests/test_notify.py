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


class _NodeStore:
    """Stand-in for app.node_store.NodeStore: just enough for notify_engine
    to read the local declaration live (deck["node_store"].get("local")
    ["engines"]) — mirrors tests/test_api.py's _declare_local convention
    without pulling in the real file-backed store."""

    def __init__(self, engines):
        self._engines = engines

    def get(self, node_id):
        if node_id != "local":
            return None
        return {"engines": self._engines}


class _LocalClients:
    """Stand-in for app.local_clients.LocalClients: resource -> client dict
    lookup, mirroring tests/test_api.py's FakeLocalClients (live off the
    dict, not captured at construction)."""

    def __init__(self, clients):
        self._clients = clients

    def client_for(self, resource):
        return self._clients.get(resource)


# E1 Task 9: mirrors tests/test_api.py's _GGUF_A_ENTRY shape, resource name
# kept as "lemonade" here (unlike that file's "gguf-a") so this module's
# pre-E1 single-resource tests below stay byte-identical in intent — only
# the plumbing they go through changed, not what they assert.
_LEMONADE_ENTRY = {
    "resource": "lemonade", "kind": "lemonade",
    "connection": {"url": "http://llama-server:8080",
                   "metrics_url": "http://llama-server:8001/metrics",
                   "container": "ods-llama-server"},
    "gpu_index": 1,
    "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900},
}


def _deck(loaded=None, dockerctl=None, engines=None, clients=None):
    engines = [_LEMONADE_ENTRY] if engines is None else engines
    clients = {"lemonade": _Lemonade(loaded)} if clients is None else clients
    return {
        "node_store": _NodeStore(engines),
        "local_clients": _LocalClients(clients),
        "dockerctl": dockerctl or _DockerCtl(),
    }


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


# ===========================================================================
# E1 Task 9 (T6 review class): the restart hook iterates every DECLARED
# engine whose kind matches the destination's, resolving each one's OWN
# container from ITS declared connection — never a single settings-level
# alias. Two lemonade-kind resources restart their own containers
# independently.
# ===========================================================================


def _gguf_b_entry(**over):
    entry = {**_LEMONADE_ENTRY, "resource": "gguf-b",
            "connection": {**_LEMONADE_ENTRY["connection"], "container": "ods-gguf-b"}}
    entry.update(over)
    return entry


def test_two_declared_lemonade_resources_both_restart():
    """With TWO lemonade-kind resources declared, a moved-in GGUF restarts
    BOTH containers — each declares its own (T6 review class obligation)."""
    deck = _deck(
        engines=[_LEMONADE_ENTRY, _gguf_b_entry()],
        clients={"lemonade": _Lemonade(None), "gguf-b": _Lemonade(None)},
    )

    assert notify_engine(_loc("lemonade"), deck) is None

    assert deck["dockerctl"].calls == [
        ("stop", "ods-llama-server"), ("start", "ods-llama-server"),
        ("stop", "ods-gguf-b"), ("start", "ods-gguf-b"),
    ]


def test_one_loaded_sibling_still_restarts_independently():
    """A resource with a model loaded defers (never yanks); a SIBLING
    resource with nothing loaded still restarts on its own — the defer on
    one must not block the other."""
    deck = _deck(
        engines=[_LEMONADE_ENTRY, _gguf_b_entry()],
        clients={"lemonade": _Lemonade("extra.a.gguf"), "gguf-b": _Lemonade(None)},
    )

    warning = notify_engine(_loc("lemonade"), deck)

    assert warning and "restart deferred" in warning and "lemonade" in warning
    assert deck["dockerctl"].calls == [("stop", "ods-gguf-b"), ("start", "ods-gguf-b")]


def test_undeclared_box_is_a_noop():
    """No declared engines at all: notify_engine must not KeyError, just do
    nothing (same "empty declaration blocks nothing" posture as
    app.storage.unit_in_use)."""
    deck = _deck(engines=[], clients={})
    assert notify_engine(_loc("lemonade"), deck) is None
    assert deck["dockerctl"].calls == []
