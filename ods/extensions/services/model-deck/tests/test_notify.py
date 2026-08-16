"""Tests for app.notify.notify_engine — post-move engine registration hooks."""
import pytest

from app.engines import EngineError, GuardError
from app.events import tail_events
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


class _SelectiveDockerCtl:
    """Fails start() for exactly ONE named container, succeeds for every
    other — the fixture the multi-resource failure-isolation test below
    needs: proof that a SIBLING resource's restart still runs to
    completion rather than being abandoned by an earlier resource's
    failure (E1 Task 9 review fix)."""

    def __init__(self, fail_container: str, exc: Exception):
        self.calls = []
        self._fail_container = fail_container
        self._exc = exc

    def stop(self, name):
        self.calls.append(("stop", name))

    def start(self, name):
        self.calls.append(("start", name))
        if name == self._fail_container:
            raise self._exc


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


def _deck(tmp_path, loaded=None, dockerctl=None, engines=None, clients=None):
    # events_path always wired (E1 Task 9 review fix): notify_engine now
    # logs a resource-scoped event on a restart failure, so every test
    # needs a real, writable path even the ones that never trigger it.
    engines = [_LEMONADE_ENTRY] if engines is None else engines
    clients = {"lemonade": _Lemonade(loaded)} if clients is None else clients
    return {
        "node_store": _NodeStore(engines),
        "local_clients": _LocalClients(clients),
        "dockerctl": dockerctl or _DockerCtl(),
        "events_path": tmp_path / "events.jsonl",
    }


def _loc(engine):
    return {"name": "x", "engine": engine}


def test_lemonade_idle_restarts_container(tmp_path):
    deck = _deck(tmp_path, loaded=None)
    assert notify_engine(_loc("lemonade"), deck) is None
    assert deck["dockerctl"].calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]


def test_lemonade_loaded_defers_with_warning(tmp_path):
    deck = _deck(tmp_path, loaded="extra.a.gguf")
    warning = notify_engine(_loc("lemonade"), deck)
    assert warning and "restart deferred" in warning
    assert deck["dockerctl"].calls == []


def test_comfyui_and_none_are_noops(tmp_path):
    deck = _deck(tmp_path)
    assert notify_engine(_loc("comfyui"), deck) is None
    assert notify_engine(_loc("none"), deck) is None
    assert deck["dockerctl"].calls == []


def test_lemonade_stop_timeout_then_start_succeeds_returns_none(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr("app.notify.time.sleep", lambda s: slept.append(s))
    dockerctl = _DockerCtl(stop_exc=EngineError("timed out"))
    deck = _deck(tmp_path, loaded=None, dockerctl=dockerctl)

    result = notify_engine(_loc("lemonade"), deck)

    assert result is None
    assert dockerctl.calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]
    assert slept == [10]


def test_lemonade_stop_timeout_then_start_also_fails_raises_and_logs(tmp_path, monkeypatch):
    """A real double failure still propagates (Let It Crash) — AND is no
    longer invisible (E1 Task 9 review fix): a resource-scoped
    'notify-restart-failed' event names which resource/container it was."""
    monkeypatch.setattr("app.notify.time.sleep", lambda s: None)
    dockerctl = _DockerCtl(
        stop_exc=EngineError("stop timed out"),
        start_exc=EngineError("start timed out"),
    )
    deck = _deck(tmp_path, loaded=None, dockerctl=dockerctl)

    with pytest.raises(EngineError, match="start timed out"):
        notify_engine(_loc("lemonade"), deck)

    assert dockerctl.calls == [("stop", "ods-llama-server"), ("start", "ods-llama-server")]

    events = tail_events(deck["events_path"])
    failures = [e for e in events if e["kind"] == "notify-restart-failed"]
    assert len(failures) == 1
    assert failures[0]["detail"]["resource"] == "lemonade"
    assert failures[0]["detail"]["container"] == "ods-llama-server"
    assert "start timed out" in failures[0]["detail"]["error"]


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


def test_two_declared_lemonade_resources_both_restart(tmp_path):
    """With TWO lemonade-kind resources declared, a moved-in GGUF restarts
    BOTH containers — each declares its own (T6 review class obligation)."""
    deck = _deck(
        tmp_path,
        engines=[_LEMONADE_ENTRY, _gguf_b_entry()],
        clients={"lemonade": _Lemonade(None), "gguf-b": _Lemonade(None)},
    )

    assert notify_engine(_loc("lemonade"), deck) is None

    assert deck["dockerctl"].calls == [
        ("stop", "ods-llama-server"), ("start", "ods-llama-server"),
        ("stop", "ods-gguf-b"), ("start", "ods-gguf-b"),
    ]


def test_one_loaded_sibling_still_restarts_independently(tmp_path):
    """A resource with a model loaded defers (never yanks); a SIBLING
    resource with nothing loaded still restarts on its own — the defer on
    one must not block the other."""
    deck = _deck(
        tmp_path,
        engines=[_LEMONADE_ENTRY, _gguf_b_entry()],
        clients={"lemonade": _Lemonade("extra.a.gguf"), "gguf-b": _Lemonade(None)},
    )

    warning = notify_engine(_loc("lemonade"), deck)

    assert warning and "restart deferred" in warning and "lemonade" in warning
    assert deck["dockerctl"].calls == [("stop", "ods-gguf-b"), ("start", "ods-gguf-b")]


def test_undeclared_box_is_a_noop(tmp_path):
    """No declared engines at all: notify_engine must not KeyError, just do
    nothing (same "empty declaration blocks nothing" posture as
    app.storage.unit_in_use)."""
    deck = _deck(tmp_path, engines=[], clients={})
    assert notify_engine(_loc("lemonade"), deck) is None
    assert deck["dockerctl"].calls == []


# ===========================================================================
# E1 Task 9 review fix: a restart failure must not fail INVISIBLY, and one
# resource's failure must not silently determine whether its SIBLING ever
# got attempted. Chosen semantic (see app/notify.py's module docstring for
# the full rationale): CONTINUE — every declared entry gets its own
# restart attempt regardless of an earlier one's failure (isolated per
# resource, mirroring app.arbiter's _execute_restore/app.engine_kinds'
# execute_unload precedent) — then the first failure raises once every
# entry has been attempted.
# ===========================================================================


def test_first_failure_logs_and_sibling_still_restarts_then_raises(tmp_path):
    """Two declared lemonade-kind resources; the FIRST one's start() raises.
    Pins the chosen semantic explicitly: the failing resource shows BOTH a
    stop AND a start attempt in dockerctl.calls (it failed MID-restart, not
    before ever trying), the SIBLING shows its own complete stop+start pair
    (it was NOT abandoned — "never attempted" is what halt-and-log would
    have produced instead, and does not happen here), the failure still
    propagates to the caller, and the event log names exactly which
    resource/container failed."""
    dockerctl = _SelectiveDockerCtl(
        fail_container="ods-llama-server", exc=EngineError("start timed out")
    )
    deck = _deck(
        tmp_path,
        engines=[_LEMONADE_ENTRY, _gguf_b_entry()],
        clients={"lemonade": _Lemonade(None), "gguf-b": _Lemonade(None)},
        dockerctl=dockerctl,
    )

    with pytest.raises(EngineError, match="start timed out"):
        notify_engine(_loc("lemonade"), deck)

    # The failing resource: attempted mid-restart (stop AND start both ran).
    assert ("stop", "ods-llama-server") in dockerctl.calls
    assert ("start", "ods-llama-server") in dockerctl.calls
    # The sibling: NOT abandoned — its own full stop+start pair still ran,
    # proving isolation rather than an aborted loop (distinguishes this
    # from "never attempted", which a halt-and-log semantic would show as
    # dockerctl.calls containing no "ods-gguf-b" entries at all).
    assert dockerctl.calls.count(("stop", "ods-gguf-b")) == 1
    assert dockerctl.calls.count(("start", "ods-gguf-b")) == 1

    events = tail_events(deck["events_path"])
    failures = [e for e in events if e["kind"] == "notify-restart-failed"]
    assert len(failures) == 1
    assert failures[0]["detail"]["resource"] == "lemonade"
    assert failures[0]["detail"]["container"] == "ods-llama-server"
    assert "start timed out" in failures[0]["detail"]["error"]


# ===========================================================================
# E1 final-review item 3a: GuardError is deliberately NOT an EngineError
# subclass (app/engines/__init__.py:30-38) — a container outside
# settings.park_allowlist makes DockerCtl.stop()/start() raise GuardError
# (app/engines/docker_ctl.py:197-199), not EngineError. Before this fix the
# per-resource try/except above named EngineError only, so a park-allowlist
# refusal escaped it entirely: the loop aborted mid-way, every SIBLING's
# restart was skipped, and no notify-restart-failed event was ever logged
# for the resource that actually failed.
# ===========================================================================


def test_first_failure_guard_error_isolates_sibling_and_logs(tmp_path):
    """Mirrors test_first_failure_logs_and_sibling_still_restarts_then_raises
    above, GuardError substituted for EngineError — same shape, same
    assertions: the failing resource shows a full stop+start attempt (it
    failed mid-restart, not before ever trying — DockerCtl's real _guard()
    fires inside stop()/start() themselves), the SIBLING shows its own
    complete stop+start pair (not abandoned), the failure still propagates,
    and the event log names exactly which resource/container was refused.
    Fixture resources are gguf-a/gguf-b (away from live topology), not the
    seeded lemonade/hipfire/comfyui triple — the point is general to any
    declared resource, not specific to one name."""
    gguf_a = _gguf_b_entry(resource="gguf-a",
                           connection={**_LEMONADE_ENTRY["connection"],
                                       "container": "ods-gguf-a"})
    gguf_b = _gguf_b_entry()
    dockerctl = _SelectiveDockerCtl(
        fail_container="ods-gguf-a",
        exc=GuardError("container 'ods-gguf-a' is not in the park allowlist"),
    )
    deck = _deck(
        tmp_path,
        engines=[gguf_a, gguf_b],
        clients={"gguf-a": _Lemonade(None), "gguf-b": _Lemonade(None)},
        dockerctl=dockerctl,
    )

    with pytest.raises(GuardError, match="park allowlist"):
        notify_engine(_loc("lemonade"), deck)

    # The failing resource: attempted mid-restart (stop AND start both ran).
    assert ("stop", "ods-gguf-a") in dockerctl.calls
    assert ("start", "ods-gguf-a") in dockerctl.calls
    # The sibling: NOT abandoned — its own full stop+start pair still ran.
    assert dockerctl.calls.count(("stop", "ods-gguf-b")) == 1
    assert dockerctl.calls.count(("start", "ods-gguf-b")) == 1

    events = tail_events(deck["events_path"])
    failures = [e for e in events if e["kind"] == "notify-restart-failed"]
    assert len(failures) == 1
    assert failures[0]["detail"]["resource"] == "gguf-a"
    assert failures[0]["detail"]["container"] == "ods-gguf-a"
    assert "park allowlist" in failures[0]["detail"]["error"]
