"""tests/test_node_observer.py — snapshot semantics, zero network.

Status vocabulary (the UI compares these strings — vocabulary rule):
online | offline | error | unconfigured, produced ONLY here.
"""
import json

import pytest

from app.engines import EngineError
from app.engines.node_agent import NodeAgentUnreachable
from app.node_observer import NodeObserver
from app.node_store import NodeStore


class FakeClient:
    def __init__(self, gpu=None, serving=None, gpu_raises=None):
        self._gpu = gpu or {"backend": "cuda", "gpus": [{"index": 0}], "error": None}
        self._serving = serving
        self._gpu_raises = gpu_raises
        self.closed = False

    def gpu(self):
        if self._gpu_raises:
            raise self._gpu_raises
        return self._gpu

    def serving(self):
        if self._serving is None:
            raise EngineError("no serving probe")
        return self._serving

    def close(self):
        self.closed = True


@pytest.fixture
def store(tmp_path):
    s = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    s.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
           "address": "http://hera:7720"}, credential="s3cret")
    return s


def _observer(store, tmp_path, client):
    calls = []

    def factory(address, key):
        calls.append((address, key))
        return client

    obs = NodeObserver(store, tmp_path / "events.jsonl", client_factory=factory)
    obs.calls = calls
    return obs


def test_gpu_probe_governs_online(store, tmp_path):
    client = FakeClient(serving={"model": "big-model", "endpoint_ok": True,
                                 "container_status": "running"})
    obs = _observer(store, tmp_path, client)
    obs.tick()
    snap = obs.snapshot()["hera"]
    assert snap["status"] == "online"
    assert snap["last_seen"] is not None
    assert snap["gpus"] == [{"index": 0}]
    assert snap["serving"]["model"] == "big-model"
    assert obs.calls == [("http://hera:7720", "s3cret")]
    assert client.closed


def test_answers_but_reports_collector_failure_stays_online_with_message(store, tmp_path):
    # app/node_observer.py:132 `"error": gpu.get("error")` — the gpu probe
    # answered (2xx, well-formed body) but the body itself says its own
    # collector is broken. status stays "online" (the probe DID answer;
    # `error` is a passthrough of what it said, not a probe failure), and
    # the message must not be dropped. Every other FakeClient in this file
    # uses error=None, so this is the only test that can catch a dropped
    # passthrough.
    obs = _observer(store, tmp_path, FakeClient(
        gpu={"backend": "cuda", "gpus": [], "error": "collector unavailable"}))
    obs.tick()
    snap = obs.snapshot()["hera"]
    assert snap["status"] == "online"
    assert snap["error"] == "collector unavailable"


def test_serving_failure_only_degrades_serving(store, tmp_path):
    # dashboard-api remote_nodes.py:12-19 semantics: the gpu probe ALONE
    # governs status; a dead serving probe degrades serving to None.
    obs = _observer(store, tmp_path, FakeClient(serving=None))
    obs.tick()
    snap = obs.snapshot()["hera"]
    assert snap["status"] == "online"
    assert snap["serving"] is None


def test_transport_failure_is_offline_and_keeps_last_seen(store, tmp_path):
    good = FakeClient()
    obs = _observer(store, tmp_path, good)
    obs.tick()
    seen = obs.snapshot()["hera"]["last_seen"]
    obs._client_factory = lambda a, k: FakeClient(
        gpu_raises=NodeAgentUnreachable("refused"))
    obs.tick()
    snap = obs.snapshot()["hera"]
    assert snap["status"] == "offline"
    assert snap["last_seen"] == seen          # preserved, not cleared
    assert snap["gpus"] is None


def test_bad_answer_is_error_not_offline(store, tmp_path):
    obs = _observer(store, tmp_path, FakeClient(gpu_raises=EngineError("500 boom")))
    obs.tick()
    snap = obs.snapshot()["hera"]
    assert snap["status"] == "error"
    assert "boom" in snap["error"]


def test_missing_credential_is_unconfigured_and_never_probed(store, tmp_path):
    store.add({"id": "zeus", "label": "Zeus Box", "agent_kind": "node-agent",
               "address": "http://zeus:7720"})   # no credential
    obs = _observer(store, tmp_path, FakeClient())
    obs.tick()
    snap = obs.snapshot()["zeus"]
    assert snap["status"] == "unconfigured"
    assert snap["error"]                          # backend-authored sentence
    assert ("http://zeus:7720", "") not in obs.calls


def test_local_nodes_are_never_probed(store, tmp_path):
    store._save(store._load() + [{"id": "local", "label": "This Box",
                                  "agent_kind": "local", "added_ts": "t"}])
    obs = _observer(store, tmp_path, FakeClient())
    obs.tick()
    assert "local" not in obs.snapshot()
    assert all(addr != "" for addr, _ in obs.calls)


def test_registry_is_reread_every_tick(store, tmp_path):
    obs = _observer(store, tmp_path, FakeClient())
    obs.tick()
    store.remove("hera")
    obs.tick()
    assert "hera" not in obs.snapshot()   # removal is live, no restart


def test_malformed_nodes_do_not_stall_well_formed_siblings(store, tmp_path):
    """Per-entry isolation, now split across two layers:

    - NodeStore._load() gates SHAPE (app/node_store.py) — a non-dict
      element, a dict with a non-string label, and a dict with a bogus
      agent_kind never even reach the observer; they're written straight
      into the FILE (bypassing store.add()'s _validate()) to prove the
      store's own boundary gate is what drops them, not anything here.
    - The observer still gates one SEMANTIC: a node-agent entry with no
      address is shape-valid (address is legitimately optional at the
      store layer) but can't be probed, so it's skipped, not crashed on.

    Either way, well-formed siblings are still probed and the snapshot swaps.
    """
    store.add({"id": "atlas", "label": "Atlas Box", "agent_kind": "node-agent",
               "address": "http://atlas:7720"}, credential="s3cr3t")
    data = json.loads((tmp_path / "nodes.json").read_text())
    data.append("not a dict")                                     # shape: not a dict
    data.append({"id": "bad-entry", "label": "Bad",
                 "agent_kind": "node-agent"})                      # semantics: no address
    data.append({"id": "bad-label", "label": 123, "agent_kind": "node-agent",
                 "address": "http://bad-label:7720"})              # shape: label not a string
    data.append({"id": "bad-kind", "label": "Weird", "agent_kind": "vampire",
                 "address": "http://bad-kind:7720"})                # shape: bogus agent_kind
    (tmp_path / "nodes.json").write_text(json.dumps(data))

    obs = _observer(store, tmp_path, FakeClient())
    obs.tick()
    snap = obs.snapshot()
    assert "hera" in snap
    assert "atlas" in snap
    assert snap["hera"]["status"] == "online"
    assert snap["atlas"]["status"] == "online"
    # None of the malformed entries appear, however they were dropped.
    assert "bad-entry" not in snap
    assert "bad-label" not in snap
    assert "bad-kind" not in snap
    # Only the two well-formed nodes were probed.
    assert len(obs.calls) == 2
    assert ("http://hera:7720", "s3cret") in obs.calls
    assert ("http://atlas:7720", "s3cr3t") in obs.calls
