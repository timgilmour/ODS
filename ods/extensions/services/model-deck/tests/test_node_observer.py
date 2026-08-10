"""tests/test_node_observer.py — snapshot semantics, zero network.

Status vocabulary (the UI compares these strings — vocabulary rule):
online | offline | error | unconfigured, produced ONLY here.
"""
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
    """Per-entry isolation: non-dict elements and missing id/address don't abort
    the whole tick. Well-formed siblings are still probed and snapshot swaps."""
    # Add a second well-formed node
    store.add({"id": "atlas", "label": "Atlas Box", "agent_kind": "node-agent",
               "address": "http://atlas:7720"}, credential="s3cr3t")
    # Inject malformed entries directly into the persisted list
    data = store._load()
    data.append("not a dict")  # non-dict element
    data.append({"id": "bad-entry", "label": "Bad", "agent_kind": "node-agent"})
    # missing address
    store._save(data)
    # Probe with a client that tracks calls
    obs = _observer(store, tmp_path, FakeClient())
    obs.tick()
    snap = obs.snapshot()
    # Both well-formed nodes should be present (hera and atlas)
    assert "hera" in snap
    assert "atlas" in snap
    assert snap["hera"]["status"] == "online"
    assert snap["atlas"]["status"] == "online"
    # Malformed entries should not appear
    assert "bad-entry" not in snap
    # Only the two well-formed nodes should have been probed
    assert len(obs.calls) == 2
    assert ("http://hera:7720", "s3cret") in obs.calls
    assert ("http://atlas:7720", "s3cr3t") in obs.calls
