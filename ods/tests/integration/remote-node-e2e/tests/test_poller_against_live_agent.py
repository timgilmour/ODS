"""dashboard-api's poller against a real node-agent container.

What this covers that dashboard-api/tests/test_remote_nodes_poller.py cannot:
those tests drive the poller through an httpx MockTransport, so the wire
payload is written by the same person, in the same file, as the code asserting
on it. A shape mismatch between the two independently-written sides -- the
agent's response model and the poller's parsing -- is invisible to them by
construction. Here the bytes come from the actual service image over an actual
TCP connection.

Everything the agent reports traces back to the stubbed nvidia-smi in
../stubs/, so the vendored collector's argv, CSV parsing and rounding all run
for real without needing a GPU on the host.
"""
import json
import os

import httpx
import pytest

import remote_nodes
from models import IndividualGPU

AGENT_URL = os.environ["NODE_AGENT_URL"]
AGENT_KEY = os.environ["NODE_AGENT_KEY"]
DEAD_URL = os.environ["DEAD_NODE_URL"]
DEGRADED_URL = os.environ["DEGRADED_NODE_URL"]
STUB_MODEL = os.environ["SERVING_STUB_MODEL"]

STUB_UUID = "GPU-e2e-0000-0000-0000-000000000000"


def _nodes_env(url=AGENT_URL, name="e2e-node"):
    return json.dumps([{"name": name, "display_name": "E2E Node", "url": url}])


def _keys_env(key=AGENT_KEY, name="e2e-node"):
    return json.dumps({name: key})


def setup_function(_fn):
    remote_nodes._STATE.clear()
    remote_nodes._WARNED_ONCE.clear()


def _client():
    return httpx.AsyncClient(timeout=remote_nodes.NODE_TIMEOUT_SECONDS)


async def _poll_once():
    async with _client() as client:
        await remote_nodes.poll_all_nodes_once(client)


@pytest.mark.asyncio
async def test_live_agent_reports_collector_output_end_to_end(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.platform == "nvidia"
    assert status.last_seen is not None
    assert status.error is None

    (gpu,) = status.gpus
    assert gpu.uuid == STUB_UUID
    assert gpu.name == "NVIDIA GB10 (e2e stub)"
    assert gpu.memory_used_mb == 12345
    assert gpu.memory_total_mb == 122880
    # Computed by the collector, not by the stub: 12345/122880 -> 10.0.
    assert gpu.memory_percent == 10.0
    assert gpu.utilization_percent == 7
    assert gpu.temperature_c == 45
    assert gpu.power_w == 30.5
    assert gpu.utilization_available is True
    assert gpu.temperature_available is True


@pytest.mark.asyncio
async def test_serving_probe_crosses_two_containers(monkeypatch):
    """node-agent -> serving-stub is a second real hop, made by the agent
    itself while answering our poll."""
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.serving is not None
    assert status.serving.model == STUB_MODEL
    assert status.serving.endpoint_ok is True
    # No Docker socket is mounted, which is the documented default.
    assert status.serving.container_status is None


@pytest.mark.asyncio
async def test_assigned_services_never_cross_the_wire(monkeypatch):
    """Regression guard for the bug only the metal deploy caught: the stub
    reports a compute process large enough that the vendored collector labels
    the GPU ["llama-server"], which on a remote node names a service on the
    *dashboard* host. node-agent must scrub it."""
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.gpus[0].assigned_services == []


@pytest.mark.asyncio
async def test_broken_collector_is_online_with_an_error(monkeypatch):
    """The third state the dashboard renders: the node answers, its vendor
    tooling does not (wedged driver, devices not exposed). Must stay online --
    it is reachable -- with the collector's message carried through, distinct
    from both offline and auth-error. Exercised against a second real agent
    container whose nvidia-smi fails."""
    monkeypatch.setenv("ODS_REMOTE_NODES",
                       _nodes_env(url=DEGRADED_URL, name="e2e-degraded"))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env(name="e2e-degraded"))
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.last_seen is not None
    assert status.gpus == []
    assert status.error is not None
    assert "collector" in status.error


@pytest.mark.asyncio
async def test_wrong_key_is_error_not_offline(monkeypatch):
    """A reachable node that rejects us must be distinguishable from one that
    is down -- the dashboard renders them differently."""
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env(key="wrong-key"))
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "error"
    assert "401" in status.error
    assert status.gpus == []


@pytest.mark.asyncio
async def test_unreachable_node_is_offline(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env(url=DEAD_URL))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "offline"


@pytest.mark.asyncio
async def test_going_offline_preserves_last_seen(monkeypatch):
    """The node keeps its identity and its last-known timestamp when it stops
    answering.

    The agent is made unreachable by repointing the node at an address nothing
    answers on, rather than by stopping its container: the poller's in-process
    _STATE has to survive across both cycles, and giving this container the
    Docker socket to stop a sibling is the exact grant node-agent's own README
    tells operators to refuse. What the poller sees -- a refused connection
    from a node it had previously reached -- is identical either way.
    """
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()
    (online,) = remote_nodes.get_remote_node_statuses()
    assert online.status == "online"
    seen = online.last_seen
    assert seen is not None

    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env(url=DEAD_URL))
    await _poll_once()

    (offline,) = remote_nodes.get_remote_node_statuses()
    assert offline.status == "offline"
    assert offline.last_seen == seen
    assert offline.gpus == []
    assert offline.platform == "nvidia"  # carried from the last good poll


@pytest.mark.asyncio
async def test_recovers_to_online_without_a_restart(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env(url=DEAD_URL))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", _keys_env())
    await _poll_once()
    assert remote_nodes.get_remote_node_statuses()[0].status == "offline"

    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    await _poll_once()

    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.gpus[0].uuid == STUB_UUID


@pytest.mark.asyncio
async def test_a_dead_node_does_not_take_down_a_live_one(monkeypatch):
    """Per-node isolation, with a real timeout on the dead one rather than a
    mocked exception."""
    monkeypatch.setenv("ODS_REMOTE_NODES", json.dumps([
        {"name": "e2e-node", "url": AGENT_URL},
        {"name": "deadbox", "url": DEAD_URL},
    ]))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS",
                       json.dumps({"e2e-node": AGENT_KEY,
                                   "deadbox": AGENT_KEY}))
    await _poll_once()

    by_name = {s.name: s for s in remote_nodes.get_remote_node_statuses()}
    assert by_name["e2e-node"].status == "online"
    assert by_name["e2e-node"].gpus[0].uuid == STUB_UUID
    assert by_name["deadbox"].status == "offline"


@pytest.mark.asyncio
async def test_keys_file_delivery_works_against_a_live_node(tmp_path,
                                                            monkeypatch):
    """The file-based key path, proven by a real 200 rather than a parsed
    dict."""
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"e2e-node": AGENT_KEY}))
    keys.chmod(0o600)
    monkeypatch.setenv("ODS_REMOTE_NODES", _nodes_env())
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(keys))
    await _poll_once()

    assert remote_nodes.get_remote_node_statuses()[0].status == "online"


@pytest.mark.asyncio
async def test_unauthenticated_request_gets_a_bodyless_401():
    """Asserted against the real ASGI stack: a plain HTTPException would ship
    a {"detail": ...} body, which the custom handler exists to prevent."""
    async with _client() as client:
        resp = await client.get(f"{AGENT_URL}/v1/node/gpu")
    assert resp.status_code == 401
    assert resp.content == b""


@pytest.mark.asyncio
async def test_openapi_surface_is_not_served():
    async with _client() as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            resp = await client.get(f"{AGENT_URL}{path}")
            assert resp.status_code == 404, path


@pytest.mark.asyncio
async def test_wire_payload_satisfies_dashboard_api_own_model():
    """The parity test in node-agent compares field *names* between the shim
    and dashboard-api. This checks the actual bytes validate against
    dashboard-api's model, which is the assumption the poller relies on."""
    async with _client() as client:
        resp = await client.get(f"{AGENT_URL}/v1/node/gpu",
                                headers={"Authorization": f"Bearer {AGENT_KEY}"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["backend"] == "nvidia"
    assert payload["error"] is None
    gpus = [IndividualGPU(**raw) for raw in payload["gpus"]]
    assert [g.uuid for g in gpus] == [STUB_UUID]


@pytest.mark.asyncio
async def test_info_advertises_the_capabilities_seam():
    async with _client() as client:
        resp = await client.get(f"{AGENT_URL}/v1/node/info",
                                headers={"Authorization": f"Bearer {AGENT_KEY}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "e2e-node"
    assert body["platform"] == "nvidia"
    assert body["capabilities"] == ["metrics"]
    # Remote control is deliberately unbuilt; the namespace must stay empty.
    async with _client() as client:
        actions = await client.get(
            f"{AGENT_URL}/v1/node/actions/restart",
            headers={"Authorization": f"Bearer {AGENT_KEY}"})
    assert actions.status_code == 404
