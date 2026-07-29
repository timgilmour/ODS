import asyncio
import json

import httpx
import pytest

import remote_nodes


NODES_ENV = json.dumps([{"name": "sparky", "display_name": "DGX Spark GB10",
                         "url": "http://sparky.test:7720",
                         "key_env": "TEST_NODE_KEY"}])

GPU = {"index": 0, "uuid": "GPU-x", "name": "GB10", "memory_used_mb": 1,
       "memory_total_mb": 2, "memory_percent": 50.0,
       "utilization_percent": 7, "temperature_c": 40}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             timeout=2.0)


def setup_function(_fn):
    remote_nodes._STATE.clear()


def test_load_nodes_parses_env(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")
    nodes = remote_nodes.load_remote_nodes()
    assert len(nodes) == 1
    assert nodes[0].name == "sparky"
    assert nodes[0].key == "sekrit"


def test_load_nodes_malformed_returns_empty(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", "{not json")
    assert remote_nodes.load_remote_nodes() == []


def test_load_nodes_absent_returns_empty(monkeypatch):
    monkeypatch.delenv("ODS_REMOTE_NODES", raising=False)
    assert remote_nodes.load_remote_nodes() == []


@pytest.mark.asyncio
async def test_poll_online(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer sekrit"
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia",
                                             "gpus": [GPU]})
        if request.url.path == "/v1/node/serving":
            return httpx.Response(200, json={"model": "heretic",
                                             "endpoint_ok": True,
                                             "container_status": "running"})
        return httpx.Response(404)

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.platform == "nvidia"
    assert status.gpus[0].utilization_percent == 7
    assert status.serving.model == "heretic"
    assert status.last_seen is not None


@pytest.mark.asyncio
async def test_poll_offline_preserves_last_seen(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def up(request):
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(up) as client:
        await remote_nodes.poll_all_nodes_once(client)
    seen = remote_nodes.get_remote_node_statuses()[0].last_seen

    def down(request):
        raise httpx.ConnectError("refused")

    async with _client(down) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "offline"
    assert status.last_seen == seen
    assert status.gpus == []


@pytest.mark.asyncio
async def test_poll_auth_failure_is_error_not_offline(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        return httpx.Response(401)

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "error"
    assert "401" in status.error


@pytest.mark.asyncio
async def test_one_bad_node_does_not_block_others(monkeypatch):
    two = json.loads(NODES_ENV) + [{"name": "deadbox",
                                    "url": "http://dead.test:7720",
                                    "key_env": "TEST_NODE_KEY"}]
    monkeypatch.setenv("ODS_REMOTE_NODES", json.dumps(two))
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        if request.url.host == "dead.test":
            raise httpx.ConnectError("refused")
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    by_name = {s.name: s for s in remote_nodes.get_remote_node_statuses()}
    assert by_name["sparky"].status == "online"
    assert by_name["deadbox"].status == "offline"


@pytest.mark.asyncio
async def test_stale_state_pruned_on_name_reuse(monkeypatch):
    """A name removed from config, then reused with a different url, must not
    inherit the previous occupant's platform/last_seen (_STATE must be
    pruned, not just filtered at read time)."""
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def up(request):
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(up) as client:
        await remote_nodes.poll_all_nodes_once(client)
    assert remote_nodes.get_remote_node_statuses()[0].status == "online"

    # Node removed from config entirely; polling again must prune it from
    # internal state, not merely hide it from the getter.
    monkeypatch.setenv("ODS_REMOTE_NODES", "[]")

    def unreachable(request):
        raise AssertionError("no nodes configured; transport should be idle")

    async with _client(unreachable) as client:
        await remote_nodes.poll_all_nodes_once(client)
    assert remote_nodes.get_remote_node_statuses() == []
    assert "sparky" not in remote_nodes._STATE

    # Same name reused, but pointed at a DIFFERENT url that refuses to
    # connect. The first poll of the new target must be a clean "offline"
    # with no state inherited from the old occupant of this name.
    reused = json.dumps([{"name": "sparky", "url": "http://sparky2.test:7720",
                          "key_env": "TEST_NODE_KEY"}])
    monkeypatch.setenv("ODS_REMOTE_NODES", reused)

    def refused(request):
        raise httpx.ConnectError("refused")

    async with _client(refused) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "offline"
    assert status.last_seen is None
    assert status.platform == "unknown"


@pytest.mark.asyncio
async def test_serving_probe_failure_does_not_demote_status(monkeypatch):
    """The serving probe is auxiliary: any failure on it (transport error,
    timeout, bad response) degrades to serving=None without touching the
    node's online status or error field, which are governed by the GPU
    endpoint alone."""
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia", "gpus": [GPU]})
        if request.url.path == "/v1/node/serving":
            raise httpx.ReadTimeout("serving probe timed out")
        return httpx.Response(404)

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "online"
    assert status.serving is None
    assert status.error is None


@pytest.mark.asyncio
async def test_poll_malformed_gpu_body_is_error(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODES_ENV)
    monkeypatch.setenv("TEST_NODE_KEY", "sekrit")

    def handler(request):
        if request.url.path == "/v1/node/gpu":
            return httpx.Response(200, json={"backend": "nvidia",
                                             "gpus": [{"bogus": True}]})
        return httpx.Response(200, json={"model": None, "endpoint_ok": False,
                                         "container_status": None})

    async with _client(handler) as client:
        await remote_nodes.poll_all_nodes_once(client)
    (status,) = remote_nodes.get_remote_node_statuses()
    assert status.status == "error"
    assert status.error is not None
