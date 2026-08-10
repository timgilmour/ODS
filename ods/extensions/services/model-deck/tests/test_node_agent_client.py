"""tests/test_node_agent_client.py — the observe-only node-agent client.

Wire shapes come from extensions/services/node-agent/app.py:81-107:
  GET /v1/node/info    -> {name, hostname, platform, capabilities, gpus}
  GET /v1/node/gpu     -> {backend, gpus, error}
  GET /v1/node/serving -> {model, endpoint_ok, container_status}
"""
import httpx
import pytest

from app.engines import EngineError
from app.engines.node_agent import NodeAgentClient, NodeAgentUnreachable


def _client(handler):
    return NodeAgentClient("http://hera:7720", "s3cret",
                           transport=httpx.MockTransport(handler))


def test_gpu_hits_the_endpoint_with_the_bearer_key():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"backend": "cuda", "gpus": [], "error": None})

    body = _client(handler).gpu()
    assert seen["path"] == "/v1/node/gpu"
    assert seen["auth"] == "Bearer s3cret"
    assert body["backend"] == "cuda"


def test_transport_failure_is_unreachable_not_plain_engine_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(NodeAgentUnreachable):
        _client(handler).gpu()


def test_non_2xx_is_engine_error_but_not_unreachable():
    def handler(request):
        return httpx.Response(500, text="collector exploded")

    with pytest.raises(EngineError) as exc_info:
        _client(handler).serving()
    assert not isinstance(exc_info.value, NodeAgentUnreachable)


def test_info_and_serving_paths():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    c = _client(handler)
    c.info()
    c.serving()
    assert paths == ["/v1/node/info", "/v1/node/serving"]


def test_200_with_non_json_body_is_engine_error():
    """2xx response with non-JSON body raises EngineError, not JSONDecodeError."""
    def handler(request):
        return httpx.Response(200, text="not json")

    with pytest.raises(EngineError) as exc_info:
        _client(handler).gpu()
    assert "non-JSON response" in str(exc_info.value)
    assert not isinstance(exc_info.value, NodeAgentUnreachable)
