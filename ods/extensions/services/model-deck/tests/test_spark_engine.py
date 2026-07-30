"""Tests for app.engines.spark — remote single-slot swap client for sparky.

Same transport-injection style as test_engines.py: the client takes
`node_transport=` / `serving_transport=` kwargs, and the litellm guard is
exercised through a real LiteLLMClient wired to its own MockTransport.
"""

import json

import httpx
import pytest

from app.engines import BusyError, EngineError, GuardError
from app.engines.litellm import LiteLLMClient
from app.engines.spark import SparkClient

NODE_URL = "http://sparky:7720"
SERVING_URL = "http://sparky:8000"


def _litellm(default_api_base="http://hipfire:11435/v1"):
    body = {"data": [
        {"model_name": "default",
         "litellm_params": {"model": "openai/x", "api_base": default_api_base}},
    ]}

    def handler(request):
        return httpx.Response(200, json=body, request=request)

    return LiteLLMClient("http://litellm:4000", "k",
                         transport=httpx.MockTransport(handler))


def _node_handler(profiles=("laguna", "mm27b"), swap_status=None,
                  serving=None, swap_response=(202, {"id": "u1"})):
    calls = []
    serving = serving or {"model": "aeon", "endpoint_ok": True,
                          "container_status": None}

    def handler(request):
        calls.append(request)
        path = request.url.path
        if path == "/v1/node/profiles":
            return httpx.Response(200, json={
                "profiles": list(profiles), "swap_status": swap_status},
                request=request)
        if path == "/v1/node/serving":
            return httpx.Response(200, json=serving, request=request)
        if path == "/v1/node/swap":
            code, body = swap_response
            return httpx.Response(code, json=body, request=request)
        return httpx.Response(404, request=request)

    handler.calls = calls
    return handler


def _metrics_handler(running=0, waiting=0):
    text = (f'vllm:num_requests_running{{model="m"}} {running}.0\n'
            f'vllm:num_requests_waiting{{model="m"}} {waiting}.0\n')

    def handler(request):
        return httpx.Response(200, text=text, request=request)

    return handler


def _client(node_handler, metrics_handler=None, litellm=None):
    return SparkClient(
        node_url=NODE_URL,
        node_key="node-key",
        serving_url=SERVING_URL,
        litellm=litellm or _litellm(),
        node_transport=httpx.MockTransport(node_handler),
        serving_transport=httpx.MockTransport(
            metrics_handler or _metrics_handler()),
    )


# --- status ---

def test_status_combines_profiles_and_serving():
    client = _client(_node_handler())
    s = client.status()
    assert s["profiles"] == ["laguna", "mm27b"]
    assert s["swap_status"] is None
    assert s["serving"]["model"] == "aeon"
    assert s["serving"]["endpoint_ok"] is True


def test_status_sends_bearer_key_to_node():
    handler = _node_handler()
    _client(handler).status()
    assert all(r.headers["Authorization"] == "Bearer node-key"
               for r in handler.calls)


def test_status_raises_engineerror_on_node_transport_failure():
    def handler(request):
        raise httpx.ConnectError("down")
    client = _client(handler)
    with pytest.raises(EngineError):
        client.status()


# --- swap guards ---

def test_swap_posts_profile_and_returns_request_id():
    handler = _node_handler()
    client = _client(handler)
    out = client.swap("laguna")
    assert out == {"id": "u1", "profile": "laguna"}
    swap_calls = [r for r in handler.calls if r.url.path == "/v1/node/swap"]
    assert len(swap_calls) == 1
    assert json.loads(swap_calls[0].content) == {"profile": "laguna"}


def test_swap_refuses_when_litellm_default_targets_spark():
    handler = _node_handler()
    client = _client(handler,
                     litellm=_litellm(default_api_base=f"{SERVING_URL}/v1"))
    with pytest.raises(GuardError):
        client.swap("laguna")
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_litellm_guard_survives_force():
    client = _client(_node_handler(),
                     litellm=_litellm(default_api_base=f"{SERVING_URL}/v1"))
    with pytest.raises(GuardError):
        client.swap("laguna", force=True)


def test_swap_refuses_when_litellm_unreachable():
    def boom(request):
        raise httpx.ConnectError("litellm down")
    litellm = LiteLLMClient("http://litellm:4000", "k",
                            transport=httpx.MockTransport(boom))
    handler = _node_handler()
    client = _client(handler, litellm=litellm)
    with pytest.raises(EngineError):
        client.swap("laguna")
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_refuses_when_serving_busy():
    handler = _node_handler()
    client = _client(handler, metrics_handler=_metrics_handler(running=2))
    with pytest.raises(GuardError):
        client.swap("laguna")
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_force_skips_busy_guard():
    handler = _node_handler()
    client = _client(handler, metrics_handler=_metrics_handler(running=2))
    out = client.swap("laguna", force=True)
    assert out["id"] == "u1"


def test_swap_counts_waiting_requests_as_busy():
    client = _client(_node_handler(),
                     metrics_handler=_metrics_handler(waiting=1))
    with pytest.raises(GuardError):
        client.swap("laguna")


def test_swap_skips_busy_check_when_endpoint_down():
    handler = _node_handler(serving={"model": None, "endpoint_ok": False,
                                     "container_status": None})

    def metrics_boom(request):
        raise httpx.ConnectError("vllm is down, of course")

    client = _client(handler, metrics_handler=metrics_boom)
    assert client.swap("laguna")["id"] == "u1"


def test_swap_errors_when_busy_state_unknowable():
    def metrics_boom(request):
        raise httpx.ConnectError("metrics unreachable")
    client = _client(_node_handler(), metrics_handler=metrics_boom)
    with pytest.raises(EngineError):
        client.swap("laguna")


def test_swap_maps_node_409_to_busyerror():
    client = _client(_node_handler(
        swap_response=(409, {"detail": "helper is mid-swap"})))
    with pytest.raises(BusyError):
        client.swap("laguna")


def test_swap_maps_node_404_to_engineerror():
    client = _client(_node_handler(
        swap_response=(404, {"detail": "unknown profile"})))
    with pytest.raises(EngineError):
        client.swap("ghost")
