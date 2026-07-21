"""Tests for app.engines — Lemonade, ComfyUI, and litellm HTTP clients.

Every client accepts a `transport=` kwarg (an httpx.BaseTransport) so tests
can inject httpx.MockTransport instead of touching the network. Handlers
below capture the outgoing httpx.Request so body/headers/method can be
asserted, mirroring the real wire contracts documented in the task brief.
"""

import json

import httpx
import pytest

from app.engines import EngineError, GuardError
from app.engines.comfyui import ComfyClient
from app.engines.lemonade import LemonadeClient
from app.engines.litellm import LiteLLMClient


def _transport(handler):
    return httpx.MockTransport(handler)


def _json_handler(status_code, body):
    def handler(request):
        return httpx.Response(status_code, json=body, request=request)

    return handler


def _recording_handler(status_code, body):
    """Handler that also appends every request it sees to a list."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status_code, json=body, request=request)

    handler.calls = calls
    return handler


def _raising_handler(exc):
    def handler(request):
        raise exc

    return handler


# --- EngineError / GuardError shape ---


def test_guarderror_is_not_an_engineerror():
    assert not issubclass(GuardError, EngineError)


# --- LemonadeClient.status() ---


def test_lemonade_status_parses_loaded_model():
    handler = _json_handler(200, {"model_loaded": "extra.Qwen3.5-27B-Q4_K_M.gguf", "other": "ignored"})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.status() == {"loaded": "extra.Qwen3.5-27B-Q4_K_M.gguf"}


def test_lemonade_status_parses_none_when_no_model_loaded():
    handler = _json_handler(200, {"model_loaded": None})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.status() == {"loaded": None}


def test_lemonade_status_hits_health_endpoint_with_auth_header():
    handler = _recording_handler(200, {"model_loaded": None})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    client.status()

    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "GET"
    assert req.url.path == "/api/v1/health"
    assert req.headers["authorization"] == "Bearer testkey"


def test_lemonade_status_raises_engineerror_on_non_2xx():
    handler = _json_handler(500, {"error": "boom"})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.status()


def test_lemonade_status_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.status()


# --- LemonadeClient.load() / unload() ---


def test_lemonade_load_posts_correct_path_body_and_auth_header():
    handler = _recording_handler(200, {})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    result = client.load("extra.Qwen3.5-27B-Q4_K_M.gguf")

    assert result is None
    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "POST"
    assert req.url.path == "/api/v1/load"
    assert json.loads(req.content) == {"model_name": "extra.Qwen3.5-27B-Q4_K_M.gguf"}
    assert req.headers["authorization"] == "Bearer testkey"


def test_lemonade_unload_posts_correct_path_and_body():
    handler = _recording_handler(200, {})
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    client.unload("extra.Qwen3.5-27B-Q4_K_M.gguf")

    req = handler.calls[0]
    assert req.method == "POST"
    assert req.url.path == "/api/v1/unload"
    assert json.loads(req.content) == {"model_name": "extra.Qwen3.5-27B-Q4_K_M.gguf"}


def test_lemonade_load_raises_engineerror_with_response_text_on_non_2xx():
    def handler(request):
        return httpx.Response(422, text="model not found", request=request)

    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError, match="model not found"):
        client.load("does-not-exist.gguf")


def test_lemonade_unload_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.unload("some-model.gguf")


# --- LemonadeClient.activity() ---


def test_lemonade_activity_sums_prompt_and_predicted_tokens():
    metrics_text = (
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
        "# TYPE llamacpp:prompt_tokens_total counter\n"
        "llamacpp:prompt_tokens_total 1234.0\n"
        "# HELP llamacpp:tokens_predicted_total Number of tokens predicted.\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        "llamacpp:tokens_predicted_total 5678.0\n"
        "llamacpp:some_other_metric_total 999.0\n"
    )

    def handler(request):
        return httpx.Response(200, text=metrics_text, request=request)

    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.activity() == 1234 + 5678


def test_lemonade_activity_returns_none_when_neither_metric_present():
    metrics_text = "llamacpp:some_other_metric_total 999.0\n"

    def handler(request):
        return httpx.Response(200, text=metrics_text, request=request)

    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.activity() is None


def test_lemonade_activity_returns_none_on_transport_error():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.activity() is None


def test_lemonade_activity_returns_none_on_non_2xx():
    def handler(request):
        return httpx.Response(503, text="unavailable", request=request)

    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.activity() is None


def test_lemonade_activity_returns_none_on_malformed_metric_value():
    metrics_text = "llamacpp:prompt_tokens_total notanumber\n"

    def handler(request):
        return httpx.Response(200, text=metrics_text, request=request)

    client = LemonadeClient("http://lemonade:8000", "testkey", transport=_transport(handler))

    assert client.activity() is None


# --- ComfyClient.queue_len() ---


def test_comfy_queue_len_sums_running_and_pending():
    handler = _json_handler(200, {"queue_running": [["a"]], "queue_pending": [["b"], ["c"]]})
    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    assert client.queue_len() == 3


def test_comfy_queue_len_zero_when_both_empty():
    handler = _json_handler(200, {"queue_running": [], "queue_pending": []})
    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    assert client.queue_len() == 0


def test_comfy_queue_len_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.queue_len()


def test_comfy_client_sends_no_auth_header():
    handler = _recording_handler(200, {"queue_running": [], "queue_pending": []})
    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    client.queue_len()

    assert "authorization" not in handler.calls[0].headers


# --- ComfyClient.free() ---


def test_comfy_free_posts_exact_body_when_queue_empty():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []}, request=request)
        return httpx.Response(200, json={}, request=request)

    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    result = client.free()

    assert result is None
    assert [c.url.path for c in calls] == ["/queue", "/free"]
    free_req = calls[1]
    assert free_req.method == "POST"
    assert json.loads(free_req.content) == {"unload_models": True, "free_memory": True}


def test_comfy_free_raises_guarderror_when_queue_nonempty_and_does_not_post_free():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"queue_running": [["job"]], "queue_pending": []}, request=request)

    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    with pytest.raises(GuardError):
        client.free()

    assert [c.url.path for c in calls] == ["/queue"]  # /free never fired


def test_comfy_free_raises_engineerror_when_queue_check_fails_transport():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused")

    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.free()

    assert [c.url.path for c in calls] == ["/queue"]  # fails safe: /free never fired


def test_comfy_free_raises_engineerror_on_non_2xx_free_response():
    def handler(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []}, request=request)
        return httpx.Response(500, text="free failed", request=request)

    client = ComfyClient("http://comfyui:8188", transport=_transport(handler))

    with pytest.raises(EngineError, match="free failed"):
        client.free()


# --- LiteLLMClient ---

_MODEL_INFO_BODY = {
    "data": [
        {
            "model_name": "default",
            "litellm_params": {
                "model": "openai/extra.Qwen3.5-27B-Q4_K_M.gguf",
                "api_base": "http://llama-server:8080/api/v1",
            },
        },
        {
            "model_name": "hipfire",
            "litellm_params": {
                "model": "openai/qwen36-35b-a3b.mq4",
                "api_base": "http://hipfire:11435/v1",
            },
        },
        {
            "model_name": "lemonade",
            "litellm_params": {
                "model": "openai/some-lemonade-model",
                "api_base": "http://lemonade:8000/api/v1",
            },
        },
        {
            "model_name": "*",
            "litellm_params": {
                "model": "openai/*",
                "api_base": "http://catchall:8080/v1",
            },
        },
    ]
}


def test_litellm_route_table_maps_model_name_to_litellm_model():
    handler = _json_handler(200, _MODEL_INFO_BODY)
    client = LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))

    assert client.route_table() == {
        "default": "openai/extra.Qwen3.5-27B-Q4_K_M.gguf",
        "hipfire": "openai/qwen36-35b-a3b.mq4",
        "lemonade": "openai/some-lemonade-model",
        "*": "openai/*",
    }


def test_litellm_route_table_hits_model_info_with_auth_header():
    handler = _recording_handler(200, _MODEL_INFO_BODY)
    client = LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))

    client.route_table()

    req = handler.calls[0]
    assert req.method == "GET"
    assert req.url.path == "/model/info"
    assert req.headers["authorization"] == "Bearer testkey"


def test_litellm_default_targets_hipfire_false_when_default_points_elsewhere():
    handler = _json_handler(200, _MODEL_INFO_BODY)
    client = LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))

    assert client.default_targets_hipfire() is False


def test_litellm_default_targets_hipfire_true_when_default_points_at_hipfire():
    body = json.loads(json.dumps(_MODEL_INFO_BODY))  # deep copy
    body["data"][0]["litellm_params"]["api_base"] = "http://hipfire:11435/v1"
    handler = _json_handler(200, body)
    client = LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))

    assert client.default_targets_hipfire() is True


def test_litellm_raises_engineerror_on_non_2xx():
    handler = _json_handler(401, {"error": "unauthorized"})
    client = LiteLLMClient("http://litellm:4000", "badkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.route_table()


def test_litellm_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        client.route_table()
