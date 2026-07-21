"""Tests for app.engines — Lemonade, ComfyUI, and litellm HTTP clients.

Every client accepts a `transport=` kwarg (an httpx.BaseTransport) so tests
can inject httpx.MockTransport instead of touching the network. Handlers
below capture the outgoing httpx.Request so body/headers/method can be
asserted, mirroring the real wire contracts documented in the task brief.
"""

import json

import httpx
import pytest

from app.engines import BusyError, EngineError, GuardError
from app.engines.comfyui import ComfyClient
from app.engines.docker_ctl import DockerCtl
from app.engines.hipfire import HipfireClient
from app.engines.hostagent import HostAgent
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


# --- DockerCtl ---


def _dockerctl(handler, allowlist=("ods-hipfire",)):
    return DockerCtl("http://docker-ctl:2375", list(allowlist), transport=_transport(handler))


def test_dockerctl_stop_raises_guarderror_naming_container_when_not_allowlisted():
    handler = _recording_handler(204, {})
    ctl = _dockerctl(handler)

    with pytest.raises(GuardError, match="ods-comfyui"):
        ctl.stop("ods-comfyui")

    assert handler.calls == []  # no HTTP call made


def test_dockerctl_start_raises_guarderror_naming_container_when_not_allowlisted():
    handler = _recording_handler(204, {})
    ctl = _dockerctl(handler)

    with pytest.raises(GuardError, match="ods-comfyui"):
        ctl.start("ods-comfyui")

    assert handler.calls == []  # no HTTP call made


def test_dockerctl_stop_hits_correct_path():
    handler = _recording_handler(204, {})
    ctl = _dockerctl(handler)

    result = ctl.stop("ods-hipfire")

    assert result is None
    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "POST"
    assert req.url.path == "/containers/ods-hipfire/stop"


def test_dockerctl_start_hits_correct_path():
    handler = _recording_handler(204, {})
    ctl = _dockerctl(handler)

    result = ctl.start("ods-hipfire")

    assert result is None
    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "POST"
    assert req.url.path == "/containers/ods-hipfire/start"


def test_dockerctl_stop_treats_304_as_success():
    def handler(request):
        return httpx.Response(304, text="already stopped", request=request)

    ctl = _dockerctl(handler)

    assert ctl.stop("ods-hipfire") is None


def test_dockerctl_start_treats_304_as_success():
    def handler(request):
        return httpx.Response(304, text="already running", request=request)

    ctl = _dockerctl(handler)

    assert ctl.start("ods-hipfire") is None


def test_dockerctl_stop_raises_engineerror_with_response_text_on_other_non_2xx():
    def handler(request):
        return httpx.Response(500, text="boom", request=request)

    ctl = _dockerctl(handler)

    with pytest.raises(EngineError, match="boom"):
        ctl.stop("ods-hipfire")


def test_dockerctl_stop_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.stop("ods-hipfire")


def test_dockerctl_start_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.start("ods-hipfire")


def test_dockerctl_running_true_when_state_running_true():
    handler = _json_handler(200, {"State": {"Running": True}})
    ctl = _dockerctl(handler)

    assert ctl.running("ods-hipfire") is True


def test_dockerctl_running_false_when_state_running_false():
    handler = _json_handler(200, {"State": {"Running": False}})
    ctl = _dockerctl(handler)

    assert ctl.running("ods-hipfire") is False


def test_dockerctl_running_hits_correct_path():
    handler = _recording_handler(200, {"State": {"Running": True}})
    ctl = _dockerctl(handler)

    ctl.running("ods-hipfire")

    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "GET"
    assert req.url.path == "/containers/ods-hipfire/json"


def test_dockerctl_running_raises_engineerror_on_404():
    def handler(request):
        return httpx.Response(404, text="no such container", request=request)

    ctl = _dockerctl(handler)

    with pytest.raises(EngineError, match="no such container"):
        ctl.running("ods-hipfire")


def test_dockerctl_running_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.running("ods-hipfire")


def test_dockerctl_running_not_gated_by_allowlist():
    """running() is a read; only stop()/start() enforce the allowlist."""
    handler = _json_handler(200, {"State": {"Running": True}})
    ctl = _dockerctl(handler)

    assert ctl.running("ods-comfyui") is True


# --- HipfireClient ---


def _litellm(handler):
    return LiteLLMClient("http://litellm:4000", "testkey", transport=_transport(handler))


def _litellm_default_targets_hipfire(value):
    body = json.loads(json.dumps(_MODEL_INFO_BODY))  # deep copy
    if value:
        body["data"][0]["litellm_params"]["api_base"] = "http://hipfire:11435/v1"
    return _litellm(_json_handler(200, body))


def test_hipfire_status_parked_when_container_not_running():
    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": False}}))
    litellm = _litellm_default_targets_hipfire(False)
    client = HipfireClient("http://hipfire:11435/health", dockerctl, "ods-hipfire", litellm)

    assert client.status() == "parked"


def test_hipfire_status_running_when_container_running_and_health_200():
    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": True}}))
    litellm = _litellm_default_targets_hipfire(False)

    def health_handler(request):
        return httpx.Response(200, text="ok", request=request)

    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(health_handler),
    )

    assert client.status() == "running"


def test_hipfire_status_loading_when_container_running_and_health_503():
    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": True}}))
    litellm = _litellm_default_targets_hipfire(False)

    def health_handler(request):
        return httpx.Response(503, text="loading", request=request)

    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(health_handler),
    )

    assert client.status() == "loading"


def test_hipfire_status_loading_on_other_non_2xx_health_response():
    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": True}}))
    litellm = _litellm_default_targets_hipfire(False)

    def health_handler(request):
        return httpx.Response(500, text="error", request=request)

    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(health_handler),
    )

    assert client.status() == "loading"


def test_hipfire_status_raises_engineerror_when_health_transport_fails_while_running():
    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": True}}))
    litellm = _litellm_default_targets_hipfire(False)
    health_handler = _raising_handler(httpx.ConnectError("connection refused"))
    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(health_handler),
    )

    with pytest.raises(EngineError):
        client.status()


def test_hipfire_status_does_not_check_health_when_parked():
    """Parked (container not running) short-circuits before any health GET."""
    calls = []

    def health_handler(request):
        calls.append(request)
        return httpx.Response(200, text="ok", request=request)

    dockerctl = _dockerctl(_json_handler(200, {"State": {"Running": False}}))
    litellm = _litellm_default_targets_hipfire(False)
    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(health_handler),
    )

    assert client.status() == "parked"
    assert calls == []


def test_hipfire_park_raises_guarderror_when_default_targets_hipfire_and_never_stops():
    stop_calls = []

    def dockerctl_handler(request):
        stop_calls.append(request)
        return httpx.Response(204, request=request)

    dockerctl = _dockerctl(dockerctl_handler)
    litellm = _litellm_default_targets_hipfire(True)
    client = HipfireClient("http://hipfire:11435/health", dockerctl, "ods-hipfire", litellm)

    with pytest.raises(GuardError):
        client.park()

    assert stop_calls == []  # /stop never fired


def test_hipfire_park_propagates_engineerror_when_route_check_transport_fails():
    stop_calls = []

    def dockerctl_handler(request):
        stop_calls.append(request)
        return httpx.Response(204, request=request)

    dockerctl = _dockerctl(dockerctl_handler)
    litellm = _litellm(_raising_handler(httpx.ConnectError("connection refused")))
    client = HipfireClient("http://hipfire:11435/health", dockerctl, "ods-hipfire", litellm)

    with pytest.raises(EngineError):
        client.park()

    assert stop_calls == []  # fails safe: /stop never fired


def test_hipfire_park_stops_container_when_default_does_not_target_hipfire():
    calls = []

    def dockerctl_handler(request):
        calls.append(request)
        return httpx.Response(204, request=request)

    dockerctl = _dockerctl(dockerctl_handler)
    litellm = _litellm_default_targets_hipfire(False)
    client = HipfireClient("http://hipfire:11435/health", dockerctl, "ods-hipfire", litellm)

    result = client.park()

    assert result is None
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/containers/ods-hipfire/stop"


def test_hipfire_resume_starts_container_and_does_not_poll_status():
    calls = []

    def dockerctl_handler(request):
        calls.append(request)
        return httpx.Response(204, request=request)

    dockerctl = _dockerctl(dockerctl_handler)
    litellm = _litellm_default_targets_hipfire(False)
    client = HipfireClient("http://hipfire:11435/health", dockerctl, "ods-hipfire", litellm)

    result = client.resume()

    assert result is None
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/containers/ods-hipfire/start"


# --- HostAgent.activate() ---


def test_hostagent_activate_posts_correct_path_body_and_auth_header():
    handler = _recording_handler(200, {"status": "activated"})
    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    agent.activate("extra.Qwen3.5-27B-Q4_K_M.gguf")

    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/model/activate"
    assert req.headers["authorization"] == "Bearer testkey"
    assert json.loads(req.content) == {"model_id": "extra.Qwen3.5-27B-Q4_K_M.gguf"}


def test_hostagent_activate_returns_parsed_json_on_200():
    handler = _json_handler(200, {"status": "activated", "model_id": "extra.foo.gguf"})
    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    assert agent.activate("extra.foo.gguf") == {"status": "activated", "model_id": "extra.foo.gguf"}


def test_hostagent_activate_raises_busyerror_on_409():
    handler = _json_handler(409, {"error": "activation already in progress"})
    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    with pytest.raises(BusyError):
        agent.activate("extra.foo.gguf")


def test_hostagent_activate_raises_engineerror_on_500():
    def handler(request):
        return httpx.Response(500, text="internal error", request=request)

    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError, match="internal error"):
        agent.activate("extra.foo.gguf")


def test_hostagent_activate_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    with pytest.raises(EngineError):
        agent.activate("extra.foo.gguf")


def test_hostagent_read_timeout_is_600_seconds():
    handler = _json_handler(200, {"status": "activated"})
    agent = HostAgent("http://host-agent:7331", "testkey", transport=_transport(handler))

    timeout = agent._client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 600.0
    assert timeout.write == 30.0
    assert timeout.pool == 5.0
