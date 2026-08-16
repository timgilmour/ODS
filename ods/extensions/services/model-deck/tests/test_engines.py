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
from app.engines.node_agent import NodeAgentUnreachable
from app.engines.sglang_omni import SglangOmniClient


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


# --- LemonadeClient.load_in_flight() ---


def test_load_in_flight_true_during_load_false_after_even_on_error(monkeypatch):
    client = LemonadeClient("http://lemonade:8000", "testkey")
    seen = {}

    def fake_request(method, path, **kwargs):
        seen["in_flight_during"] = client.load_in_flight()
        raise EngineError("boom")

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.load_in_flight() is False
    with pytest.raises(EngineError):
        client.load("m.gguf")
    assert seen["in_flight_during"] is True
    assert client.load_in_flight() is False  # cleared even on the error path


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


def test_dockerctl_stop_sends_5s_grace_period_query_param():
    handler = _recording_handler(204, {})
    ctl = _dockerctl(handler)

    ctl.stop("ods-hipfire")

    req = handler.calls[0]
    assert req.url.params["t"] == "5"


def test_dockerctl_stop_uses_extended_read_timeout():
    seen = {}

    def handler(request):
        seen.update(request.extensions.get("timeout", {}))
        return httpx.Response(204, request=request)

    ctl = _dockerctl(handler)
    ctl.stop("ods-hipfire")

    assert seen.get("read") == 30.0


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


# --- DockerCtl.image_ref() ---


def test_dockerctl_image_ref_returns_the_resolved_image_content_id():
    """The top-level `Image` field (Docker's resolved content ID for
    whatever is actually running), NOT `Config.Image` (the reference the
    container was CREATED WITH — a floating tag for hipfire, unchanged by
    a rebuild+recreate behind the same tag). See image_ref's docstring."""
    handler = _json_handler(
        200, {"Image": "sha256:abc123", "Config": {"Image": "ods-hipfire:latest"}})
    ctl = _dockerctl(handler)

    assert ctl.image_ref("ods-hipfire") == "sha256:abc123"


def test_dockerctl_image_ref_changes_when_the_running_image_is_rebuilt_behind_the_same_tag():
    """The exact scenario image_ref exists to detect: Config.Image (the
    floating tag) stays identical across a rebuild + recreate, but the
    resolved Image content ID changes."""
    before = _json_handler(
        200, {"Image": "sha256:old", "Config": {"Image": "ods-hipfire:latest"}})
    after = _json_handler(
        200, {"Image": "sha256:new", "Config": {"Image": "ods-hipfire:latest"}})

    assert _dockerctl(before).image_ref("ods-hipfire") == "sha256:old"
    assert _dockerctl(after).image_ref("ods-hipfire") == "sha256:new"


def test_dockerctl_image_ref_hits_the_same_inspect_path_as_running():
    handler = _recording_handler(200, {"Image": "sha256:x"})
    ctl = _dockerctl(handler)

    ctl.image_ref("ods-hipfire")

    req = handler.calls[0]
    assert req.method == "GET"
    assert req.url.path == "/containers/ods-hipfire/json"


def test_dockerctl_image_ref_not_gated_by_allowlist():
    """A read, same posture as running() — see its own test."""
    handler = _json_handler(200, {"Image": "sha256:x"})
    ctl = _dockerctl(handler)

    assert ctl.image_ref("ods-comfyui") == "sha256:x"


def test_dockerctl_image_ref_raises_engineerror_on_404():
    def handler(request):
        return httpx.Response(404, text="no such container", request=request)

    ctl = _dockerctl(handler)

    with pytest.raises(EngineError, match="no such container"):
        ctl.image_ref("ods-hipfire")


# --- DockerCtl.exec_run() ---


def _docker_stream_frame(stream_type: int, payload: bytes) -> bytes:
    """One frame of Docker's exec/start multiplexed stream: a stream-type
    byte + 3 reserved zero bytes + a 4-byte big-endian length, then that
    many bytes of payload."""
    return bytes([stream_type, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


def _exec_transport(exec_id="exec-1", stream=b"", create_status=201, start_status=200):
    """A handler that plays the two-call POST .../exec -> POST
    /exec/{id}/start sequence, recording both requests."""
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/containers/ods-hipfire/exec":
            return httpx.Response(create_status, json={"Id": exec_id}, request=request)
        if request.url.path == f"/exec/{exec_id}/start":
            return httpx.Response(start_status, content=stream, request=request)
        raise AssertionError(f"unexpected path {request.url.path!r}")

    handler.calls = calls
    return handler


def test_dockerctl_exec_run_raises_guarderror_when_not_allowlisted():
    handler = _exec_transport()
    ctl = _dockerctl(handler)

    with pytest.raises(GuardError, match="ods-comfyui"):
        ctl.exec_run("ods-comfyui", "python3", "print(1)")

    assert handler.calls == []  # no HTTP call made


def test_dockerctl_exec_run_creates_then_starts_with_the_given_command():
    handler = _exec_transport(stream=_docker_stream_frame(1, b"ok"))
    ctl = _dockerctl(handler)

    ctl.exec_run("ods-hipfire", "python3", "print(1)")

    assert len(handler.calls) == 2
    create, start = handler.calls
    assert create.url.path == "/containers/ods-hipfire/exec"
    assert json.loads(create.content)["Cmd"] == ["python3", "-c", "print(1)"]
    assert json.loads(create.content)["Tty"] is False
    assert start.url.path == "/exec/exec-1/start"
    assert json.loads(start.content) == {"Detach": False, "Tty": False}


def test_dockerctl_exec_run_keeps_only_stdout_frames():
    stream = (
        _docker_stream_frame(1, b"hello ")
        + _docker_stream_frame(2, b"stderr-noise-should-be-dropped")
        + _docker_stream_frame(1, b"world")
    )
    handler = _exec_transport(stream=stream)
    ctl = _dockerctl(handler)

    assert ctl.exec_run("ods-hipfire", "python3", "print(1)") == "hello world"


def test_dockerctl_exec_run_raises_engineerror_when_create_fails():
    handler = _exec_transport(create_status=409)
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.exec_run("ods-hipfire", "python3", "print(1)")


def test_dockerctl_exec_run_raises_engineerror_when_start_fails():
    handler = _exec_transport(start_status=500)
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.exec_run("ods-hipfire", "python3", "print(1)")


def test_dockerctl_exec_run_raises_engineerror_on_transport_failure():
    handler = _raising_handler(httpx.ConnectError("connection refused"))
    ctl = _dockerctl(handler)

    with pytest.raises(EngineError):
        ctl.exec_run("ods-hipfire", "python3", "print(1)")


def test_demux_stdout_stops_cleanly_on_a_truncated_trailing_frame():
    """A frame whose declared length overruns what's actually present must
    not raise -- matches this module's degrade-not-crash posture."""
    from app.engines.docker_ctl import _demux_stdout

    good = _docker_stream_frame(1, b"complete")
    truncated_header = bytes([1, 0, 0, 0]) + (999).to_bytes(4, "big") + b"short"

    assert _demux_stdout(good + truncated_header) == "complete"


# --- DockerEngineExec ---


def test_dockerengineexec_call_returns_image_ref_and_exec_stdout():
    from app.engines.docker_ctl import DockerEngineExec

    def handler(request):
        if request.url.path == "/containers/ods-hipfire/json":
            return httpx.Response(200, json={"Image": "sha256:v1"}, request=request)
        if request.url.path == "/containers/ods-hipfire/exec":
            return httpx.Response(201, json={"Id": "e1"}, request=request)
        if request.url.path == "/exec/e1/start":
            return httpx.Response(200, content=_docker_stream_frame(1, b"catalog"), request=request)
        raise AssertionError(request.url.path)

    ctl = _dockerctl(handler)
    exec_fn = DockerEngineExec(ctl, "ods-hipfire")

    version, output = exec_fn("local", "hipfire", "python3", "print(1)")

    assert version == "sha256:v1"
    assert output == "catalog"


def test_dockerengineexec_version_property_reads_image_ref_without_exec():
    from app.engines.docker_ctl import DockerEngineExec

    handler = _recording_handler(200, {"Image": "sha256:v1"})
    ctl = _dockerctl(handler)
    exec_fn = DockerEngineExec(ctl, "ods-hipfire")

    assert exec_fn.version == "sha256:v1"
    assert len(handler.calls) == 1
    assert handler.calls[0].url.path == "/containers/ods-hipfire/json"


def test_dockerengineexec_version_degrades_to_none_on_engineerror():
    from app.engines.docker_ctl import DockerEngineExec

    def handler(request):
        return httpx.Response(404, text="no such container", request=request)

    ctl = _dockerctl(handler)
    exec_fn = DockerEngineExec(ctl, "ods-hipfire")

    assert exec_fn.version is None


def test_dockerengineexec_version_degrades_to_none_on_malformed_inspect_body():
    """image_ref() raises KeyError (not EngineError) on a 2xx body missing
    the expected "Image" key -- the peek is billed best-effort and must
    degrade the same way, not propagate."""
    from app.engines.docker_ctl import DockerEngineExec

    handler = _json_handler(200, {"unexpected": "shape"})
    ctl = _dockerctl(handler)
    exec_fn = DockerEngineExec(ctl, "ods-hipfire")

    assert exec_fn.version is None


# --- exec_run -> parse_probe_output integration (realistic interleaving) ---


def test_exec_run_output_survives_interleaved_stderr_frames_into_parse_probe_output():
    """The shape production actually sees: vLLM's own logging noise on
    stderr frames interleaved with the probe's sentinel-wrapped JSON split
    across stdout frames. exec_run's demux must hand parse_probe_output
    exactly the stdout text, with stderr chatter never in the mix, and the
    result must still parse into a real catalog end-to-end."""
    from app.harvest import _SENTINEL, parse_probe_output

    from tests.test_harvest import PROBE_OUTPUT

    stream = (
        _docker_stream_frame(2, b"INFO 08-04 12:00:00 [vllm] platform banner\n")
        + _docker_stream_frame(1, (_SENTINEL + "\n").encode())
        + _docker_stream_frame(2, b"WARNING more stderr chatter mid-probe\n")
        + _docker_stream_frame(1, PROBE_OUTPUT.encode())
        + _docker_stream_frame(2, b"WARNING trailing stderr noise\n")
        + _docker_stream_frame(1, ("\n" + _SENTINEL + "\n").encode())
    )
    handler = _exec_transport(stream=stream)
    ctl = _dockerctl(handler)

    output = ctl.exec_run("ods-hipfire", "python3", "print(1)")
    catalog = parse_probe_output(output, engine_version="0.26.0", now="t")

    assert "max-model-len" in catalog["value"]["options"]
    assert "WARNING" not in output  # stderr frames never reached stdout


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
    client = _busy_client([{"queue_depth": 0, "requests_served": 0}])

    result = client.park()

    assert result is None
    assert len(client._dockerctl.calls) == 1
    assert client._dockerctl.calls[0].method == "POST"
    assert client._dockerctl.calls[0].url.path == "/containers/ods-hipfire/stop"


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


# --- HipfireClient.stats() + busy guard ---


def _dockerctl_lifecycle(running: bool):
    """DockerCtl whose GET .../json reports `running` and whose POST
    .../{stop,start} succeeds, recording lifecycle calls in .calls."""
    calls = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"State": {"Running": running}}, request=request)
        calls.append(request)
        return httpx.Response(204, request=request)

    ctl = _dockerctl(handler)
    ctl.calls = calls
    return ctl


def _stats_transport(bodies):
    """health/stats transport: /health -> 200 ok; /stats -> successive JSON
    bodies from `bodies` (last one repeats)."""
    served = {"i": 0}

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, text="ok", request=request)
        body = bodies[min(served["i"], len(bodies) - 1)]
        served["i"] += 1
        return httpx.Response(200, json=body, request=request)

    return _transport(handler)


def _busy_client(bodies, *, running=True, window_s=600.0, clock=None, route_on_hipfire=False):
    kwargs = {"transport": _stats_transport(bodies), "activity_window_s": window_s}
    if clock is not None:
        kwargs["clock"] = clock
    client = HipfireClient(
        "http://hipfire:11435/health",
        _dockerctl_lifecycle(running),
        "ods-hipfire",
        _litellm_default_targets_hipfire(route_on_hipfire),
        **kwargs,
    )
    return client


def test_hipfire_stats_returns_parsed_body():
    client = _busy_client([{"queue_depth": 1, "requests_served": 7}])

    assert client.stats() == {"queue_depth": 1, "requests_served": 7}


def test_hipfire_stats_raises_engineerror_on_transport_failure():
    dockerctl = _dockerctl_lifecycle(True)
    litellm = _litellm_default_targets_hipfire(False)
    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        litellm,
        transport=_transport(_raising_handler(httpx.ConnectError("connection refused"))),
    )

    with pytest.raises(EngineError):
        client.stats()


def test_hipfire_park_raises_guarderror_when_request_in_flight():
    client = _busy_client([{"queue_depth": 1, "requests_served": 3}])

    with pytest.raises(GuardError, match="queue_depth=1"):
        client.park()

    assert client._dockerctl.calls == []  # /stop never fired


def test_hipfire_park_force_skips_busy_guard_but_still_stops():
    client = _busy_client([{"queue_depth": 1, "requests_served": 3}])

    client.park(force=True)

    assert len(client._dockerctl.calls) == 1
    assert client._dockerctl.calls[0].url.path == "/containers/ods-hipfire/stop"


def test_hipfire_park_force_does_not_skip_route_guard():
    client = _busy_client([{"queue_depth": 0, "requests_served": 0}], route_on_hipfire=True)

    with pytest.raises(GuardError, match="default route"):
        client.park(force=True)

    assert client._dockerctl.calls == []


def test_hipfire_park_ok_when_idle_and_no_requests_ever_served():
    """requests_served == 0 means no request since daemon start — never busy,
    even on the very first observation."""
    client = _busy_client([{"queue_depth": 0, "requests_served": 0}])

    client.park()

    assert len(client._dockerctl.calls) == 1


def test_hipfire_park_busy_within_activity_window_after_observed_request():
    """A requests_served increase marks activity; parking inside the window
    is refused, and allowed again once the window has elapsed."""
    now = {"t": 100.0}
    client = _busy_client(
        [
            {"queue_depth": 0, "requests_served": 0},
            {"queue_depth": 0, "requests_served": 5},
            {"queue_depth": 0, "requests_served": 5},
            {"queue_depth": 0, "requests_served": 5},
        ],
        window_s=600.0,
        clock=lambda: now["t"],
    )

    client.stats()  # baseline: nothing served yet
    now["t"] = 200.0
    client.stats()  # observes the increase -> activity at t=200

    now["t"] = 300.0  # 100s later, inside the 600s window
    with pytest.raises(GuardError, match="activity window"):
        client.park()

    now["t"] = 900.0  # 700s after activity, outside the window
    client.park()
    assert len(client._dockerctl.calls) == 1


def test_hipfire_park_busy_on_first_sight_of_prior_requests():
    """First-ever observation with requests_served > 0: we cannot know how
    recent the traffic was, so be conservative until the window elapses."""
    now = {"t": 100.0}
    client = _busy_client(
        [{"queue_depth": 0, "requests_served": 42}],
        window_s=600.0,
        clock=lambda: now["t"],
    )

    with pytest.raises(GuardError, match="activity window"):
        client.park()

    now["t"] = 800.0  # window elapsed with no further change
    client.park()
    assert len(client._dockerctl.calls) == 1


def test_hipfire_park_ok_when_window_disabled():
    """activity_window_s=0 disables the recency rule; only queue_depth guards."""
    client = _busy_client([{"queue_depth": 0, "requests_served": 42}], window_s=0.0)

    client.park()

    assert len(client._dockerctl.calls) == 1


def test_hipfire_ensure_not_busy_skips_stats_when_parked():
    """A parked container has nothing in flight; /stats is never queried."""
    stats_calls = []

    def handler(request):
        stats_calls.append(request)
        return httpx.Response(200, json={}, request=request)

    client = HipfireClient(
        "http://hipfire:11435/health",
        _dockerctl_lifecycle(False),
        "ods-hipfire",
        _litellm_default_targets_hipfire(False),
        transport=_transport(handler),
    )

    client.ensure_not_busy("park it")
    assert stats_calls == []


def test_hipfire_ensure_not_busy_propagates_engineerror_when_stats_unreachable():
    """Running but /stats unreachable -> we can't tell -> fail safe."""
    dockerctl = _dockerctl_lifecycle(True)
    client = HipfireClient(
        "http://hipfire:11435/health",
        dockerctl,
        "ods-hipfire",
        _litellm_default_targets_hipfire(False),
        transport=_transport(_raising_handler(httpx.ConnectError("connection refused"))),
    )

    with pytest.raises(EngineError):
        client.ensure_not_busy("park it")

    assert dockerctl.calls == []


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


# --- HostAgent.lifecycle() ---


def test_hostagent_lifecycle_reports_busy():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={
        "status": "idle",
        "lifecycleActive": True,
        "activeOperation": "model_activation",
        "activeTarget": "qwen3-30b",
    }))
    agent = HostAgent("http://agent:7710", "key", transport=transport)
    assert agent.lifecycle() == {
        "active": True, "operation": "model_activation", "target": "qwen3-30b",
    }


def test_hostagent_lifecycle_idle_when_no_lifecycle_keys():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "idle"}))
    agent = HostAgent("http://agent:7710", "key", transport=transport)
    assert agent.lifecycle() == {"active": False, "operation": None, "target": None}


def test_hostagent_lifecycle_never_raises():
    def _boom(req):
        raise httpx.ConnectError("agent down")
    for transport in (
        httpx.MockTransport(_boom),
        httpx.MockTransport(lambda req: httpx.Response(500, text="err")),
        httpx.MockTransport(lambda req: httpx.Response(200, text="not json")),
    ):
        agent = HostAgent("http://agent:7710", "key", transport=transport)
        assert agent.lifecycle() == {"active": False, "operation": None, "target": None}


def test_hostagent_lifecycle_hits_correct_path_with_auth_header():
    handler = _recording_handler(200, {"status": "idle"})
    agent = HostAgent("http://agent:7710", "key", transport=_transport(handler))

    agent.lifecycle()

    assert len(handler.calls) == 1
    req = handler.calls[0]
    assert req.method == "GET"
    assert req.url.path == "/v1/model/status"
    assert req.headers["authorization"] == "Bearer key"


def test_hostagent_lifecycle_uses_short_probe_timeout():
    seen = {}

    def handler(request):
        seen.update(request.extensions.get("timeout", {}))
        return httpx.Response(200, json={"status": "idle"})

    agent = HostAgent("http://agent:7710", "key", transport=httpx.MockTransport(handler))
    agent.lifecycle()

    assert seen == {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}


def test_hostagent_lifecycle_idle_when_response_body_is_not_an_object():
    """A 2xx response whose body is valid-but-non-object JSON (null, a list,
    a bare string) must not blow up data.get(...) — treat it as idle."""
    for body in (None, [], "ok"):
        transport = httpx.MockTransport(lambda req, body=body: httpx.Response(200, json=body))
        agent = HostAgent("http://agent:7710", "key", transport=transport)
        assert agent.lifecycle() == {"active": False, "operation": None, "target": None}


def test_lemonade_activity_uses_separate_metrics_url():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "metrics-host":
            return httpx.Response(200, text="llamacpp:prompt_tokens_total 7\nllamacpp:tokens_predicted_total 3\n")
        return httpx.Response(404)

    client = LemonadeClient(
        "http://lemonade-host:8080",
        "k",
        transport=httpx.MockTransport(handler),
        metrics_url="http://metrics-host:8001/metrics",
    )
    assert client.activity() == 10
    assert seen == ["http://metrics-host:8001/metrics"]


def test_lemonade_load_uses_long_read_timeout():
    seen = {}

    def handler(request):
        seen.update(request.extensions.get("timeout", {}))
        return httpx.Response(200, json={})

    client = LemonadeClient("http://h:8080", "k", transport=httpx.MockTransport(handler))
    client.load("extra.m.gguf")
    assert seen.get("read") == 180.0


# --- _build_deck wiring (hipfire stats/activity window) ---


def test_build_deck_wires_hipfire_stats_url_and_activity_window(tmp_path, monkeypatch):
    from app.main import _build_deck
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_DECK_HIPFIRE_ACTIVITY_WINDOW_S", "123.5")
    settings = Settings()

    deck = _build_deck(settings)

    assert settings.hipfire_activity_window_s == 123.5
    assert deck["hipfire"]._activity_window_s == 123.5
    assert deck["hipfire"]._stats_url == "http://ods-hipfire:11435/stats"


# --- dual-HipfireClient cold busy-guard closure (E1 Task 6) -----------------
#
# Branch-blocking obligation carried from Task 3's review: since Task 3,
# World.snapshot polls the hipfire client built by LocalClients every tick
# (feeding HipfireClient's own conversation-activity tracker), but
# deck["hipfire"] — the SEPARATE instance app.routers.control's park route
# and app.sets' apply() guard busy-ness through via ensure_not_busy — was
# built independently in _build_deck and never polled, so its tracker sat
# permanently cold and the FIRST park after any deck start was falsely
# refused ("hipfire served a request 0s ago") until the window elapsed from
# the failed attempt. _build_deck now ALIASES deck["lemonade"]/deck["comfy"]/
# deck["hipfire"] onto whatever LocalClients builds for that resource
# whenever it's declared (see that assignment's own comment for the chosen
# mechanism and why lemonade/comfy are included too).


def test_local_clients_alias_closes_the_dual_hipfire_busy_guard_gap(tmp_path, monkeypatch):
    """The instance World.snapshot polls every tick (through
    local_clients) and the instance ensure_not_busy's recency check guards
    actuation through (deck["hipfire"]/deck["lemonade"]/deck["comfy"]) must
    be the SAME object — same identity is the simplest honest proof the
    tracker they read is the one that actually gets fed. A pre-existing
    intent.json naming "hipfire" satisfies seed_engines_if_missing's
    presence-proof gate, so the coexistence triple is declared BEFORE
    _build_deck first runs — the real, live-deployed condition this
    closes for (every currently-running box, not a brand-new install; see
    the fallback test below and
    test_build_deck_wires_hipfire_stats_url_and_activity_window above for
    that other case)."""
    from app.main import _build_deck
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    (tmp_path / "intent.json").write_text('{"local/hipfire": {"state": "loaded"}}')
    settings = Settings()

    deck = _build_deck(settings)

    assert deck["hipfire"] is deck["local_clients"].client_for("hipfire")
    assert deck["lemonade"] is deck["local_clients"].client_for("lemonade")
    assert deck["comfy"] is deck["local_clients"].client_for("comfyui")


def test_local_clients_alias_falls_back_when_nothing_is_declared(tmp_path, monkeypatch):
    """The other half of the same mechanism: a brand-new/empty declaration
    (no presence proof — fresh tmp_path, no intent.json/policy.json) must
    NOT leave deck["hipfire"] as None — every consumer (control.py's
    routes, app.sets' apply(), app.notify) expects a real, always-present
    client regardless of declaration state."""
    from app.main import _build_deck
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    settings = Settings()

    deck = _build_deck(settings)

    assert deck["local_clients"].client_for("hipfire") is None  # nothing declared
    assert deck["hipfire"] is not None
    assert deck["hipfire"]._activity_window_s == settings.hipfire_activity_window_s


# --- _build_watcher wiring: remote catalog harvest routes (task 8, C2) -----
#
# Watcher._configurable_engines does no pairing of its own anymore (see its
# docstring) -- the vocabulary decision moved here, into app.main. These
# prove the actual production wiring, not just the seam: a spark-configured
# box's one route keys off the registry node's id (LEGACY_SPARK_SEED_ID for
# the env-seed path these tests exercise), never settings.node_label
# (the C1 live-deploy bug's original vocabulary), and a box with no spark
# leaves harvest fully disabled, same as every pre-C2 build.


def _spark_env(monkeypatch, node_key="k"):
    monkeypatch.setenv("MODEL_DECK_SPARK_NODE_URL", "http://sparky:7720")
    monkeypatch.setenv("MODEL_DECK_SPARK_SERVING_URL", "http://sparky:8000")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": node_key}))


def test_build_watcher_routes_spark_catalog_by_node_id_not_label(tmp_path, monkeypatch):
    from app.engines.docker_ctl import EngineExecRouter
    from app.engines.spark import SparkCatalogExec
    from app.main import _build_watcher
    from app.node_store import LEGACY_SPARK_SEED_ID
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    # A node_label set away from its "local" default -- exactly the live
    # deploy configuration ("autarch") the original vocabulary bug needed
    # to actually show up (see Watcher._configurable_engines' docstring).
    monkeypatch.setenv("MODEL_DECK_NODE_LABEL", "autarch")
    _spark_env(monkeypatch)
    settings = Settings()

    watcher = _build_watcher(settings)

    assert watcher._configurable_engines() == [(LEGACY_SPARK_SEED_ID, "vllm")]
    assert isinstance(watcher._engine_exec, EngineExecRouter)
    assert watcher._engine_exec.pairs == [(LEGACY_SPARK_SEED_ID, "vllm")]
    assert isinstance(watcher._engine_exec._routes[(LEGACY_SPARK_SEED_ID, "vllm")], SparkCatalogExec)
    # Never the label, and never under the label's own value either.
    assert ("autarch", "vllm") not in watcher._engine_exec.pairs


def test_build_watcher_leaves_harvest_disabled_without_spark(tmp_path, monkeypatch):
    """No spark configured -> routes stays {} -> engine_exec is None and
    _harvest_catalogs' `if self._engine_exec is None: return` short-circuits
    -- harvest fully off, same posture as every pre-C2 build."""
    from app.main import _build_watcher
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    settings = Settings()

    watcher = _build_watcher(settings)

    assert watcher._configurable_engines() == []
    assert watcher._engine_exec is None


def test_build_watcher_stashes_engine_exec_on_the_shared_deck(tmp_path, monkeypatch):
    """Task 3 (manual force-harvest route): app.routers.settings.harvest_now
    reads deck["engine_exec"]/deck["configurable_engines"] off
    request.app.state.deck — which _build_deck's settings-id cache (see its
    module docstring) makes the SAME dict _build_watcher builds the real
    Watcher from. Building the watcher must stash both keys onto that
    shared dict, not a private copy, or the HTTP route and the watcher's own
    harvest loop would silently disagree about what's configurable."""
    from app.engines.docker_ctl import EngineExecRouter
    from app.main import _build_deck, _build_watcher
    from app.node_store import LEGACY_SPARK_SEED_ID
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    _spark_env(monkeypatch)
    settings = Settings()

    watcher = _build_watcher(settings)
    deck = _build_deck(settings)  # same cached dict, by settings identity

    assert deck["configurable_engines"] == [(LEGACY_SPARK_SEED_ID, "vllm")]
    assert isinstance(deck["engine_exec"], EngineExecRouter)
    assert deck["engine_exec"] is watcher._engine_exec


def test_build_deck_wires_engine_exec_without_build_watcher(tmp_path, monkeypatch):
    """FINDING 1 fix (task 3 review, 2026-08-08): deck["engine_exec"]/
    deck["configurable_engines"] used to be stashed only inside
    _build_watcher -- which lifespan() SKIPS ENTIRELY under
    MODEL_DECK_NO_WATCHER=1 (main.py's own documented "bare-uvicorn runs
    that don't want the background loop" support). In that mode both keys
    were absent from app.state.deck, and since harvest_now checks the pair
    before the exec, POST /settings/harvest/... always 422'd "not a
    configurable pair" even with a real spark configured -- misdiagnosing
    "watcher never wired" as "pair not configured". _build_deck runs in
    EVERY mode (create_app() always calls it; _build_watcher does not), so
    it must wire both keys itself -- proven here by calling _build_deck
    alone, _build_watcher never invoked at all."""
    from app.engines.docker_ctl import EngineExecRouter
    from app.main import _build_deck
    from app.node_store import LEGACY_SPARK_SEED_ID
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    _spark_env(monkeypatch)
    settings = Settings()

    deck = _build_deck(settings)

    assert deck["configurable_engines"] == [(LEGACY_SPARK_SEED_ID, "vllm")]
    assert isinstance(deck["engine_exec"], EngineExecRouter)


def test_no_watcher_app_state_deck_still_has_engine_exec_when_spark_is_configured(
    tmp_path, monkeypatch
):
    """Symptom-level repro of the FINDING 1 fix above: a fully-assembled
    create_app() under MODEL_DECK_NO_WATCHER=1 (lifespan never calls
    _build_watcher) must still expose engine_exec/configurable_engines on
    app.state.deck when spark is configured -- the manual force-harvest
    route (app.routers.settings.harvest_now) reads off
    request.app.state.deck, not off any watcher instance."""
    from app.engines.docker_ctl import EngineExecRouter
    from app.main import create_app
    from app.node_store import LEGACY_SPARK_SEED_ID

    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")
    _spark_env(monkeypatch)

    app = create_app()

    assert app.state.deck["configurable_engines"] == [(LEGACY_SPARK_SEED_ID, "vllm")]
    assert isinstance(app.state.deck["engine_exec"], EngineExecRouter)


# --- SglangOmniClient ---
#
# Wire shapes come from extensions/services/node-agent/app.py:139-171:
#   GET  /v1/node/engine/{resource}/status -> 200 {"reachable", "healthy",
#                                              "busy_requests"}; 404 unknown
#   POST /v1/node/engine/{resource}/up     -> 202 {"accepted": true}; 404
#                                              unknown, 409 pending, 503
#                                              swap-ctl disabled
#   POST /v1/node/engine/{resource}/down   -> same as up


def _sglang_client(handler, resource="sglang-omni-1"):
    return SglangOmniClient("http://hera:7720", "s3cret", resource,
                            transport=httpx.MockTransport(handler))


def test_sglang_omni_status_returns_wire_dict_verbatim():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"reachable": True, "healthy": True,
                                         "busy_requests": 0}, request=request)

    body = _sglang_client(handler).status()
    assert seen["path"] == "/v1/node/engine/sglang-omni-1/status"
    assert seen["auth"] == "Bearer s3cret"
    assert body == {"reachable": True, "healthy": True, "busy_requests": 0}


def test_sglang_omni_status_raises_engineerror_naming_resource_on_404():
    def handler(request):
        return httpx.Response(404, json={"detail": "unknown engine"}, request=request)

    with pytest.raises(EngineError) as exc_info:
        _sglang_client(handler, resource="mystery-omni").status()
    assert "mystery-omni" in str(exc_info.value)


def test_sglang_omni_status_raises_engineerror_on_transport_failure():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EngineError) as exc_info:
        _sglang_client(handler).status()
    # Never a bare httpx exception, and the concrete NodeAgentUnreachable
    # type (an EngineError subclass) survives the resource-naming re-wrap.
    assert isinstance(exc_info.value, NodeAgentUnreachable)


def test_sglang_omni_up_posts_to_correct_path_and_returns_none_on_202():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["method"] = request.method
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(202, json={"accepted": True}, request=request)

    assert _sglang_client(handler).up() is None
    assert seen["path"] == "/v1/node/engine/sglang-omni-1/up"
    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer s3cret"


def test_sglang_omni_down_posts_to_correct_path_and_returns_none_on_202():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(202, json={"accepted": True}, request=request)

    assert _sglang_client(handler).down() is None
    assert seen["path"] == "/v1/node/engine/sglang-omni-1/down"
    assert seen["method"] == "POST"


@pytest.mark.parametrize("status_code", [404, 409, 503])
def test_sglang_omni_up_raises_engineerror_naming_resource_on_non_202(status_code):
    def handler(request):
        return httpx.Response(status_code, json={"detail": "nope"}, request=request)

    with pytest.raises(EngineError) as exc_info:
        _sglang_client(handler, resource="sglang-omni-2").up()
    assert "sglang-omni-2" in str(exc_info.value)


@pytest.mark.parametrize("status_code", [404, 409, 503])
def test_sglang_omni_down_raises_engineerror_naming_resource_on_non_202(status_code):
    def handler(request):
        return httpx.Response(status_code, json={"detail": "nope"}, request=request)

    with pytest.raises(EngineError) as exc_info:
        _sglang_client(handler, resource="sglang-omni-2").down()
    assert "sglang-omni-2" in str(exc_info.value)


def test_sglang_omni_up_raises_engineerror_on_transport_failure():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EngineError):
        _sglang_client(handler).up()


def test_sglang_omni_down_raises_engineerror_on_transport_failure():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EngineError):
        _sglang_client(handler).down()
