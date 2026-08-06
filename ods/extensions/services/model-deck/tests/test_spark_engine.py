"""Tests for app.engines.spark — remote single-slot swap client for sparky.

Same transport-injection style as test_engines.py: the client takes
`node_transport=` / `serving_transport=` kwargs, and the litellm guard is
exercised through a real LiteLLMClient wired to its own MockTransport.
"""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.engines import BusyError, EngineError, GuardError
from app.engines.litellm import LiteLLMClient
from app.engines.spark import SparkClient, boot_in_flight

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


DEFAULT_PROFILES = (
    {"name": "laguna", "engine": "vllm", "health_url": None, "container": None},
    {"name": "mm27b", "engine": "vllm", "health_url": None, "container": None},
)

# A node whose last swap landed on a non-vllm (ComfyUI) profile — used to
# exercise the engine-aware busy guard.
COMFY_MIX_PROFILES = (
    {"name": "laguna", "engine": "vllm", "health_url": None, "container": None},
    {"name": "comfyui", "engine": "comfyui",
     "health_url": "http://sparky:8188/system_stats", "container": "comfyui"},
)

# A node whose last swap landed on ds4 — live profiles.json shape
# (2026-08-04): ds4 serves Prometheus metrics at :8000/metrics, same port
# vLLM profiles use on sparky's host networking.
DS4_MIX_PROFILES = (
    {"name": "laguna", "engine": "vllm", "health_url": None, "container": None},
    {"name": "ds4", "engine": "ds4",
     "health_url": "http://127.0.0.1:8000/metrics", "container": "spark-ds4"},
)

# A profile whose engine has no entry in _BUSY_METRICS at all (not comfyui,
# not vllm, not ds4) — exercises the mapping's "no entry -> skip" default
# for any future engine that hasn't been wired up yet.
UNMAPPED_ENGINE_PROFILES = (
    {"name": "laguna", "engine": "vllm", "health_url": None, "container": None},
    {"name": "mystery", "engine": "sglang", "health_url": None, "container": None},
)


def _node_handler(profiles=DEFAULT_PROFILES, swap_status=None,
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


def _no_matching_gauge_metrics_handler():
    # A 200 scrape that doesn't contain ANY of the expected gauge prefixes
    # for either engine -- simulates a misconfigured endpoint or an
    # upstream metric rename, not a transport failure.
    text = "# HELP some_other_metric unrelated\nsome_other_metric 1.0\n"

    def handler(request):
        return httpx.Response(200, text=text, request=request)

    return handler


def _ds4_metrics_handler(inflight=0):
    # ds4_requests_started_total is a counter, not the guard signal — its
    # presence in the payload checks that the guard doesn't accidentally
    # count it (startswith("ds4_requests_inflight") must not match it).
    text = (f'ds4_requests_inflight {inflight}.0\n'
            f'ds4_requests_started_total 42.0\n')

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
    assert s["profiles"] == list(DEFAULT_PROFILES)
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


# --- models (characteristics derive pass) ---

def _models_handler(body):
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=body, request=request)
    return handler


def test_models_returns_the_serving_v1_models_body():
    body = {"data": [{"id": "heretic", "max_model_len": 131072}]}
    client = _client(_node_handler(), metrics_handler=_models_handler(body))
    assert client.models() == body


def test_models_raises_engineerror_on_serving_transport_failure():
    def handler(request):
        raise httpx.ConnectError("serving down")
    client = _client(_node_handler(), metrics_handler=handler)
    with pytest.raises(EngineError):
        client.models()


def test_models_raises_engineerror_on_non_2xx():
    def handler(request):
        return httpx.Response(500, text="boom", request=request)
    client = _client(_node_handler(), metrics_handler=handler)
    with pytest.raises(EngineError):
        client.models()


def test_models_raises_engineerror_on_garbage_200_body():
    """A 200 with a non-JSON body (misconfigured proxy, HTML error page,
    truncated response, ...) must not escape as a raw JSONDecodeError — the
    derive pass's ``except (EngineError, BusyError)`` catch wouldn't see it."""
    def handler(request):
        return httpx.Response(200, text="not json", request=request)
    client = _client(_node_handler(), metrics_handler=handler)
    with pytest.raises(EngineError):
        client.models()


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


# --- boot-window guard (found live 2026-07-30: helper "done" != endpoint up;
# the whole autotune window was swappable with no warning) ---

def test_swap_refuses_while_previous_swap_still_booting():
    handler = _node_handler(
        swap_status={"state": "done", "profile": "laguna", "id": "u0",
                     "message": "swap launched", "ts": "2026-07-30T22:35:02Z"},
        serving={"model": None, "endpoint_ok": False, "container_status": None})
    client = _client(handler)
    with pytest.raises(GuardError) as exc:
        client.swap("mm27b")
    assert "boot" in str(exc.value)
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_force_overrides_boot_window_guard():
    handler = _node_handler(
        swap_status={"state": "done", "profile": "laguna", "id": "u0",
                     "message": "swap launched", "ts": "2026-07-30T22:35:02Z"},
        serving={"model": None, "endpoint_ok": False, "container_status": None})
    client = _client(handler)
    assert client.swap("mm27b", force=True)["id"] == "u1"


def test_swap_allowed_when_endpoint_down_with_failed_last_swap():
    # A swap whose swap.sh failed never started a boot — swapping away is
    # the fix, not an interruption; no guard, no force needed.
    handler = _node_handler(
        swap_status={"state": "error", "profile": "qwen35", "id": "u0",
                     "message": "swap.sh failed (see swap.log)",
                     "ts": "2026-07-30T22:35:02Z"},
        serving={"model": None, "endpoint_ok": False, "container_status": None})
    client = _client(handler)
    assert client.swap("mm27b")["id"] == "u1"


# --- engine-aware busy guard (comfyui profiles have no /metrics to probe;
# v1 has no ComfyUI queue visibility, so the operator owns not swapping
# mid-render) ---

def test_swap_away_from_comfyui_skips_vllm_busy_probe():
    handler = _node_handler(
        profiles=COMFY_MIX_PROFILES,
        swap_status={"state": "done", "profile": "comfyui", "id": "u0",
                     "message": "swap launched", "ts": "2026-07-31T00:00:00Z"},
        swap_response=(202, {"id": "i1", "profile": "heretic"}))

    def metrics_404(request):
        # no /metrics route registered for a comfyui-served node
        return httpx.Response(404, text="not found", request=request)

    client = _client(handler, metrics_handler=metrics_404)
    out = client.swap("heretic")
    assert out["id"] == "i1"  # swap succeeded, no EngineError from /metrics


def test_swap_between_vllm_profiles_still_probes_metrics():
    handler = _node_handler(
        profiles=COMFY_MIX_PROFILES,
        swap_status={"state": "done", "profile": "laguna", "id": "u0",
                     "message": "swap launched", "ts": "2026-07-31T00:00:00Z"})
    client = _client(handler, metrics_handler=_metrics_handler(running=1))
    with pytest.raises(GuardError):
        client.swap("mm27b")


def test_swap_with_no_swap_status_probes_metrics():
    # swap_status None -> conservative: engine unknown, vLLM busy guard runs
    # (existing behavior).
    handler = _node_handler(profiles=COMFY_MIX_PROFILES, swap_status=None)
    client = _client(handler, metrics_handler=_metrics_handler(running=1))
    with pytest.raises(GuardError):
        client.swap("mm27b")


def test_swap_skips_busy_probe_for_engine_with_no_busy_metrics_mapping():
    # A named engine that simply isn't in _BUSY_METRICS yet (not comfyui)
    # takes the same skip path as comfyui: mapping miss -> no probe, not an
    # EngineError from a 404'd /metrics call.
    handler = _node_handler(
        profiles=UNMAPPED_ENGINE_PROFILES,
        swap_status={"state": "done", "profile": "mystery", "id": "u0",
                     "message": "swap launched", "ts": "2026-08-04T00:00:00Z"},
        swap_response=(202, {"id": "i3", "profile": "laguna"}))

    def metrics_404(request):
        return httpx.Response(404, text="not found", request=request)

    client = _client(handler, metrics_handler=metrics_404)
    out = client.swap("laguna")
    assert out["id"] == "i3"


# --- ds4 busy guard (ds4_requests_inflight gauge at :8000/metrics, same
# port vLLM serves on sparky's host networking) ---

def test_swap_away_from_ds4_refused_when_ds4_busy():
    handler = _node_handler(
        profiles=DS4_MIX_PROFILES,
        swap_status={"state": "done", "profile": "ds4", "id": "u0",
                     "message": "swap launched", "ts": "2026-08-04T00:00:00Z"})
    client = _client(handler, metrics_handler=_ds4_metrics_handler(inflight=1))
    with pytest.raises(GuardError):
        client.swap("laguna")
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_away_from_ds4_allowed_when_ds4_idle():
    handler = _node_handler(
        profiles=DS4_MIX_PROFILES,
        swap_status={"state": "done", "profile": "ds4", "id": "u0",
                     "message": "swap launched", "ts": "2026-08-04T00:00:00Z"},
        swap_response=(202, {"id": "i2", "profile": "laguna"}))
    client = _client(handler, metrics_handler=_ds4_metrics_handler(inflight=0))
    out = client.swap("laguna")
    assert out["id"] == "i2"


def test_swap_away_from_ds4_refused_when_metrics_matches_no_known_gauge():
    # A 200 scrape that matches none of ds4's expected prefixes must NOT
    # read as "0 busy" -- that would silently let a swap kill an in-flight
    # generation if the gauge were ever renamed or the scrape misconfigured.
    # ds4 fails closed here (Tim's 2026-08-04 ruling); see
    # _STRICT_BUSY_ENGINES.
    handler = _node_handler(
        profiles=DS4_MIX_PROFILES,
        swap_status={"state": "done", "profile": "ds4", "id": "u0",
                     "message": "swap launched", "ts": "2026-08-04T00:00:00Z"})
    client = _client(handler, metrics_handler=_no_matching_gauge_metrics_handler())
    with pytest.raises(EngineError):
        client.swap("laguna")
    assert not [r for r in handler.calls if r.url.path == "/v1/node/swap"]


def test_swap_between_vllm_profiles_allows_when_metrics_matches_no_known_gauge():
    # Pins the "vLLM untouched" requirement: today's fail-open-on-absent-
    # gauges behavior is unchanged for vllm -- only ds4 fails closed.
    handler = _node_handler()  # DEFAULT_PROFILES, swap_status=None -> engine
                                # unknown -> "vllm" fallback, same as today.
    client = _client(handler, metrics_handler=_no_matching_gauge_metrics_handler())
    assert client.swap("laguna")["id"] == "u1"


# --- boot-in-flight (lifecycle reconciler's boot window) ---

def _swap_status(state="done", ts=None, profile="laguna"):
    return {"state": state, "profile": profile, "id": "u0",
            "message": "swap launched",
            "ts": ts or datetime.now(UTC).isoformat().replace("+00:00", "Z")}


def test_boot_in_flight_true_while_endpoint_down_after_a_recent_swap():
    assert boot_in_flight({
        "swap_status": _swap_status("done"),
        "serving": {"model": None, "endpoint_ok": False},
    }) is True


def test_boot_in_flight_false_once_the_endpoint_is_up():
    """swap_status stays 'done' forever — the live endpoint is what ends the
    boot window."""
    assert boot_in_flight({
        "swap_status": _swap_status("done"),
        "serving": {"model": "aeon", "endpoint_ok": True},
    }) is False


def test_boot_in_flight_false_for_a_failed_swap():
    """A swap whose helper errored never started a boot."""
    assert boot_in_flight({
        "swap_status": _swap_status("error"),
        "serving": {"model": None, "endpoint_ok": False},
    }) is False


def test_boot_in_flight_expires_so_a_dead_model_is_not_shielded_forever():
    """THE regression this bound exists for: hours after a successful swap,
    swap_status still reads 'done'. Without ageing it out, a model that died
    later would look like it was still booting and never be restored."""
    old = (datetime.now(UTC) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    assert boot_in_flight({
        "swap_status": _swap_status("done", ts=old),
        "serving": {"model": None, "endpoint_ok": False},
    }) is False


def test_boot_in_flight_without_a_timestamp_stays_conservative():
    """Unsure means do not act — guessing wrong costs a multi-minute swap."""
    assert boot_in_flight({
        "swap_status": {"state": "done", "profile": "laguna"},
        "serving": {"model": None, "endpoint_ok": False},
    }) is True

    assert boot_in_flight({
        "swap_status": {"state": "done", "ts": "not-a-timestamp"},
        "serving": {"model": None, "endpoint_ok": False},
    }) is True


def test_boot_in_flight_false_with_no_swap_status_at_all():
    assert boot_in_flight({"swap_status": None,
                           "serving": {"model": None, "endpoint_ok": False}}) is False


def test_swap_in_progress_reads_the_live_node():
    handler = _node_handler(
        swap_status=_swap_status("done"),
        serving={"model": None, "endpoint_ok": False, "container_status": None})
    assert _client(handler).swap_in_progress() is True


def test_swap_in_progress_false_while_serving():
    assert _client(_node_handler(swap_status=_swap_status("done"))).swap_in_progress() is False
