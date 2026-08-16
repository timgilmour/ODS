import json

import httpx
import pytest
from fastapi.testclient import TestClient

import serving
import swapctl
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def test_serving_requires_auth():
    r = client.get("/v1/node/serving")
    assert r.status_code == 401
    assert r.content == b""


def test_probe_happy_path(monkeypatch):
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}


def test_probe_endpoint_down(monkeypatch):
    def _boom(url):
        raise serving.ProbeError("connect timeout")
    monkeypatch.setattr(serving, "_fetch_models_payload", _boom)
    monkeypatch.setattr(serving, "_container_status", lambda name: None)
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    r = client.get("/v1/node/serving", headers=AUTH)
    body = r.json()
    assert body["endpoint_ok"] is False
    assert body["model"] is None


def test_probe_unconfigured(monkeypatch):
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": None, "endpoint_ok": False,
                        "container_status": None}


def test_container_status_is_null_when_docker_is_absent(monkeypatch):
    """The docker socket/CLI mounts are opt-in and commented out by default
    (they grant host-root-equivalent access -- see README "Security"), so with
    NODE_SERVING_CONTAINER set but no docker present the endpoint must still
    answer 200 with container_status: null."""
    def _no_docker(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(serving.subprocess, "run", _no_docker)
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"model": None, "endpoint_ok": False,
                        "container_status": None}


def test_probe_malformed_bare_list(monkeypatch):
    """Endpoint returns bare list instead of {data: [...]}."""
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: [1, 2])
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint_ok"] is False
    assert body["model"] is None


def test_probe_malformed_data_not_dict(monkeypatch):
    """Endpoint returns {data: [...]} but items aren't dicts."""
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": ["not-a-dict"]})
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint_ok"] is False
    assert body["model"] is None


def test_probe_comfyui_engine_uses_health_url(monkeypatch):
    monkeypatch.setattr(
        serving.swapctl, "current_profile_meta",
        lambda: {"name": "comfyui", "engine": "comfyui",
                 "health_url": "http://127.0.0.1:8188/system_stats",
                 "container": "spark-comfyui"})
    monkeypatch.setattr(serving, "_fetch_raw", lambda url: None)
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    result = serving.probe()
    assert result == {"model": "comfyui", "endpoint_ok": True,
                      "container_status": "running"}


def test_probe_comfyui_engine_down(monkeypatch):
    monkeypatch.setattr(
        serving.swapctl, "current_profile_meta",
        lambda: {"name": "comfyui", "engine": "comfyui",
                 "health_url": "http://127.0.0.1:8188/system_stats",
                 "container": "spark-comfyui"})

    def _boom(url):
        raise serving.ProbeError("connection refused")

    monkeypatch.setattr(serving, "_fetch_raw", _boom)
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    result = serving.probe()
    assert result["model"] == "comfyui"
    assert result["endpoint_ok"] is False


def test_probe_comfyui_engine_no_health_url_no_container(monkeypatch):
    """A profile with no health_url/container override just reports the
    profile name; endpoint_ok stays False rather than probing anything."""
    monkeypatch.setattr(
        serving.swapctl, "current_profile_meta",
        lambda: {"name": "comfyui", "engine": "comfyui",
                 "health_url": None, "container": None})
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "")
    result = serving.probe()
    assert result == {"model": "comfyui", "endpoint_ok": False,
                      "container_status": None}


def test_probe_vllm_profile_unchanged(monkeypatch):
    monkeypatch.setattr(
        serving.swapctl, "current_profile_meta",
        lambda: {"name": "heretic", "engine": "vllm",
                 "health_url": None, "container": None})
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}


def test_probe_no_swap_status_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(serving.swapctl, "current_profile_meta", lambda: None)
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}


def test_probe_survives_non_dict_profiles_json_entry(monkeypatch, tmp_path):
    """End-to-end regression for the reviewer-flagged 500: a real (unmocked)
    swapctl reading a valid-JSON-but-non-dict profiles.json entry (e.g.
    {"comfyui": 5}) must not blow up GET /v1/node/serving, a route polled
    continuously. The fix (swapctl.profile_meta() treating a non-dict entry
    as {}) means ALL fields fall back to defaults, including engine ->
    "vllm" -- so this profile is indistinguishable from an unconfigured one
    and probe() correctly takes the ordinary env-configured path, not the
    comfyui-shaped branch. The point of this test is the 200, not the shape."""
    vllm = tmp_path / "vllm"
    ctl = tmp_path / "ctl"
    vllm.mkdir()
    ctl.mkdir()
    (vllm / "compose-comfyui.yaml").write_text("services: {}\n")
    (vllm / "profiles.json").write_text(json.dumps({"comfyui": 5}))
    (ctl / "status.json").write_text(json.dumps(
        {"state": "done", "profile": "comfyui", "id": "x",
         "message": "swap launched", "ts": "2026-07-31T00:00:00Z"}))
    monkeypatch.setattr(serving.swapctl.nodeconfig, "NODE_VLLM_DIR", str(vllm))
    monkeypatch.setattr(serving.swapctl.nodeconfig, "NODE_SWAP_CTL_DIR", str(ctl))
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")

    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}


# ---------------------------------------------------------------------------
# vllm-without-probe-URL warning (2026-08-13 incident follow-up, Part B).
# An empty NODE_SERVING_PROBE_URL on a node with vLLM profiles makes serving
# detection permanently blind (the vllm/env probe path returns all-null), and
# the misconfiguration was invisible for 4 days (08-12 incident). The probe
# result carries a warning whenever the node CONFIG is in that state — keyed
# on the profile list, not the active profile, because a blind future swap
# is the hazard.
# ---------------------------------------------------------------------------


_VLLM_PROFILE = {"name": "laguna", "engine": "vllm",
                 "health_url": None, "container": None}


def test_probe_warns_when_vllm_profile_and_no_probe_url(monkeypatch):
    monkeypatch.setattr(serving.swapctl, "current_profile_meta", lambda: None)
    monkeypatch.setattr(serving.swapctl, "list_profiles",
                        lambda: [_VLLM_PROFILE])
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json()["warning"] == (
        "vllm profiles configured but NODE_SERVING_PROBE_URL is unset — "
        "serving detection is blind")


def test_probe_warning_rides_the_non_vllm_branch(monkeypatch):
    """The field describes node CONFIG, not the current profile: with ds4
    active, a configured vllm profile still means the next blind swap lands
    unobservable — the warning must ride this branch too."""
    monkeypatch.setattr(
        serving.swapctl, "current_profile_meta",
        lambda: {"name": "ds4", "engine": "ds4",
                 "health_url": "http://127.0.0.1:8000/metrics",
                 "container": "spark-ds4"})
    monkeypatch.setattr(serving.swapctl, "list_profiles",
                        lambda: [_VLLM_PROFILE])
    monkeypatch.setattr(serving, "_fetch_raw", lambda url: None)
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    body = r.json()
    assert body["model"] == "ds4"
    assert body["warning"] == serving.PROBE_URL_WARNING


def test_probe_no_warning_when_probe_url_set(monkeypatch):
    monkeypatch.setattr(serving.swapctl, "current_profile_meta", lambda: None)
    monkeypatch.setattr(serving.swapctl, "list_profiles",
                        lambda: [_VLLM_PROFILE])
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "laguna"}]})
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert "warning" not in r.json()


def test_probe_no_warning_without_vllm_profiles(monkeypatch):
    monkeypatch.setattr(serving.swapctl, "current_profile_meta", lambda: None)
    monkeypatch.setattr(
        serving.swapctl, "list_profiles",
        lambda: [{"name": "ds4", "engine": "ds4",
                  "health_url": "http://127.0.0.1:8000/metrics",
                  "container": "spark-ds4"}])
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert "warning" not in r.json()


def test_probe_no_warning_when_swapctl_disabled(monkeypatch):
    """Generic node (no NODE_VLLM_DIR/NODE_SWAP_CTL_DIR): no profiles exist,
    so there is nothing to know — and list_profiles raising SwapCtlDisabled
    must not break the probe."""
    def _disabled():
        raise swapctl.SwapCtlDisabled()

    monkeypatch.setattr(serving.swapctl, "current_profile_meta", _disabled)
    monkeypatch.setattr(serving.swapctl, "list_profiles", _disabled)
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    assert "warning" not in r.json()


def test_probe_survives_an_unreadable_profiles_dir(monkeypatch, caplog):
    """A warning-only feature must never take down what it decorates: a
    profiles dir that raises on inspection (e.g. root-owned, not
    traversable — swapctl's path.is_file() runs outside its try block)
    means the warning is UNDETERMINABLE, not that the probe or agent boot
    should crash. Pre-warning, this config booted and served probes."""
    def _perm(*_a, **_k):
        raise PermissionError("profiles dir not traversable")

    monkeypatch.setattr(serving.swapctl, "current_profile_meta", lambda: None)
    monkeypatch.setattr(serving.swapctl, "list_profiles", _perm)
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.status_code == 200
    assert "warning" not in r.json()

    with caplog.at_level("WARNING", logger="serving"):
        serving.log_startup_warning()   # must not raise either


def test_startup_log_emits_the_warning(monkeypatch, caplog):
    monkeypatch.setattr(serving.swapctl, "list_profiles",
                        lambda: [_VLLM_PROFILE])
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL", "")
    with caplog.at_level("WARNING", logger="serving"):
        serving.log_startup_warning()
    assert serving.PROBE_URL_WARNING in caplog.text


def test_startup_log_silent_when_configured(monkeypatch, caplog):
    monkeypatch.setattr(serving.swapctl, "list_profiles",
                        lambda: [_VLLM_PROFILE])
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    with caplog.at_level("WARNING", logger="serving"):
        serving.log_startup_warning()
    assert caplog.text == ""


def test_fetch_raw_retries_once_on_a_single_dropped_check(monkeypatch):
    """C2 defense 2 (2026-08-06 review, model-deck Task 2): a single
    dropped health check must not read as a dead model downstream (Model
    Deck's observe_spark now consults endpoint_ok, and this is the ONE
    un-retried network call that fed that reading). One retry absorbs a
    transient blip; this is a DIFFERENT deployment target from the
    model-deck fixes (this file runs on the remote node as the
    ods-node-agent container, e.g. sparky — not wherever the Deck backend
    runs)."""
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("blip")
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(serving.httpx, "get", fake_get)

    resp = serving._fetch_raw("http://x/health")

    assert resp.status_code == 200
    assert len(calls) == 2


def test_fetch_raw_raises_after_two_consecutive_failures(monkeypatch):
    """The retry absorbs ONE blip, not a genuine outage — a second
    consecutive failure must still surface as ProbeError, and no further
    retries beyond that: this already runs on the Deck's tick/TTL probing
    cadence, so a genuinely dead endpoint is caught on the next scheduled
    probe regardless; retrying more here would just delay every probe of a
    truly-down endpoint."""
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        raise httpx.ConnectError("still down")

    monkeypatch.setattr(serving.httpx, "get", fake_get)

    with pytest.raises(serving.ProbeError):
        serving._fetch_raw("http://x/health")

    assert len(calls) == 2


def test_fetch_raw_uses_a_one_second_per_attempt_timeout(monkeypatch):
    """PINS the per-attempt timeout (2026-08-06 re-review): two attempts at
    2.0 s each left only ~0.9 s of headroom under the Deck's node-client
    httpx.Timeout(5.0) (app/engines/spark.py) once _container_status's own
    budget is counted — and when that budget blows, the Deck reads
    EngineError -> _UNREACHABLE_SPARK -> 'unreachable', which
    plan_reconcile treats as inert, so a WEDGED (hung, not dead) model
    would never be restored: the same failure this whole plan exists to
    kill, in a different costume. Tim's ruling: keep the retry, shrink the
    attempt to 1.0 s, so two attempts cost what one used to and the
    headroom returns to where it was. This test exists so a future edit
    back toward 2.0 fails loudly here instead of silently eating the
    budget on the live ds4 path."""
    timeouts = []

    def fake_get(url, timeout):
        timeouts.append(timeout)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(serving.httpx, "get", fake_get)

    with pytest.raises(serving.ProbeError):
        serving._fetch_raw("http://x/health")

    assert timeouts == [1.0, 1.0]


def test_probe_swapctl_disabled_falls_back_to_env(monkeypatch):
    """Generic nodes without NODE_VLLM_DIR/NODE_SWAP_CTL_DIR configured raise
    SwapCtlDisabled from current_profile_meta(); probe() must swallow that and
    take the env-configured path (the remote-node e2e harness's contract)."""
    def _disabled():
        raise swapctl.SwapCtlDisabled()

    monkeypatch.setattr(serving.swapctl, "current_profile_meta", _disabled)
    monkeypatch.setattr(serving, "_fetch_models_payload",
                        lambda url: {"data": [{"id": "heretic"}]})
    monkeypatch.setattr(serving, "_container_status", lambda name: "running")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_PROBE_URL",
                        "http://localhost:8000/v1/models")
    monkeypatch.setattr(serving.nodeconfig, "NODE_SERVING_CONTAINER", "aeon-vllm")
    r = client.get("/v1/node/serving", headers=AUTH)
    assert r.json() == {"model": "heretic", "endpoint_ok": True,
                        "container_status": "running"}
