import json

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
