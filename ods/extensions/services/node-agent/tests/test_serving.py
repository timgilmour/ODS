from fastapi.testclient import TestClient

import serving
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
