from fastapi.testclient import TestClient

import app as app_module
import gpu_collect
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake(name):
    def _inner():
        return ([{
            "index": 0, "uuid": "GPU-abc", "name": name,
            "memory_used_mb": 2048, "memory_total_mb": 122880,
            "memory_percent": 1.7, "utilization_percent": 40,
            "temperature_c": 55, "power_w": 90.0, "memory_type": "unified",
        }], None)
    return _inner


def test_gpu_requires_auth():
    assert client.get("/v1/node/gpu").status_code == 401


def test_gpu_returns_metrics(monkeypatch):
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("NVIDIA GB10"))
    r = client.get("/v1/node/gpu", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "nvidia"
    assert body["gpus"][0]["utilization_percent"] == 40
    assert body["error"] is None


def test_gpu_uses_ttl_cache(monkeypatch):
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("FIRST"))
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("SECOND"))
    # within TTL: still FIRST
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"


def test_gpu_reports_collector_failure_as_error(monkeypatch):
    """Design doc's "node up, collector failing" state.

    The collector returning ``None`` means the collector itself is absent or
    failed. That is NOT the same as a node with genuinely zero GPUs, and
    flattening both to ``[]`` left the dashboard with an empty card body and no
    explanation. ``/v1/node/gpu`` surfaces the distinction in a nullable
    ``error`` field (additive, so older dashboards ignore it).
    """
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed", lambda: None)
    r = client.get("/v1/node/gpu", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["gpus"] == []
    assert body["error"]
    assert "collector" in body["error"].lower()


def test_gpu_zero_gpus_is_not_an_error(monkeypatch):
    """An empty list is a truthful "this node has no GPUs" answer."""
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed", lambda: [])
    r = client.get("/v1/node/gpu", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["gpus"] == []
    assert body["error"] is None
