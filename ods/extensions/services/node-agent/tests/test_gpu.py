from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake(name):
    def _inner():
        return [{
            "index": 0, "uuid": "GPU-abc", "name": name,
            "memory_used_mb": 2048, "memory_total_mb": 122880,
            "memory_percent": 1.7, "utilization_percent": 40,
            "temperature_c": 55, "power_w": 90.0, "memory_type": "unified",
        }]
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


def test_gpu_uses_ttl_cache(monkeypatch):
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("FIRST"))
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake("SECOND"))
    # within TTL: still FIRST
    assert client.get("/v1/node/gpu", headers=AUTH).json()["gpus"][0]["name"] == "FIRST"
