from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake_gpus():
    return [{
        "index": 0, "uuid": "GPU-abc", "name": "NVIDIA GB10",
        "memory_used_mb": 1024, "memory_total_mb": 122880,
        "memory_percent": 0.8, "utilization_percent": 5,
        "temperature_c": 45, "power_w": 30.0, "memory_type": "unified",
    }]


def test_info_requires_auth():
    assert client.get("/v1/node/info").status_code == 401
    bad = client.get("/v1/node/info", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


def test_info_returns_identity_and_inventory(monkeypatch):
    monkeypatch.setattr(app_module, "_collect_gpus_uncached", _fake_gpus)
    r = client.get("/v1/node/info", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["capabilities"] == ["metrics"]
    assert body["platform"] == "nvidia"
    assert body["gpus"][0]["name"] == "NVIDIA GB10"
    assert body["name"]
    assert body["hostname"]
