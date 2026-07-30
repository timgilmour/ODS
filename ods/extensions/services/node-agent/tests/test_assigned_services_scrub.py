"""Deck assignments (assigned_services) are dashboard-host-local by design:
deck/placement integration for remote GPUs is deliberately out of scope, so a
remote node must never claim one. The vendored
collector (dashboard-api/gpu.py) fills assigned_services from either the
deck's assignment file or, absent one, an inference from local compute
processes -- on a remote node that inference is meaningless (it labels an
ODS service that runs on the DASHBOARD host, not this node) and must never
reach the dashboard. The node-agent must always report an empty
assigned_services list for every GPU, regardless of what the collector
returns.
"""
from fastapi.testclient import TestClient

import app as app_module
import gpu_collect
from app import app
from models import IndividualGPU

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-key"}


def _collector_with_bogus_assignment():
    return [IndividualGPU(
        index=0, uuid="GPU-abc", name="NVIDIA GB10",
        memory_used_mb=2048, memory_total_mb=122880,
        memory_percent=1.7, utilization_percent=40,
        temperature_c=55, power_w=90.0, memory_type="unified",
        assigned_services=["llama-server"],
    )]


def test_collect_detailed_gpus_scrubs_assigned_services(monkeypatch):
    """Unit level: the collector adapter itself must scrub the field."""
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed",
                        _collector_with_bogus_assignment)
    result = gpu_collect.collect_detailed_gpus("nvidia")
    assert result is not None
    assert result[0].assigned_services == []


def test_collect_detailed_gpus_scrubs_assigned_services_amd(monkeypatch):
    monkeypatch.setattr(gpu_collect, "get_gpu_info_amd_detailed",
                        _collector_with_bogus_assignment)
    result = gpu_collect.collect_detailed_gpus("amd")
    assert result is not None
    assert result[0].assigned_services == []


def test_collect_detailed_gpus_none_passthrough(monkeypatch):
    """No GPUs found (None) must remain None, not raise."""
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed", lambda: None)
    assert gpu_collect.collect_detailed_gpus("nvidia") is None


def test_gpu_route_never_exposes_assigned_services(monkeypatch):
    """Integration: /v1/node/gpu must not leak the local-host inference."""
    app_module._gpu_cache["expires"] = 0.0
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed",
                        _collector_with_bogus_assignment)
    r = client.get("/v1/node/gpu", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["gpus"][0]["assigned_services"] == []


def test_info_route_never_exposes_assigned_services(monkeypatch):
    """Integration: /v1/node/info must not leak the local-host inference."""
    monkeypatch.setattr(gpu_collect, "get_gpu_info_nvidia_detailed",
                        _collector_with_bogus_assignment)
    r = client.get("/v1/node/info", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["gpus"][0]["assigned_services"] == []
