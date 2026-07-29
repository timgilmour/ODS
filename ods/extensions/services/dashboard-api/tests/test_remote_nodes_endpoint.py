"""/api/gpu/detailed carries remote-node statuses (Task 7 wiring)."""

import pytest

import remote_nodes
from models import IndividualGPU, RemoteNodeStatus
from routers import gpu as gpu_router


def _local_gpu(_backend):
    return [IndividualGPU(
        index=0, uuid="GPU-local", name="R9700", memory_used_mb=1,
        memory_total_mb=2, memory_percent=50.0, utilization_percent=3,
        temperature_c=40)]


@pytest.mark.asyncio
async def test_gpu_detailed_includes_remote_nodes(monkeypatch):
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", _local_gpu)
    monkeypatch.setattr(gpu_router, "decode_gpu_assignment", lambda: None)
    monkeypatch.setattr(
        remote_nodes, "get_remote_node_statuses",
        lambda: [RemoteNodeStatus(name="sparky", platform="nvidia",
                                  status="online")])

    try:
        result = await gpu_router.gpu_detailed()
    finally:
        gpu_router._detailed_cache["expires"] = 0.0
        gpu_router._detailed_cache["value"] = None

    assert result.nodes[0].name == "sparky"
    assert result.gpus[0].name == "R9700"


@pytest.mark.asyncio
async def test_gpu_detailed_nodes_empty_when_unconfigured(monkeypatch):
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", _local_gpu)
    monkeypatch.setattr(gpu_router, "decode_gpu_assignment", lambda: None)
    monkeypatch.setattr(remote_nodes, "get_remote_node_statuses", lambda: [])

    try:
        result = await gpu_router.gpu_detailed()
    finally:
        gpu_router._detailed_cache["expires"] = 0.0
        gpu_router._detailed_cache["value"] = None

    assert result.nodes == []
