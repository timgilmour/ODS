"""/api/gpu/detailed carries remote-node statuses (Task 7 wiring)."""

import pytest
from fastapi import HTTPException

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


@pytest.mark.asyncio
async def test_gpu_detailed_serves_nodes_when_host_has_no_local_gpu(monkeypatch):
    """A GPU-less dashboard host with a remote node must still see the node.

    That is the design doc's motivating case, and a transient local
    ``nvidia-smi`` failure must not blank the remote sections either, so the
    endpoint returns 200 with ``gpus: []`` instead of 503 whenever at least
    one remote node status exists.
    """
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", lambda _backend: [])
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

    assert result.gpus == []
    assert result.gpu_count == 0
    assert result.nodes[0].name == "sparky"
    # aggregate_gpu_details() rejects an empty list, so the router must supply
    # a zero-valued aggregate rather than crash.
    assert result.aggregate.gpu_count == 0
    assert result.aggregate.memory_total_mb == 0
    assert result.aggregate.memory_percent == 0.0


@pytest.mark.asyncio
async def test_gpu_detailed_still_503_without_local_gpus_or_nodes(monkeypatch):
    """Unchanged behaviour when there is nothing at all to report."""
    gpu_router._detailed_cache["expires"] = 0.0
    gpu_router._detailed_cache["value"] = None
    monkeypatch.setattr(gpu_router, "_get_raw_gpus", lambda _backend: None)
    monkeypatch.setattr(gpu_router, "decode_gpu_assignment", lambda: None)
    monkeypatch.setattr(remote_nodes, "get_remote_node_statuses", lambda: [])

    try:
        with pytest.raises(HTTPException) as excinfo:
            await gpu_router.gpu_detailed()
    finally:
        gpu_router._detailed_cache["expires"] = 0.0
        gpu_router._detailed_cache["value"] = None

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "No GPU data available"
