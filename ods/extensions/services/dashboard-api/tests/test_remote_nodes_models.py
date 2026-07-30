"""RemoteNodeStatus models + MultiGPUStatus back-compat contract."""
from models import GPUInfo, IndividualGPU, MultiGPUStatus, RemoteNodeServing, RemoteNodeStatus

BASELINE_MULTI_GPU_FIELDS = {
    "gpu_count", "backend", "gpus", "topology", "assignment",
    "split_mode", "tensor_split", "aggregate",
}


def _aggregate():
    return GPUInfo(name="agg", memory_used_mb=1, memory_total_mb=2,
                   memory_percent=50.0, utilization_percent=1, temperature_c=40)


def test_multigpustatus_backcompat_additive_only():
    fields = set(MultiGPUStatus.model_fields)
    assert BASELINE_MULTI_GPU_FIELDS <= fields
    assert fields - BASELINE_MULTI_GPU_FIELDS == {"nodes"}


def test_nodes_default_empty_and_serializes():
    status = MultiGPUStatus(gpu_count=0, backend="amd", gpus=[],
                            aggregate=_aggregate())
    assert status.nodes == []
    assert status.model_dump()["nodes"] == []


def test_remote_node_status_shape():
    node = RemoteNodeStatus(
        name="sparky", display_name="DGX Spark GB10", platform="nvidia",
        status="online", last_seen="2026-07-29T18:00:00+00:00",
        gpus=[IndividualGPU(index=0, uuid="GPU-x", name="GB10",
                            memory_used_mb=1, memory_total_mb=2,
                            memory_percent=50.0, utilization_percent=1,
                            temperature_c=40)],
        serving=RemoteNodeServing(model="heretic", endpoint_ok=True),
    )
    assert node.status == "online"
    assert node.error is None
