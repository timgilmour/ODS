"""Selects the vendored dashboard-api collector for this node's backend."""
from typing import Optional

try:
    from gpu import get_gpu_info_amd_detailed, get_gpu_info_nvidia_detailed
except ImportError:  # pragma: no cover - vendored at container build time
    get_gpu_info_nvidia_detailed = None
    get_gpu_info_amd_detailed = None


def collect_detailed_gpus(backend: str) -> Optional[list]:
    if backend == "amd" and get_gpu_info_amd_detailed:
        return get_gpu_info_amd_detailed()
    if get_gpu_info_nvidia_detailed:
        return get_gpu_info_nvidia_detailed()
    return None
