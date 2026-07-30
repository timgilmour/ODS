"""Selects the vendored dashboard-api collector for this node's backend."""
from typing import Optional

try:
    from gpu import get_gpu_info_amd_detailed, get_gpu_info_nvidia_detailed
except ImportError:  # pragma: no cover - vendored at container build time
    get_gpu_info_nvidia_detailed = None
    get_gpu_info_amd_detailed = None


def _scrub_assigned_services(gpus: Optional[list]) -> Optional[list]:
    """Deck assignment is dashboard-host-local by design: remote nodes are
    never deck resources (see multi-node GPU visibility design doc,
    "out of scope"). The vendored collector fills assigned_services from
    the deck's assignment file or, absent one, an inference from local
    compute processes -- on a remote node that inference names a service
    running on the DASHBOARD host, not this node, so it must never reach
    the dashboard as if it were this node's own deck assignment.
    """
    if not gpus:
        return gpus
    for gpu in gpus:
        gpu.assigned_services = []
    return gpus


def collect_detailed_gpus(backend: str) -> Optional[list]:
    if backend == "amd" and get_gpu_info_amd_detailed:
        return _scrub_assigned_services(get_gpu_info_amd_detailed())
    if get_gpu_info_nvidia_detailed:
        return _scrub_assigned_services(get_gpu_info_nvidia_detailed())
    return None
