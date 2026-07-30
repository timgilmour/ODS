"""Shim satisfying gpu.py imports. Field parity with dashboard-api/models.py
is enforced by tests/test_model_parity.py."""
from typing import Optional
from pydantic import BaseModel


class GPUInfo(BaseModel):
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: int
    temperature_c: int
    power_w: Optional[float] = None
    memory_type: str = "discrete"
    gpu_backend: str = "nvidia"
    gpu_count: int = 1
    memory_usage_available: bool = True
    utilization_available: bool = True
    temperature_available: bool = True


class IndividualGPU(BaseModel):
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    utilization_percent: int
    temperature_c: int
    power_w: Optional[float] = None
    memory_type: str = "discrete"
    assigned_services: list[str] = []
    memory_usage_available: bool = True
    utilization_available: bool = True
    temperature_available: bool = True
