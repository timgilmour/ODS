"""Guards shim model parity with dashboard-api. Runs in the repo checkout
(where both services exist); skipped inside the deployed container."""
import importlib.util
import sys
from pathlib import Path

import pytest

DASHBOARD_API = Path(__file__).resolve().parents[2] / "dashboard-api"


def _load_dashboard_models():
    if not (DASHBOARD_API / "models.py").exists():
        pytest.skip("dashboard-api not present (deployed container)")
    sys.path.insert(0, str(DASHBOARD_API))
    try:
        spec = importlib.util.spec_from_file_location(
            "da_models", DASHBOARD_API / "models.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def test_individual_gpu_fields_match():
    import models as shim
    da = _load_dashboard_models()
    assert set(shim.IndividualGPU.model_fields) == set(
        da.IndividualGPU.model_fields)


def test_gpu_info_fields_match():
    import models as shim
    da = _load_dashboard_models()
    assert set(shim.GPUInfo.model_fields) == set(da.GPUInfo.model_fields)
