"""Guards the host_agent_client shim's import surface against drift.

The vendored collector (dashboard-api/gpu.py, copied in at image build time)
imports names from host_agent_client for its windows-host code paths. The
node-agent ships a local stub instead of the real client, so if gpu.py ever
imports a new name the container fails at import time with an ImportError --
the same drift class tests/test_model_parity.py already guards for models.py.
Runs in the repo checkout; self-skips in the deployed container.
"""
import ast
from pathlib import Path

import pytest

DASHBOARD_API_GPU = (
    Path(__file__).resolve().parents[2] / "dashboard-api" / "gpu.py"
)


def _names_imported_from_host_agent_client(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "host_agent_client":
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "host_agent_client":
                    # A plain module import means any attribute could be used;
                    # attribute-level parity cannot be proven from the AST.
                    pytest.fail("gpu.py imports host_agent_client as a module; "
                                "the shim cannot guarantee attribute parity")
    return names


def test_shim_provides_everything_the_vendored_collector_imports():
    if not DASHBOARD_API_GPU.exists():
        pytest.skip("dashboard-api not present (deployed container)")
    import host_agent_client as shim

    required = _names_imported_from_host_agent_client(
        DASHBOARD_API_GPU.read_text(encoding="utf-8"))
    assert required, "expected gpu.py to import from host_agent_client"
    missing = sorted(name for name in required if not hasattr(shim, name))
    assert missing == [], (
        f"host_agent_client shim is missing {missing}; the vendored gpu.py "
        "imports it and the container would fail at import time"
    )


def test_shim_request_json_fails_closed():
    """No host-agent exists on a remote node, so the stub must raise its own
    error type rather than silently returning None into the collector."""
    import host_agent_client as shim

    with pytest.raises(shim.AgentClientError):
        shim.request_json("GET", "/anything")
