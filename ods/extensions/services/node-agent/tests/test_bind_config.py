"""NODE_AGENT_BIND: let a multi-homed remote node bind one interface.

Under ``network_mode: host`` the agent binds every interface on the node --
including any WAN or management NIC it happens to have -- and the only thing
standing between the metrics API and those interfaces is the node's own
firewall. A node with more than one NIC should be able to say "listen on the
LAN address only" and stop relying on that single control.

The bind address, like the port, is owned by the Dockerfile CMD: uvicorn reads
it before any of our Python runs, so a copy in ``nodeconfig`` would be read by
nothing (see test_nodeconfig.py::test_dead_port_var_is_not_reintroduced). That
leaves three files that must agree, so these guard the drift instead.
"""
import re
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    path = SERVICE / name
    if not path.exists():
        pytest.skip(f"{name} not present (deployed container)")
    return path.read_text(encoding="utf-8")


def test_dockerfile_cmd_honours_bind_and_still_defaults_to_all_interfaces():
    """Default must stay 0.0.0.0: existing deploys set no bind var and their
    dashboard host reaches them over the LAN address."""
    cmd = _read("Dockerfile")
    assert "${NODE_AGENT_BIND:-0.0.0.0}" in cmd
    assert "--host 0.0.0.0" not in cmd


def test_compose_passes_bind_through():
    compose = _read("compose.yaml.disabled")
    assert re.search(r"^\s+NODE_AGENT_BIND: \$\{NODE_AGENT_BIND:-\}\s*$",
                     compose, re.M)


def test_healthcheck_probes_the_bound_address_not_a_hardcoded_loopback():
    """The liveness probe connects from inside the container. Binding to a
    single LAN address makes a hardcoded 127.0.0.1 probe fail against a
    perfectly healthy agent, so the probe has to follow the bind."""
    compose = _read("compose.yaml.disabled")
    healthcheck = compose[compose.index("healthcheck:"):]
    healthcheck = healthcheck[:healthcheck.index("retries:")]
    assert "NODE_AGENT_BIND" in healthcheck


def test_manifest_declares_bind_as_optional():
    manifest = _read("manifest.yaml")
    assert "- key: NODE_AGENT_BIND" in manifest
    entry = manifest[manifest.index("- key: NODE_AGENT_BIND"):]
    entry = entry.split("- key:")[1]
    assert "required: false" in entry


def test_no_python_copy_of_the_bind_address():
    import nodeconfig
    assert not hasattr(nodeconfig, "NODE_AGENT_BIND")
