#!/usr/bin/env python3
"""Generate docs/NETWORK-PORTS.md from the repo's own port and policy data.

Three files already describe what ODS listens on, and no document joins them:

- ``extensions/services/*/manifest.yaml`` -- which port a service publishes on
  the host (``external_port_default``), which port it listens on inside its
  container (``port``), and whether it bypasses port mapping entirely
  (``host_network``).
- ``config/ports.json`` -- the env var that overrides each published port.
- ``config/network-exposure-policy.json`` -- risk, whether the surface
  authenticates, and how exposed it is meant to be.

An operator asking "what do I open in the firewall, on which box" has to read
all three. This emits the table instead, and
tests/contracts/test-network-ports-doc.py fails if the committed doc drifts
from the sources, so the table cannot rot the way a hand-written one does.

    generate-network-ports-doc.py            # print to stdout
    generate-network-ports-doc.py --write    # rewrite docs/NETWORK-PORTS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - CI installs PyYAML
    print(f"[FAIL] Missing Python dependency: {exc}")
    raise SystemExit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
SERVICES_DIR = ROOT_DIR / "extensions" / "services"
PORTS_FILE = ROOT_DIR / "config" / "ports.json"
POLICY_FILE = ROOT_DIR / "config" / "network-exposure-policy.json"
DOC_FILE = ROOT_DIR / "docs" / "NETWORK-PORTS.md"

HEADER = """<!-- GENERATED FILE -- DO NOT EDIT BY HAND.
     Regenerate with: python3 scripts/generate-network-ports-doc.py --write
     Sources: extensions/services/*/manifest.yaml, config/ports.json,
              config/network-exposure-policy.json -->

# Network ports

Every port a stock ODS install can listen on, with what guards it. Generated
from the manifests and `config/`, so it cannot drift from what the stack
actually deploys -- `tests/contracts/test-network-ports-doc.py` fails the build
if it does.

**Nothing here is a firewall policy.** It is the inventory you write one from.
The default posture is that these ports are reachable from the LAN if the host
firewall lets them be; services marked `operator-controlled` are expected to
stay local unless you deliberately publish them.

Reading the columns:

- **Host port** -- what is reachable on the machine. `internal only` means the
  container listens but nothing is published to the host; `host network` means
  the service bypasses port mapping and binds the host's interfaces directly,
  so a published-port table would not show it at all.
- **Env var** -- override for the host port, set in `.env`.
- **Auth** -- whether the surface authenticates at all. `no` does not mean
  broken; it means the port itself is the control, so scope it.
"""

FOOTER = """
## Services that publish nothing

Every bundled service not listed above has no host-facing port: it is reachable
only from inside the compose network, by the services that need it.

## Adding a service

Set `external_port_default` (or `host_network: true`) in the manifest and add a
matching entry to `config/network-exposure-policy.json` -- the network-exposure
contract test already requires the pairing. Then regenerate this file:

```bash
python3 scripts/generate-network-ports-doc.py --write
```
"""


def load_manifests() -> dict[str, dict]:
    services: dict[str, dict] = {}
    for path in sorted(SERVICES_DIR.glob("*/manifest.yaml")):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        service = manifest.get("service") or {}
        service_id = service.get("id") or path.parent.name
        services[service_id] = service
    return services


def load_env_vars() -> dict[str, str]:
    """service id -> env var overriding its published port."""
    entries = json.loads(PORTS_FILE.read_text(encoding="utf-8"))["ports"]
    return {entry["service_id"]: entry["env_var"]
            for entry in entries if entry.get("service_id")}


def load_policy() -> dict[str, dict]:
    return json.loads(POLICY_FILE.read_text(encoding="utf-8"))["services"]


def is_exposed(service: dict) -> bool:
    """Matches the network-exposure contract's own definition, deliberately.

    `external_port_default: 0` still counts: the service declares a
    host-facing port contract even though it currently publishes nothing, and
    the exposure policy demands an entry for it either way.
    """
    return (service.get("external_port_default") is not None
            or bool(service.get("host_network")))


def host_port_cell(service: dict) -> str:
    if service.get("host_network"):
        return "host network"
    external = service.get("external_port_default")
    if not external:  # 0 or absent
        return "internal only"
    return f"`{external}`"


def container_port_cell(service: dict) -> str:
    port = service.get("port")
    return f"`{port}`" if port else "n/a"


def sort_key(row: tuple[str, dict]) -> tuple[int, int, str]:
    """Published ports first and in numeric order -- that is the order someone
    writing firewall rules reads in -- then host-network, then internal-only."""
    service_id, service = row
    if service.get("host_network"):
        return (1, 0, service_id)
    external = service.get("external_port_default") or 0
    if not external:
        return (2, 0, service_id)
    return (0, external, service_id)


def render() -> str:
    manifests = load_manifests()
    env_vars = load_env_vars()
    policy = load_policy()

    rows = sorted(((sid, svc) for sid, svc in manifests.items()
                   if is_exposed(svc)), key=sort_key)

    lines = [HEADER.rstrip(), "", "## Ports", "",
             "| Host port | Container port | Service | Env var | Auth |"
             " Exposure | Notes |",
             "|---|---|---|---|---|---|---|"]
    for service_id, service in rows:
        entry = policy.get(service_id, {})
        auth = entry.get("auth_required")
        lines.append(
            f"| {host_port_cell(service)} | {container_port_cell(service)} |"
            f" `{service_id}` |"
            f" {'`' + env_vars[service_id] + '`' if service_id in env_vars else '—'} |"
            f" {'yes' if auth else 'no' if auth is not None else '—'} |"
            f" {entry.get('lan_exposure', '—')} |"
            f" {entry.get('notes', '—')} |")
    lines.append(FOOTER.rstrip())
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rendered = render()
    if "--write" in sys.argv[1:]:
        DOC_FILE.write_text(rendered, encoding="utf-8")
        print(f"[OK] Wrote {DOC_FILE.relative_to(ROOT_DIR)}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
