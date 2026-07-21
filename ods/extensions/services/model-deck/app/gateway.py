"""Container default-gateway detection.

The ods-host-agent binds the ods-network's host-side gateway IP (not
127.0.0.1, not host.docker.internal's default-bridge address, which
Docker's inter-network isolation blocks from custom networks). This
mirrors dashboard-api's config.py:_detect_container_default_gateway.
"""

from pathlib import Path


def detect_default_gateway(route_path: str = "/proc/net/route") -> str:
    """Return this container's default-gateway IP, or "" on any failure.

    /proc/net/route: default route has destination 00000000 and a
    little-endian-hex gateway in the 3rd field; flags must carry
    RTF_GATEWAY (0x2).
    """
    try:
        lines = Path(route_path).read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return ""
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 4 or fields[1] != "00000000":
            continue
        gw_hex = fields[2]
        try:
            flags = int(fields[3], 16)
            gateway_raw = int(gw_hex, 16)
        except ValueError:
            continue
        if not (flags & 0x2) or gateway_raw == 0 or len(gw_hex) != 8:
            continue
        return ".".join(str(int(gw_hex[i:i + 2], 16)) for i in (6, 4, 2, 0))
    return ""
