"""compose.swap.yaml.disabled must not carry one machine's paths as defaults.

The fork-divergence review (2026-08-12, finding 1) flagged the swap overlay's
volume defaults: ``${HOST_VLLM_DIR:-/home/tim/vllm}`` is a wrong default on
any other install, and a wrong-but-plausible mount fails SILENTLY (empty
profile dir -> "no profiles" instead of "you never told me where they are").
Compose's ``${VAR:?message}`` form fails loudly at config time instead, which
is the honest behavior for a path only the operator can know.
"""
import re
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parent.parent

HOST_DIR_VARS = ("HOST_VLLM_DIR", "HOST_SWAP_CTL_DIR", "HOST_SETTINGS_DIR")


def _read(name: str) -> str:
    path = SERVICE / name
    if not path.exists():
        pytest.skip(f"{name} not present (deployed container)")
    return path.read_text(encoding="utf-8")


def test_swap_overlay_requires_host_dirs_explicitly():
    compose = _read("compose.swap.yaml.disabled")
    for var in HOST_DIR_VARS:
        assert re.search(r"\$\{" + var + r":\?", compose), (
            f"{var} must use the required form ${{{var}:?...}} — a baked-in "
            "default is one machine's path on every other machine"
        )
    assert "/home/tim" not in compose, "no machine-specific path may survive as a default"
