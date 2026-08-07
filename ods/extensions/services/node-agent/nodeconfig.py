import os
import socket


def _env_float(name: str, default: float) -> float:
    """Parse a numeric env var, failing with a one-line message.

    A bare ``float(os.environ[...])`` at import time turned a config typo into
    a container crash-loop whose only output was a raw traceback, with nothing
    in it naming the offending variable.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(
            f"ods-node-agent: {name} must be a number, got {raw!r}"
        ) from None


NODE_AGENT_KEY = os.environ.get("NODE_AGENT_KEY", "")
NODE_NAME = os.environ.get("NODE_NAME", socket.gethostname())
GPU_BACKEND = os.environ.get("GPU_BACKEND", "nvidia").lower()
NODE_SERVING_PROBE_URL = os.environ.get("NODE_SERVING_PROBE_URL", "")
NODE_SERVING_CONTAINER = os.environ.get("NODE_SERVING_CONTAINER", "")
# Swap control is opt-in: both dirs must be mounted/configured or the
# /v1/node/{profiles,swap} endpoints answer 503. The vllm dir is expected
# read-only (profile discovery); only the ctl dir is written (request.json).
NODE_VLLM_DIR = os.environ.get("NODE_VLLM_DIR", "")
NODE_SWAP_CTL_DIR = os.environ.get("NODE_SWAP_CTL_DIR", "")
# Settings storage is opt-in like swap control: unset means the Deck-owned
# settings document, compose-text and catalog routes all answer 503 rather
# than writing into some implicit default directory.
NODE_SETTINGS_DIR = os.environ.get("NODE_SETTINGS_DIR", "")
# NODE_AGENT_PORT is deliberately absent here: the listening port is owned by
# the Dockerfile CMD (`uvicorn --port ${NODE_AGENT_PORT:-7720}`), so a second
# copy in Python was read by nothing and only added a crash-at-import path.
GPU_CACHE_TTL_SECONDS = _env_float("NODE_GPU_CACHE_TTL", 2.0)
