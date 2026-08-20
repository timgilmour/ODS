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


def _env_str(name: str, default: str) -> str:
    """Read a string env var, treating empty/whitespace-only as ABSENT.

    Compose's ``${VAR:-}`` idiom bakes an EMPTY value into the container —
    not an absent one — which silently voids any real default here (the
    class behind the 08-12 serving-blindness incident: NODE_SERVING_* baked
    empty). The deck side closed this with pydantic's
    ``env_ignore_empty=True`` (model-deck app/settings.py:25); this is the
    agent-side equivalent. Real values are stripped: a padded URL or path
    is never what the operator meant.
    """
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


NODE_AGENT_KEY = _env_str("NODE_AGENT_KEY", "")
NODE_NAME = _env_str("NODE_NAME", socket.gethostname())
GPU_BACKEND = _env_str("GPU_BACKEND", "nvidia").lower()
NODE_SERVING_PROBE_URL = _env_str("NODE_SERVING_PROBE_URL", "")
NODE_SERVING_CONTAINER = _env_str("NODE_SERVING_CONTAINER", "")
# Swap control is opt-in: both dirs must be mounted/configured or the
# /v1/node/{profiles,swap} endpoints answer 503. The vllm dir is expected
# read-only (profile discovery); only the ctl dir is written (request.json).
NODE_VLLM_DIR = _env_str("NODE_VLLM_DIR", "")
NODE_SWAP_CTL_DIR = _env_str("NODE_SWAP_CTL_DIR", "")
# engines.json is opt-in like swap control: an explicit override, else it is
# resolved beside profiles.json under NODE_VLLM_DIR (engines.py
# _configured_path()); both unset means the node has no declared engines,
# which is normal and not an error.
NODE_ENGINES_FILE = _env_str("NODE_ENGINES_FILE", "")
# Settings storage is opt-in like swap control: unset means the Deck-owned
# settings document, compose-text and catalog routes all answer 503 rather
# than writing into some implicit default directory.
NODE_SETTINGS_DIR = _env_str("NODE_SETTINGS_DIR", "")
# Instances control is opt-in like swap control: unset means the
# /v1/node/instance/* routes answer 503. This is a rw dir shared with the
# host-side instances-helper -- the agent writes instance-req.json here, the
# helper writes instance-status-<resource>.json back (forensics only).
NODE_INSTANCES_CTL_DIR = _env_str("NODE_INSTANCES_CTL_DIR", "")
# NODE_AGENT_PORT is deliberately absent here: the listening port is owned by
# the Dockerfile CMD (`uvicorn --port ${NODE_AGENT_PORT:-7720}`), so a second
# copy in Python was read by nothing and only added a crash-at-import path.
GPU_CACHE_TTL_SECONDS = _env_float("NODE_GPU_CACHE_TTL", 2.0)
