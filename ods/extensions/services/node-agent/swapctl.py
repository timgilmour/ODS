"""File-protocol swap control for the serving profile.

The privilege split (see swap-helper/swap-helper.sh): this agent is
LAN-facing and deliberately has no docker access, so a swap is only ever a
request.json dropped into the shared ctl dir. The host-side helper validates
again and runs swap.sh; its status.json is the single source of truth for
swap progress. Disabled entirely unless both dirs are configured.
"""

import json
import logging
import re
import uuid
from pathlib import Path

import nodeconfig

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_logger = logging.getLogger(__name__)
_warned_profiles_json = False

_META_DEFAULTS = {"engine": "vllm", "health_url": None, "container": None}


class SwapCtlDisabled(Exception):
    pass


class UnknownProfile(Exception):
    pass


class InvalidProfile(Exception):
    pass


class SwapInProgress(Exception):
    pass


def _dirs() -> tuple[Path, Path]:
    vllm = (nodeconfig.NODE_VLLM_DIR or "").strip()
    ctl = (nodeconfig.NODE_SWAP_CTL_DIR or "").strip()
    if not vllm or not ctl:
        raise SwapCtlDisabled()
    return Path(vllm), Path(ctl)


def enabled() -> bool:
    try:
        _dirs()
        return True
    except SwapCtlDisabled:
        return False


def _profiles_meta_map() -> dict:
    """Read <vllm>/profiles.json; absent/malformed -> {} (warn once)."""
    global _warned_profiles_json
    vllm, _ = _dirs()
    path = vllm / "profiles.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("not an object")
        _warned_profiles_json = False
        return data
    except (ValueError, OSError) as exc:
        if not _warned_profiles_json:
            _logger.warning("profiles.json unreadable, using defaults: %s", exc)
            _warned_profiles_json = True
        return {}


def _meta_from_entry(name: str, entry) -> dict:
    if not isinstance(entry, dict):
        # profiles.json is valid JSON but this entry isn't an object (e.g.
        # {"comfyui": 5}) -- treat it the same as a missing entry rather than
        # crashing .get() below. This is polled continuously by
        # GET /v1/node/serving, so a malformed sidecar must degrade to
        # defaults, not a 500.
        entry = {}
    meta = {"name": name}
    for key, default in _META_DEFAULTS.items():
        meta[key] = entry.get(key, default)
    return meta


def profile_meta(name: str) -> dict:
    """Return metadata dict for profile, with defaults for missing fields."""
    return _meta_from_entry(name, _profiles_meta_map().get(name) or {})


def current_profile_meta() -> dict | None:
    """Return metadata for the current profile, or None if no valid status."""
    status = read_status()
    if not status or status.get("state") == "error":
        return None
    name = status.get("profile")
    return profile_meta(name) if name else None


def list_profiles() -> list[dict]:
    vllm, _ = _dirs()
    names = sorted(p.name[len("compose-"):-len(".yaml")]
                   for p in vllm.glob("compose-*.yaml"))
    # One profiles.json read for the whole listing: this runs on every
    # serving poll (serving.probe_url_warning), and per-name profile_meta()
    # would re-read the sidecar N times per poll.
    meta_map = _profiles_meta_map()
    return [_meta_from_entry(n, meta_map.get(n) or {}) for n in names]


def read_status() -> dict | None:
    _, ctl = _dirs()
    path = ctl / "status.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # A half-written or corrupt status is reported as absent rather than
        # crashing the read path; the helper rewrites it atomically on the
        # next transition.
        return None


def request_swap(profile: str) -> str:
    vllm, ctl = _dirs()
    if not _NAME_RE.match(profile or ""):
        raise InvalidProfile(profile)
    if not (vllm / f"compose-{profile}.yaml").is_file():
        raise UnknownProfile(profile)
    if (ctl / "request.json").exists():
        raise SwapInProgress("a swap request is already pending")
    status = read_status()
    if status and status.get("state") == "swapping":
        raise SwapInProgress("helper is mid-swap")
    req_id = str(uuid.uuid4())
    tmp = ctl / f".request.{req_id}.tmp"
    tmp.write_text(json.dumps({"id": req_id, "profile": profile}))
    tmp.rename(ctl / "request.json")
    return req_id
