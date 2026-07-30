"""File-protocol swap control for the serving profile.

The privilege split (see swap-helper/swap-helper.sh): this agent is
LAN-facing and deliberately has no docker access, so a swap is only ever a
request.json dropped into the shared ctl dir. The host-side helper validates
again and runs swap.sh; its status.json is the single source of truth for
swap progress. Disabled entirely unless both dirs are configured.
"""

import json
import re
import uuid
from pathlib import Path

import nodeconfig

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def list_profiles() -> list[str]:
    vllm, _ = _dirs()
    return sorted(p.name[len("compose-"):-len(".yaml")]
                  for p in vllm.glob("compose-*.yaml"))


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
