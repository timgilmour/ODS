"""File-protocol settings documents, compose text and the harvest catalog.

The privilege split mirrors swapctl.py: this agent is LAN-facing and has no
docker access, so a settings PUT is only ever a JSON document dropped into
the shared settings dir. The host-side swap-helper (Task 4) reads
<dir>/<profile>.json and renders it into a compose override, and writes
catalog-<profile>.json after a harvest. Disabled entirely unless
NODE_SETTINGS_DIR is configured.
"""

import json
import os
from pathlib import Path

import nodeconfig
from swapctl import _NAME_RE, InvalidProfile

# The one shape allowed to cross the Deck<->node boundary. Exactly these keys:
# extras would mean the Deck had started asserting things (e.g. "volumes")
# this node alone is supposed to own.
EMPTY = {"args": {}, "env": {}, "argv": [], "service": None}
_KEYS = frozenset(EMPTY)


class SettingsDisabled(Exception):
    pass


def _dir() -> Path:
    raw = (nodeconfig.NODE_SETTINGS_DIR or "").strip()
    if not raw:
        raise SettingsDisabled()
    return Path(raw)


def _validate_name(profile: str) -> None:
    if not _NAME_RE.match(profile or ""):
        raise InvalidProfile(profile)


def _validate_document(document: dict) -> None:
    if not isinstance(document, dict) or set(document) != _KEYS:
        raise ValueError(f"settings document must have exactly the keys {sorted(_KEYS)}")
    if not isinstance(document["args"], dict):
        raise ValueError("args must be an object")
    if not isinstance(document["env"], dict):
        raise ValueError("env must be an object")
    argv = document["argv"]
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError("argv must be a list of strings")
    service = document["service"]
    if service is not None and not isinstance(service, str):
        raise ValueError("service must be a string or null")


def read_settings(profile: str) -> dict:
    _validate_name(profile)
    path = _dir() / f"{profile}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        # No settings yet is every profile's starting state, not an error --
        # same contract as swapctl.read_status() returning None.
        return dict(EMPTY)
    except (OSError, ValueError):
        # A half-written or corrupt document is reported as absent rather
        # than crashing the read path; the next PUT overwrites it atomically.
        return dict(EMPTY)


def write_settings(profile: str, document: dict) -> None:
    _validate_name(profile)
    _validate_document(document)
    directory = _dir()
    path = directory / f"{profile}.json"
    tmp = directory / f".{profile}.json.tmp"
    tmp.write_text(json.dumps(document))
    os.replace(tmp, path)  # atomic: a reader never observes a partial write


def read_newest_catalog() -> dict | None:
    """Newest catalog-*.json by harvested_ts, or None before any harvest.

    harvested_ts is ISO-8601 with a Z suffix, which sorts correctly as a
    plain string -- no datetime parsing needed. A corrupt or short-written
    file (the helper writes these too, non-atomically on its side) is
    skipped rather than failing the whole read.
    """
    directory = _dir()
    newest = None
    newest_ts = ""
    for path in directory.glob("catalog-*.json"):
        try:
            data = json.loads(path.read_text())
            ts = data["harvested_ts"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(ts, str):
            continue
        if newest is None or ts > newest_ts:
            newest_ts = ts
            newest = data
    return newest
