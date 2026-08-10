"""Model Deck storage locations — locations.json + on-disk marker files.

A location is a user-declared storage root (a compose bind mount) with a
role (hot/cold), a typed layout, and an optional engine binding. This module
owns ``locations.json`` (atomic writes, corrupt file self-heals to empty)
and the ``.deck-store.json`` marker written at each registered root.

THE MARKER IS THE SAFETY MECHANISM: a location whose marker is missing or
carries a different uuid is *unavailable* — a distinct state from empty. An
unmounted drive must never look like a drained model store (catalog entries
are retained by app.catalog) and must never be written into (a bare
mountpoint dir would swallow writes into the container overlay). Every
writer in the storage feature checks ``available()`` first.

Engine/type pairing is validated at registration: engine "lemonade" requires
store_type "gguf"; engine "comfyui" requires store_type "comfy". This is
what makes role=hot meaningful ("loadable because it sits in the engine's
store"), so a mismatch is a config error, not a warning.

No Settings import — path and disk_usage are injected.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid as uuidlib
from pathlib import Path

from app.engines import GuardError

MARKER_NAME = ".deck-store.json"

_ROLES = {"hot", "cold"}
_STORE_TYPES = {"gguf", "hf", "comfy", "plain"}
_ENGINES = {"lemonade", "comfyui", "none"}
_ENGINE_REQUIRES_TYPE = {"lemonade": "gguf", "comfyui": "comfy"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PATCHABLE = {"role", "watermark_gb", "archive_to", "readonly"}
_REQUIRED = {"name", "path", "role", "store_type", "engine",
             "watermark_gb", "archive_to", "readonly"}


def _validate(spec: dict) -> None:
    missing = _REQUIRED - set(spec)
    extra = set(spec) - _REQUIRED - {"uuid"}
    if missing or extra:
        raise ValueError(f"location spec: missing {sorted(missing)}, unknown {sorted(extra)}")
    if not _NAME_RE.match(spec["name"]):
        raise ValueError(f"location name {spec['name']!r} must be lowercase slug ([a-z0-9-])")
    if spec["role"] not in _ROLES:
        raise ValueError(f"role must be one of {sorted(_ROLES)}")
    if spec["store_type"] not in _STORE_TYPES:
        raise ValueError(f"store_type must be one of {sorted(_STORE_TYPES)}")
    if spec["engine"] not in _ENGINES:
        raise ValueError(f"engine must be one of {sorted(_ENGINES)}")
    required_type = _ENGINE_REQUIRES_TYPE.get(spec["engine"])
    if required_type and spec["store_type"] != required_type:
        raise ValueError(
            f"engine {spec['engine']!r} requires store_type {required_type!r}")
    wm = spec["watermark_gb"]
    if wm is not None and (isinstance(wm, bool) or not isinstance(wm, (int, float)) or wm <= 0):
        raise ValueError("watermark_gb must be a positive number or null")
    if spec["archive_to"] is not None and not isinstance(spec["archive_to"], str):
        raise ValueError("archive_to must be a location name or null")
    if not isinstance(spec["readonly"], bool):
        raise ValueError("readonly must be a bool")


class LocationStore:
    """User-declared storage locations, persisted to `path`."""

    def __init__(self, path: Path, disk_usage=shutil.disk_usage):
        self._path = path
        self._disk_usage = disk_usage
        # One lock around every load-modify-save. Reachable from FastAPI's
        # sync-route threadpool, which runs real OS threads: two concurrent
        # writes to DIFFERENT keys still read-modify-write the SAME file, so
        # one silently loses — and _save writes a fixed .tmp path, so the
        # racing os.replace can also raise FileNotFoundError into a route
        # (a 500). Same fix as the arbiter-facing stores [T9b sweep].
        self._lock = threading.Lock()

    # -- persistence (PolicyStore idiom) ------------------------------------

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save(self, data: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._path)

    # -- API ----------------------------------------------------------------

    def list(self) -> list[dict]:
        return self._load()

    def get(self, name: str) -> dict | None:
        return next((loc for loc in self._load() if loc["name"] == name), None)

    def register(self, spec: dict) -> dict:
        spec = dict(spec)
        spec.pop("uuid", None)
        _validate(spec)
        root = Path(spec["path"])
        if not root.is_dir():
            raise GuardError(f"location path {spec['path']!r} does not exist — is the drive mounted into the container?")
        # The duplicate-name check must be INSIDE the lock, with the append it
        # guards. Outside it, two concurrent registers of the same name both
        # passed the check and both appended — silent corruption of the
        # uniqueness invariant, and worse than a raced write because nothing
        # signals: routers/storage.py and routers/provenance.py build
        # name-keyed dicts from this list, so one entry simply vanishes.
        # NodeStore.add() holds its lock across the identical check for the
        # identical reason.
        with self._lock:
            if self.get(spec["name"]) is not None:
                raise ValueError(f"location {spec['name']!r} already exists")
            spec["uuid"] = uuidlib.uuid4().hex
            marker = root / MARKER_NAME
            try:
                marker.write_text(json.dumps({"uuid": spec["uuid"], "name": spec["name"]}))
            except OSError as exc:
                raise GuardError(f"cannot write marker at {marker}: {exc}") from exc
            data = self._load()
            data.append(spec)
            self._save(data)
        return spec

    def update(self, name: str, patch: dict) -> dict:
        bad = set(patch) - _PATCHABLE
        if bad:
            raise ValueError(f"field(s) not patchable: {sorted(bad)}")
        with self._lock:
            data = self._load()
            for loc in data:
                if loc["name"] == name:
                    merged = {**loc, **patch}
                    _validate({k: v for k, v in merged.items() if k != "uuid"})
                    loc.update(patch)
                    self._save(data)
                    return loc
        raise ValueError(f"unknown location {name!r}")

    def deregister(self, name: str) -> None:
        with self._lock:
            data = [loc for loc in self._load() if loc["name"] != name]
            self._save(data)

    # -- availability ---------------------------------------------------------

    def available(self, spec: dict) -> bool:
        try:
            marker = json.loads((Path(spec["path"]) / MARKER_NAME).read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(marker, dict) and marker.get("uuid") == spec["uuid"]

    def describe(self) -> list[dict]:
        out = []
        for loc in self._load():
            avail = self.available(loc)
            free = total = None
            if avail:
                usage = self._disk_usage(loc["path"])
                free, total = usage.free, usage.total
            out.append({**loc, "available": avail, "free_bytes": free, "total_bytes": total})
        return out
