"""Model Deck model catalog — typed units across storage locations.

Owns ``catalog.json`` (atomic writes, corrupt file rebuilt by the next
scan). ``scan()`` walks every AVAILABLE location and rebuilds units from
disk; sticky metadata (pinned, last_used, a mid-move "moving" state) is
merged back in by unit id so a rescan never forgets operator intent.

THE RETENTION RULE (safety-critical, mirrors app.locations' marker design):
units whose location exists but is unavailable are RETAINED from the
previous persisted catalog with state="unavailable" — an unplugged cold
drive must not make its models vanish. Units of a DEREGISTERED location are
dropped (the operator explicitly removed it).

The catalog answers "where does this model live"; app.registry keeps
answering "how big does it run" (VRAM footprints). They stay separate.

No Settings import — everything injected.
"""

import json
import os
import time
from pathlib import Path

from app.locations import MARKER_NAME

_SKIP_SUFFIXES = (".part", ".deck-staging")
_STICKY = ("pinned", "last_used")


def _skip(path: Path) -> bool:
    return path.name == MARKER_NAME or path.name.endswith(_SKIP_SUFFIXES)


def _tree_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not _skip(p))


def _build_units(loc: dict) -> list[dict]:
    root = Path(loc["path"])
    units: list[dict] = []

    def unit(relpath: str, name: str, utype: str, size: int, mtime: float) -> dict:
        return {"id": f"{loc['name']}:{relpath}", "type": utype, "name": name,
                "location": loc["name"], "relpath": relpath, "size": size,
                "mtime": mtime, "state": "resident", "pinned": False, "last_used": None}

    if loc["store_type"] == "gguf":
        for p in sorted(root.glob("*.gguf")):
            if not _skip(p):
                st = p.stat()
                units.append(unit(p.name, p.name, "gguf", st.st_size, st.st_mtime))
    elif loc["store_type"] == "hf":
        for p in sorted(root.iterdir()):
            if p.is_dir() and not _skip(p):
                units.append(unit(p.name, p.name, "hf_repo", _tree_size(p), p.stat().st_mtime))
    elif loc["store_type"] == "comfy":
        for p in sorted(root.rglob("*")):
            if p.is_file() and not _skip(p):
                rel = p.relative_to(root).as_posix()
                st = p.stat()
                units.append(unit(rel, p.name, "comfy", st.st_size, st.st_mtime))
    else:  # plain
        for p in sorted(root.iterdir()):
            if _skip(p):
                continue
            size = _tree_size(p) if p.is_dir() else p.stat().st_size
            units.append(unit(p.name, p.name, "plain", size, p.stat().st_mtime))
    return units


class Catalog:
    """Typed cross-location unit catalog, persisted to `path`."""

    def __init__(self, path: Path, location_store, clock=time.time):
        self._path = path
        self._locations = location_store
        self._clock = clock

    # -- persistence ---------------------------------------------------------

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save(self, units: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(units))
        os.replace(tmp, self._path)

    # -- API -----------------------------------------------------------------

    def scan(self) -> list[dict]:
        previous = {u["id"]: u for u in self._load()}
        known_names = set()
        fresh: list[dict] = []
        for loc in self._locations.list():
            known_names.add(loc["name"])
            if self._locations.available(loc):
                for u in _build_units(loc):
                    old = previous.get(u["id"])
                    if old:
                        for key in _STICKY:
                            u[key] = old[key]
                        if old["state"] == "moving":
                            u["state"] = "moving"
                    fresh.append(u)
            else:
                # RETENTION: unavailable ≠ empty — keep last-known units.
                for u in previous.values():
                    if u["location"] == loc["name"]:
                        fresh.append({**u, "state": "unavailable"})
        # Deregistered locations' units drop out naturally (not in known_names).
        fresh = [u for u in fresh if u["location"] in known_names]
        self._save(fresh)
        return fresh

    def units(self) -> list[dict]:
        return self._load()

    def get(self, unit_id: str) -> dict | None:
        return next((u for u in self._load() if u["id"] == unit_id), None)

    def note_used_gguf(self, filename: str) -> None:
        units = self._load()
        hit = False
        for u in units:
            if u["type"] == "gguf" and u["name"] == filename:
                u["last_used"] = self._clock()
                hit = True
        if hit:
            self._save(units)

    def set_pinned(self, unit_id: str, pinned: bool) -> dict:
        units = self._load()
        for u in units:
            if u["id"] == unit_id:
                u["pinned"] = bool(pinned)
                self._save(units)
                return u
        raise ValueError(f"unknown unit {unit_id!r}")

    def set_state(self, unit_id: str, state: str) -> None:
        if state not in ("resident", "moving"):
            raise ValueError(f"bad unit state {state!r}")
        units = self._load()
        for u in units:
            if u["id"] == unit_id:
                u["state"] = state
                self._save(units)
                return
        raise ValueError(f"unknown unit {unit_id!r}")

    def record_moved(self, unit_id: str, dest_location: str) -> dict:
        units = self._load()
        for i, u in enumerate(units):
            if u["id"] == unit_id:
                moved = {**u, "location": dest_location,
                         "id": f"{dest_location}:{u['relpath']}", "state": "resident"}
                units[i] = moved
                self._save(units)
                return moved
        raise ValueError(f"unknown unit {unit_id!r}")
