"""Model Deck footprint registry.

Tracks VRAM footprints for GGUF models: a cold-start estimate derived from
on-disk file size, and measured actuals observed at runtime once a model
has actually been loaded once. Measured always wins over estimated.

``registry.json`` is a flat mapping of ``{model_file: measured_bytes}``,
persisted next to no other state — this module owns the whole file.
Writes are atomic (temp file + ``os.replace``) since the supervisor may
crash mid-write; a corrupt or unreadable file is treated as empty rather
than raised, since it self-heals on the next ``observe()`` and Model Deck
still has the file-size estimate to fall back on in the meantime.

This is single-process, in-process state only — no cross-process locking.
The supervisor is the sole owner of registry.json.
"""

import json
import os
from pathlib import Path

# Hipfire has no live VRAM introspection of its own; this is the fixed
# footprint budgeted for it regardless of which model it's serving.
HIPFIRE_FOOTPRINT = 33_000_000_000

# Default VRAM reserved for ComfyUI when no measured/estimated footprint
# is otherwise available for it.
COMFYUI_RESERVE_DEFAULT = 24_000_000_000

_ESTIMATE_FACTOR = 1.2


class Registry:
    """Footprint estimates + measured actuals for GGUF models in `gguf_dir`."""

    def __init__(self, path: Path, gguf_dir: Path):
        self._path = path
        self._gguf_dir = gguf_dir

    def _load(self) -> dict[str, int]:
        try:
            text = self._path.read_text()
        except OSError:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, data: dict[str, int]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self._path)

    def estimate(self, model_file: str) -> int:
        """Cold-start footprint estimate: on-disk size x1.2.

        Raises FileNotFoundError if `model_file` isn't in gguf_dir.
        """
        size = (self._gguf_dir / model_file).stat().st_size
        return int(size * _ESTIMATE_FACTOR)

    # NOT lock-guarded, unlike every sibling store [T9b]: this is the only
    # write path here and it has ZERO live callers (measured footprints were
    # never wired up). A lock on dead code invents a contract nobody uses —
    # but the fixed-tmp-path race is real, so anyone re-wiring this must add
    # the per-store threading.Lock the siblings use (app/intent.py's is the
    # reference) in the same change.
    def observe(self, model_file: str, measured_bytes: int) -> None:
        """Record a measured footprint, persisted to registry.json immediately."""
        data = self._load()
        data[model_file] = measured_bytes
        self._save(data)

    def footprint(self, model_file: str) -> int:
        """Measured footprint if known, else the file-size estimate."""
        data = self._load()
        if model_file in data:
            return data[model_file]
        return self.estimate(model_file)

    def scan(self) -> list[dict]:
        """List GGUF files in gguf_dir (non-recursive, sorted by name).

        Each entry: {"file": name, "size": bytes, "footprint": footprint(name)}.
        Returns [] if gguf_dir doesn't exist.
        """
        try:
            entries = sorted(self._gguf_dir.glob("*.gguf"))
        except OSError:
            return []
        return [
            {
                "file": entry.name,
                "size": entry.stat().st_size,
                "footprint": self.footprint(entry.name),
            }
            for entry in entries
        ]
