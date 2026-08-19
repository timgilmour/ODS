"""Model Deck footprint registry.

Tracks VRAM footprints for GGUF models: a cold-start estimate derived from
on-disk file size (x1.2 headroom). The measured-actuals half (observe() /
registry.json) was deleted (open-rulings #4, 2026-08-17) — it had zero live
callers anywhere in the tree, so footprint() was already, in every production
run, identical to estimate(); this collapses that dead branch instead of
carrying it forward. Re-wiring real measurement later means re-introducing a
store WITH the per-store lock the deleted one documented as missing.
"""

from pathlib import Path

# Hipfire has no live VRAM introspection of its own; this is the fixed
# footprint budgeted for it regardless of which model it's serving.
HIPFIRE_FOOTPRINT = 33_000_000_000

_ESTIMATE_FACTOR = 1.2


class Registry:
    """Footprint estimates for GGUF models in `gguf_dir`."""

    def __init__(self, gguf_dir: Path):
        self._gguf_dir = gguf_dir

    def estimate(self, model_file: str) -> int:
        """Cold-start footprint estimate: on-disk size x1.2.

        Raises FileNotFoundError if `model_file` isn't in gguf_dir.
        """
        size = (self._gguf_dir / model_file).stat().st_size
        return int(size * _ESTIMATE_FACTOR)

    def footprint(self, model_file: str) -> int:
        """The file-size estimate — footprint()'s only source now that the
        measured-actuals half is gone; byte-identical to every production
        call before the deletion, since nothing ever wrote a measured
        value."""
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
        out = []
        for entry in entries:
            size = entry.stat().st_size
            out.append({
                "file": entry.name,
                "size": size,
                "footprint": int(size * _ESTIMATE_FACTOR),
            })
        return out
