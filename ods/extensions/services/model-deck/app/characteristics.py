"""Model Deck derived-fact cache.

The Deck reads facts from the things that own them — a checkpoint's
config.json, an engine's /v1/models — and caches them here. It never
re-declares them. Anything stored twice eventually disagrees: that is the
`--quantization modelopt` crash loop in one sentence (a compose flag
contradicting the checkpoint's own config.json), and the reason context
length currently lives in three unreconciled places.

Keys are ``"<kind>/<id>"`` — ``model/Qwen3.6-35B-A3B-heretic-NVFP4``,
``engine/boxa/vllm``. Values are ``{field: {value, source, derived_ts}}``.

Provenance is mandatory, not decoration. "context: 262144" read from a
checkpoint, read from a live engine, and asserted by a human have very
different reliability, and an operator staring at an incident needs to know
which one they are looking at. ``put_fields`` refuses a field that arrives
without ``source``/``derived_ts`` so a forgetful deriver fails loudly here
rather than quietly poisoning the cache.

Machine-owned: never hand-edit this file (see app.declared for the human
half). Missing/corrupt reads as empty and self-heals, writes are atomic —
same quality bar as app.policy and app.intent.
"""

import threading
from pathlib import Path
from app.store_io import load_json, save_json

_REQUIRED_FIELD_KEYS = ("value", "source", "derived_ts")


class CharacteristicsStore:
    """Derived facts keyed ``<kind>/<id>``, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        # The watcher thread's derive pass and HTTP request threads share ONE
        # instance, and put_fields/forget are load-modify-save. Unlocked they
        # lose writes (stale read) AND race _save's atomic replace, which
        # uses one fixed .tmp path per store [max-review c4]. IntentStore and
        # SettingsStore already lock; this was the odd one out.
        self._lock = threading.Lock()

    def _load(self) -> dict:
        data = load_json(self._path)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict) -> None:
        save_json(self._path, data, indent=1, sort_keys=True)

    def get(self) -> dict:
        return self._load()

    def entry(self, key: str) -> dict:
        return self._load().get(key, {})

    def put_fields(self, key: str, fields: dict[str, dict]) -> None:
        """Merge `fields` into `key`'s entry, leaving other fields alone.

        Merging rather than replacing matters: checkpoint derivation and
        live-surface derivation both write to the same entry and know
        nothing about each other's fields.
        """
        for name, field in fields.items():
            if not isinstance(field, dict):
                # `k not in field` below does substring matching on a str
                # (e.g. "value" in "source,value,derived_ts" is True), which
                # can pass the provenance check on a caller's typo instead of
                # rejecting it. Guard the type explicitly.
                raise ValueError(
                    f"field {name!r} of {key!r} must be a dict with "
                    f"{_REQUIRED_FIELD_KEYS}, got {type(field).__name__}"
                )
            missing = [k for k in _REQUIRED_FIELD_KEYS if k not in field]
            if missing:
                raise ValueError(
                    f"field {name!r} of {key!r} is missing {missing} — "
                    "every derived fact must carry its provenance"
                )

        # Validation above is pure and stays outside the lock; only the
        # load-modify-save needs serializing.
        with self._lock:
            data = self._load()
            entry = data.setdefault(key, {})
            entry.update(fields)
            self._save(data)

    def forget(self, key: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(key, None) is not None:
                self._save(data)
