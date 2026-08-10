"""Model Deck declared-fact store — the small human-owned allowlist.

Everything the Deck can read from an artifact, it reads (see
app.characteristics). This file holds only what genuinely cannot be
derived. The allowlist is the enforcement mechanism for
derive-don't-duplicate: if a human could declare a fact the machine can
read, the two would disagree eventually and the whole layer would become
decoration. Adding a field here is a design decision.

The five fields, and why each resists derivation:

* ``tools_verified`` — only true after an end-to-end tool call actually
  worked. vLLM and ds4 *advertise* tool support they cannot reliably parse;
  that gap cost a day on 2026-08-02.
* ``label`` / ``notes`` — human text, by definition.
* ``tags`` — where retired aliases land: 'fast', 'deep', 'ultimate' describe
  a role, not an identity (see the ontology's naming rule). The taxonomy
  that gives tags meaning is a later increment; this is just the storage.
* ``engine_preference`` — the tie-break when two engines can both serve a
  model and autodetect has no principled reason to pick one.

Human/UI-owned: the machine never writes this file. Missing/corrupt reads
as empty; writes are atomic; a rejected put leaves the file untouched.
"""

import json
import os
import threading
from pathlib import Path

# Each entry: field -> validator. Deliberately tiny; see the docstring.
ALLOWED_FIELDS = {
    "tools_verified": lambda v: isinstance(v, bool),
    "label": lambda v: isinstance(v, str),
    "notes": lambda v: isinstance(v, str),
    "tags": lambda v: isinstance(v, list) and all(isinstance(t, str) for t in v),
    "engine_preference": lambda v: isinstance(v, str),
}


class DeclaredStore:
    """Human-asserted facts keyed ``<kind>/<id>``, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        # One lock around every load-modify-save. Reachable from FastAPI's
        # sync-route threadpool, which runs real OS threads: two concurrent
        # writes to DIFFERENT keys still read-modify-write the SAME file, so
        # one silently loses — and _save writes a fixed .tmp path, so the
        # racing os.replace can also raise FileNotFoundError into a route
        # (a 500). Same fix as the arbiter-facing stores [T9b sweep].
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            text = self._path.read_text()
        except OSError:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp_path, self._path)

    def get(self) -> dict:
        return self._load()

    def entry(self, key: str) -> dict:
        return self._load().get(key, {})

    def put(self, key: str, fields: dict) -> None:
        """Merge `fields` into `key`. Validates everything before writing
        anything, so a rejected put leaves the file untouched."""
        for name, value in fields.items():
            validator = ALLOWED_FIELDS.get(name)
            if validator is None:
                raise ValueError(
                    f"{name!r} is not declarable — if the Deck can read it from the "
                    "artifact it must be derived, not declared"
                )
            if not validator(value):
                raise ValueError(f"{name!r} has the wrong type: {value!r}")

        with self._lock:
            data = self._load()
            data.setdefault(key, {}).update(fields)
            self._save(data)

    def forget(self, key: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(key, None) is not None:
                self._save(data)
