"""Model Deck desired-state (intent) store.

Records what each resource is *supposed* to be running, so the Deck can
tell three situations apart that today look identical: a model that is
deliberately unloaded, a model that died, and a resource nobody ever asked
anything of. Before this store existed, ods-hipfire sat dead for 26 hours
looking exactly like a deliberate park (2026-08-03).

``intent.json`` is a flat mapping of ``{"<node>/<resource>": record}``.
Two deliberate differences from ``app.policy``:

* **No known-key defaults.** Keys are discovered at runtime, which is what
  lets a second node or a new engine work without a code change. A missing
  file is legitimately empty, not "needs materializing".
* **Intent is last-known-good, recorded implicitly.** Callers record on
  every deliberate action rather than the operator maintaining a wishlist.

``state: "unloaded"`` is intent, NOT an absence of intent — a deliberate
park must be preserved, or the reconciler would fight the operator every
tick. Deleting a key (``forget``) is the only way to say "no opinion".

Writes are atomic (temp file + ``os.replace``); a missing or corrupt file
reads as ``{}`` rather than raising, matching ``app.policy``'s self-healing
quality bar. Writes are serialized across uvicorn handler threads and the
watcher daemon thread via an internal lock; callers need not coordinate.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

# Consecutive restore failures before a key is quarantined (stops the
# reconciler from crash-looping a resource whose config is simply wrong).
FAILURE_BUDGET = 2

VALID_STATES = ("loaded", "unloaded")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IntentStore:
    """Durable desired state per ``<node>/<resource>``, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()

    # --- persistence -------------------------------------------------------

    def _load(self) -> dict[str, dict]:
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

    def _save(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=1))
        os.replace(tmp_path, self._path)

    # --- reads -------------------------------------------------------------

    def get(self) -> dict[str, dict]:
        """Full key->record mapping. Empty when nothing has been recorded."""
        return self._load()

    # --- writes ------------------------------------------------------------

    def record(
        self,
        key: str,
        *,
        state: str,
        model: str | None,
        engine: str,
        now: str | None = None,
    ) -> None:
        """Record a deliberate action as the new intent for `key`.

        Preserves `last_healthy_ts` — the health history of the resource
        itself, unrelated to what the operator wants of it — but RESETS
        `failures`/`quarantined`. A deliberate load or unload is evidence
        that the situation changed (backend fixed, VRAM freed, a different
        model chosen), so the resource earns a fresh restore budget.
        Leaving the flag set would exclude it from automatic restore
        forever, and invisibly: derive_status only reports ``quarantined``
        on the loaded-intent branch, so a quarantined-and-parked resource
        hides the flag while still being permanently excluded.
        """
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of {VALID_STATES}, got {state!r}")

        with self._lock:
            data = self._load()
            previous = data.get(key, {})
            data[key] = {
                "state": state,
                "model": model,
                "engine": engine,
                "updated_ts": now or _now_iso(),
                "last_healthy_ts": previous.get("last_healthy_ts"),
                "failures": 0,
                "quarantined": False,
            }
            self._save(data)

    def note_healthy(self, key: str, now: str | None = None) -> None:
        """Observation confirmed intent. Stamps last_healthy_ts and clears
        the failure budget — a success is what releases a quarantine."""
        with self._lock:
            data = self._load()
            record = data.get(key)
            if record is None:
                return
            record["last_healthy_ts"] = now or _now_iso()
            record["failures"] = 0
            record["quarantined"] = False
            self._save(data)

    def note_failure(self, key: str, now: str | None = None) -> int:
        """A restore attempt failed. Returns the running consecutive count,
        and quarantines the key once it reaches FAILURE_BUDGET."""
        with self._lock:
            data = self._load()
            record = data.get(key)
            if record is None:
                return 0
            record["failures"] = record.get("failures", 0) + 1
            if record["failures"] >= FAILURE_BUDGET:
                record["quarantined"] = True
            self._save(data)
            return record["failures"]

    def clear_failures(self, key: str) -> None:
        """Operator-initiated quarantine release (the UI's 'try again')."""
        with self._lock:
            data = self._load()
            record = data.get(key)
            if record is None:
                return
            record["failures"] = 0
            record["quarantined"] = False
            self._save(data)

    def forget(self, key: str) -> None:
        """Drop all intent for `key` — 'the Deck has no opinion about this'."""
        with self._lock:
            data = self._load()
            if data.pop(key, None) is not None:
                self._save(data)
