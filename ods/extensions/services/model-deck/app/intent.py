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
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

# Consecutive restore failures before a key is quarantined (stops the
# reconciler from crash-looping a resource whose config is simply wrong).
FAILURE_BUDGET = 2

_log = logging.getLogger(__name__)

VALID_STATES = ("loaded", "unloaded")

# Who authored a record: a human-initiated route (or a hook completing an
# operator's own earlier request), or the arbiter's own automatic actions
# (idle-release, contention-eviction, pending-load retrigger). Introduced
# for app.routers.control's pull-through supersession check (task 6 follow-
# up, max-review Important-1): the deck's own automatic churn must not be
# mistaken for an operator overriding a pull-through load in flight.
VALID_ACTORS = ("operator", "deck")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IntentStore:
    """Durable desired state per ``<node>/<resource>``, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        # Keys already warned about by the boundary gate — see _load().
        self._warned: set[str] = set()

    # --- persistence -------------------------------------------------------

    def _well_formed(self, key: str, record: object) -> bool:
        """One record's boundary check. `model` is a PRESENCE check, never a
        truthiness one: ``model=None`` is a legitimate intent ("loaded, no
        opinion which model" — the correct reading for single-model engines
        like hipfire, see app.lifecycle's `wanted is None` branch).

        `actor` is optional — a pre-upgrade intent.json has none at all, and
        every reader treats missing as "operator" — but a PRESENT one must be
        valid, or app.routers.control's supersession check silently reads it
        as non-operator.
        """
        if not isinstance(record, dict):
            return False
        if record.get("state") not in VALID_STATES:
            return False
        if not record.get("engine"):
            return False
        if "model" not in record:
            return False
        if "actor" in record and record["actor"] not in VALID_ACTORS:
            return False
        return True

    def _load(self) -> dict[str, dict]:
        """Missing/corrupt file reads as empty; every surviving record then
        passes the per-record boundary gate.

        THE GATE LIVES HERE AND NOWHERE ELSE [T7 review Important-1]. This
        store used to check only the whole-FILE shape, one level up from
        where the damage was: consumers hard-index these records
        (app/lifecycle.py:62 and app/reconcile.py:57 both do
        ``intent["model"]``), so a single malformed record crashed the
        reconcile pass — and, after T7, could make the arbiter's own rollback
        raise from inside the handler whose whole job is isolating a failure.

        Malformed records are DROPPED, not repaired: unlike PolicyStore there
        is no default to heal an intent to, and "no intent" is the safe
        reading — nothing gets restored, rather than something wrong getting
        restored.

        Deliberately does NOT persist the heal, which is where this departs
        from PolicyStore._load's otherwise-identical pattern: PolicyStore has
        no lock, while this store's whole point is one guarding
        load-modify-save, and ``get()`` reads without holding it. Writing
        from an unlocked read path would race every mutation. The heal lands
        on the next ``_save()`` instead; until then no consumer can see the
        bad record anyway, which is the property that matters.
        """
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
        gated = {}
        for key, record in data.items():
            if self._well_formed(key, record):
                gated[key] = record
            elif key not in self._warned:
                # Once per key per process: _load runs on every arbiter tick,
                # so an unguarded warning here would be tick-rate spam for as
                # long as the file stays unhealed.
                self._warned.add(key)
                _log.warning("dropping malformed intent record %r from %s",
                             key, self._path)
        return gated

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
        # Default "operator", NOT "deck": nearly every call site is a
        # router-initiated record (a human-initiated route, or a hook
        # completing an operator's own earlier request), and the one place
        # this distinction is safety-critical — app.routers.control's
        # pull-through supersession check — must fail SAFE if a future
        # caller forgets the kwarg. A forgotten kwarg that defaulted to
        # "deck" would silently make an unlabeled record invisible to that
        # check (never supersede a stale pull), the wrong direction to
        # fail; defaulting to "operator" means a forgotten kwarg still
        # protects an operator's later action. Only app.arbiter's own two
        # automatic records (idle-release/contention-eviction unload,
        # pending-load retrigger) pass actor="deck" explicitly.
        actor: str = "operator",
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

        ``actor`` (see the parameter default's comment above) distinguishes
        who authored this record: ``"operator"`` or ``"deck"``. Persisted
        and returned verbatim — a legacy record with no ``actor`` key at
        all (pre-upgrade intent.json) is NOT normalized here; a reader that
        cares (``app.routers.control``'s supersession check) treats a
        missing field as ``"operator"`` itself, conservative in the same
        direction as this default.
        """
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of {VALID_STATES}, got {state!r}")
        if actor not in VALID_ACTORS:
            raise ValueError(f"actor must be one of {VALID_ACTORS}, got {actor!r}")

        with self._lock:
            data = self._load()
            previous = data.get(key, {})
            data[key] = {
                "state": state,
                "model": model,
                "engine": engine,
                "actor": actor,
                "updated_ts": now or _now_iso(),
                "last_healthy_ts": previous.get("last_healthy_ts"),
                "failures": 0,
                "quarantined": False,
            }
            self._save(data)

    def put_back(self, key: str, record: dict) -> None:
        """Restore `record` for `key` VERBATIM — the rollback primitive for an
        actuation that was recorded and then failed. Semantics: "as if the
        failed actuation never happened."

        Deliberately NOT ``record()``. An arm that pre-records its intent
        (``app.arbiter._execute``'s unload arm — "whoever actuates, records")
        must undo that write exactly on failure, and re-recording gets three
        fields wrong:

        * ``actor`` — re-records default to ``"operator"``, and stamping the
          arbiter's own ``"deck"`` is equally wrong. Either way the label no
          longer describes who actually authored the surviving intent, and
          ``app.routers.control``'s pull-through supersession check reads
          exactly that label to decide whether an operator overrode a pull
          in flight.
        * ``updated_ts`` — the settings-drift baseline
          (``app.routers.__init__``), documented there to advance only at a
          DELIBERATE load/unload: the moment a process relaunches and
          re-consumes its settings. A failed unload relaunched nothing, so a
          fresh stamp silently clears a legitimate "settings changed since
          launch" flag.
        * ``failures``/``quarantined`` — ``record()`` resets both. The
          pre-record already cleared them, so only a verbatim put-back can
          return a quarantined key to its quarantine; re-recording would put
          a crash-looping resource back into the restore rotation.

        Refuses anything that isn't a well-formed record rather than coercing
        it — a bad shape persisted here is one every reader downstream has to
        defend against. It shares ``_well_formed`` with the ``_load`` boundary
        gate deliberately: two hand-written copies of "what a record is" would
        drift, and this one would be the lenient copy that lets a bad record
        back in behind the gate.

        An earlier version of this check took ``state`` + ``engine`` only,
        on the stated grounds that those were "the fields
        derive_status/plan_reconcile dereference". That was simply wrong —
        both also hard-index ``model`` (app/lifecycle.py:62,
        app/reconcile.py:57), so a record accepted without it persisted and
        KeyError'd the next reconcile pass [T7 review Important-2].

        SINCE ``_load`` GATES, this raise is unreachable from the arbiter's
        rollback path — ``prior`` comes from ``get()``, which now yields only
        well-formed records. It stays as an internal assertion for other
        callers: raising here from inside the arbiter's ``except EngineError``
        handler would abort the very tick that handler exists to keep alive.
        """
        if not self._well_formed(key, record):
            raise ValueError(
                f"put_back({key!r}) needs a well-formed record — a dict with "
                f"state in {VALID_STATES}, a truthy engine, a 'model' key "
                f"(None is allowed), and any 'actor' in {VALID_ACTORS}; "
                f"got {record!r}")

        with self._lock:
            data = self._load()
            data[key] = dict(record)  # copied: the caller's dict stays theirs
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
