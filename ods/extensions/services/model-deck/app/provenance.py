"""Model Deck provenance store — where every artifact came from, and what
version of it is here now.

Owns ``provenance.json``: a flat ``{artifact_id: entry}`` mapping, atomic
writes, one lock. No known-key defaults — keys are discovered at runtime, the
same reasoning app.intent gives for intent.json: a new node or a new engine
must not require a code change, so a missing file is legitimately empty.

TWO FIELDS THAT LOOK ALIKE AND ARE NOT. ``current.version`` is the exact
machine identity nobody types (a digest, a commit, a sha256); ``current.label``
is the string a human recognises (``v0.5.6``). One is for comparing, the other
for reading, and neither is derivable from the other — which is precisely the
distinction app/engines/docker_ctl.py's image_ref docstring has been asking
for. ``label: None`` is honest for the many artifacts with no human version.

CORRUPT FILES ARE QUARANTINED, NOT EMPTIED (design decision D13). The house
pattern for policy/registry/catalog is to self-heal a corrupt file to empty,
which is safe there because all three are re-derivable from the world. This
file is not: it is the only home of operator-declared origins. So a corrupt
document is renamed to ``provenance.json.corrupt-<ts>`` first, and the
history JSONL remains the rebuild path.

Provenance NEVER actuates (D4/D11). It records desired state as data and
reports drift; app.reconcile stays the only restorer, and convergence — which
would need `docker pull` + container create, or a compose write on sparky —
is a later spec gated on a socket-proxy permission decision.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from app import origins, provenance_history, updates

_SOURCE_DERIVED = "derived"
_SOURCE_DECLARED = "declared"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _empty_current() -> dict:
    return {"version": None, "label": None, "detail": {},
            "source": _SOURCE_DECLARED, "observed_at": None,
            "verification": origins.UNKNOWN, "verified_at": None}


def _blank_entry(artifact_id: str, kind: str, node: str, role: str) -> dict:
    return {"artifact_id": artifact_id, "kind": kind, "node": node, "role": role,
            "origin": None, "current": _empty_current(), "desired": None,
            "update_path": None, "notes": None, "watch": [], "update": None}


def _validate(artifact_id: str, kind: str, node: str, role: str) -> None:
    parsed_kind, parsed_node, _ = origins.parse_artifact_id(artifact_id)
    if parsed_kind != kind or parsed_node != node:
        raise origins.BadArtifactId(
            f"artifact id {artifact_id!r} disagrees with kind={kind!r} node={node!r}")
    if role not in origins.ROLES:
        raise ValueError(f"role must be one of {list(origins.ROLES)}, got {role!r}")


class ProvenanceStore:
    """Current + desired provenance, persisted to `path`, with transitions
    appended to `history_path`."""

    def __init__(self, path: Path, history_path: Path):
        self._path = path
        self._history_path = history_path
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
            self._quarantine()
            return {}
        if not isinstance(data, dict):
            self._quarantine()
            return {}
        return data

    def _quarantine(self) -> None:
        """Preserve an unreadable document before starting empty (D13)."""
        stamp = _now_iso().replace(":", "").replace("-", "")
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        try:
            os.replace(self._path, target)
        except OSError:
            pass

    def _save(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self._path)

    def _record(self, artifact_id, field, before, after, cause, actor, now) -> None:
        provenance_history.append(self._history_path, {
            "ts": now, "artifact_id": artifact_id, "field": field,
            "from": before, "to": after, "cause": cause, "actor": actor})

    # --- reads -------------------------------------------------------------

    def get(self) -> dict[str, dict]:
        """Full artifact_id -> entry mapping. Empty when nothing recorded."""
        return self._load()

    def entry(self, artifact_id: str) -> dict | None:
        return self._load().get(artifact_id)

    # --- writes ------------------------------------------------------------

    def observe(self, artifact_id: str, *, kind: str, node: str, role: str,
                current: dict, now: str | None = None) -> dict:
        """Record a MEASURED identity. `current` is an origins identity dict
        (``version``/``label``/``detail``) already graded by its kind module,
        carrying that grade under ``verification``.

        An unchanged reading refreshes ``verified_at`` but not
        ``observed_at`` and writes no history: "we checked again and it is
        still v0.5.6" is not a transition, and logging it every pass would
        bury the transitions that matter.
        """
        _validate(artifact_id, kind, node, role)
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id) or _blank_entry(artifact_id, kind, node, role)
            before = entry["current"]
            first = before.get("version") is None and before.get("label") is None
            changed = (before.get("version") != current.get("version")
                       or before.get("label") != current.get("label")
                       or before.get("detail") != (current.get("detail") or {}))
            after = {
                "version": current.get("version"),
                "label": current.get("label"),
                "detail": current.get("detail") or {},
                "source": _SOURCE_DERIVED,
                "observed_at": now if changed else (before.get("observed_at") or now),
                "verification": current.get("verification", origins.UNKNOWN),
                "verified_at": now,
            }
            entry["current"] = after
            entry["role"] = role
            data[artifact_id] = entry
            self._save(data)
            if changed:
                self._record(artifact_id, "current", None if first else before,
                             after, "observed", "watcher", now)
            return entry

    def mark_unavailable(self, artifact_id: str, now: str | None = None) -> None:
        """The source could not be reached. RETENTION: the last known
        ``current`` is kept — app.catalog's rule that an unavailable location
        must not make its models vanish, applied to provenance."""
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            if entry is None:
                return
            entry["current"]["verification"] = origins.UNAVAILABLE
            entry["current"]["verified_at"] = now
            data[artifact_id] = entry
            self._save(data)

    def declare_origin(self, artifact_id: str, *, kind: str, node: str, role: str,
                       origin: dict | None, update_path: str | None = None,
                       notes: str | None = None, now: str | None = None) -> dict:
        """An operator asserts where this came from. Never touches
        ``current`` — app.facts' rule that origin is never dropped, in the
        other direction: a declaration must not overwrite a measurement."""
        _validate(artifact_id, kind, node, role)
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id) or _blank_entry(artifact_id, kind, node, role)
            before = entry["origin"]
            entry["origin"] = origin
            entry["role"] = role
            if update_path is not None:
                entry["update_path"] = update_path
            if notes is not None:
                entry["notes"] = notes
            data[artifact_id] = entry
            self._save(data)
            self._record(artifact_id, "origin", before, origin, "declared", "api", now)
            return entry

    def set_desired(self, artifact_id: str, *, version: str | None,
                    label: str | None, set_by: str = "operator",
                    now: str | None = None) -> dict:
        """Record the version this artifact SHOULD be at. Data only — nothing
        converges to it in v1 (D11)."""
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            if entry is None:
                raise KeyError(artifact_id)
            before = entry["desired"]
            after = {"version": version, "label": label,
                     "set_by": set_by, "set_at": now}
            entry["desired"] = after
            data[artifact_id] = entry
            self._save(data)
            self._record(artifact_id, "desired", before, after, "declared", "api", now)
            return entry

    def clear_desired(self, artifact_id: str, now: str | None = None) -> None:
        """'No opinion' — app.intent.forget's semantics for desired version.
        Clearing an absent opinion is a non-action: no write, no history."""
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            if entry is None or entry.get("desired") is None:
                return
            before = entry["desired"]
            entry["desired"] = None
            data[artifact_id] = entry
            self._save(data)
            self._record(artifact_id, "desired", before, None, "declared", "api", now)

    def set_watch(self, artifact_id: str, sources: list[dict],
                  now: str | None = None) -> None:
        """Replace the watch list. Every source is validated first, so a bad
        entry is rejected whole rather than half-written.

        RETENTION IS BOUNDED BY WATCH: any recorded verdict whose source id is
        no longer watched is dropped here, not carried. A verdict for a
        source nobody watches answers a question nobody is asking -- the same
        orphaning the ledger's retention-by-absence rule exists to prevent.

        AN UNCHANGED WATCH LIST IS A NON-ACTION: no write, no history, the
        same treatment clear_desired gives an already-absent desired version.
        Task 9 calls this every collector pass with the same freshly-computed
        sources when nothing upstream has moved; recording a "declared"
        transition on every one of those calls would grow the history file
        forever for a value that never changed.
        """
        for source in sources:
            updates.validate_watch(source)
        sources = list(sources)
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            before = list((entry or {}).get("watch") or [])
            if before == sources:
                return
            if entry is None:
                kind, node, _ = origins.parse_artifact_id(artifact_id)
                entry = _blank_entry(artifact_id, kind, node, "other")
            entry["watch"] = sources

            live = {s["id"] for s in sources}
            update = entry.get("update")
            if update:
                kept = [s for s in update.get("sources", [])
                       if s.get("id") in live]
                entry["update"] = {**update, "sources": kept,
                                   "status": updates.rollup(
                                       [s.get("status") for s in kept])}

            data[artifact_id] = entry
            self._save(data)
            self._record(artifact_id, "watch", before, sources,
                         "declared", "operator", now)

    def record_update(self, artifact_id: str, source_results: list[dict],
                      now: str | None = None) -> str:
        """Merge this pass's per-source results and return the rollup.

        AN UNAVAILABLE RESULT NEVER OVERWRITES A GOOD ONE. A source that could
        not be reached keeps its previous verdict AND its previous
        `checked_at`, with the failure recorded as `stale_note`. A
        stale-but-true answer beats a fresh unknown, and the alternative is
        that one network blip erases what we correctly learned yesterday.

        RETENTION IS BOUNDED BY WATCH here too: a result for a source id no
        longer watched is dropped, never retained. An unknown artifact id is
        a no-op -- it creates nothing and never raises, because nothing about
        update-checking may fail a collector tick.
        """
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            if entry is None:
                return updates.UNAVAILABLE

            live = {s["id"] for s in (entry.get("watch") or [])}
            previous = {s["id"]: s
                        for s in ((entry.get("update") or {}).get("sources") or [])}

            merged = []
            for result in source_results:
                source_id = result.get("id")
                if source_id not in live:
                    continue                      # orphan: never retained
                prior = previous.get(source_id)
                if result.get("status") == updates.UNAVAILABLE and prior and \
                        prior.get("status") != updates.UNAVAILABLE:
                    merged.append({**prior, "stale_note": result.get("note")})
                else:
                    merged.append({**result, "checked_at": now,
                                   "stale_note": None})

            status = updates.rollup([s.get("status") for s in merged])
            before = (entry.get("update") or {}).get("status")
            entry["update"] = {"status": status, "sources": merged,
                               "checked_at": now}
            data[artifact_id] = entry
            self._save(data)
            if before != status:
                self._record(artifact_id, "update.status", before, status,
                             "checked", "update-checker", now)
            return status

    def record_deep_verify(self, artifact_id: str, sha256: str,
                           now: str | None = None) -> dict:
        """The on-demand hash landed. This is the ONLY path that can make a
        weights entry EXACT (design decision D5)."""
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            if entry is None:
                raise KeyError(artifact_id)
            before = entry["current"]
            changed = before.get("version") != sha256
            after = {**before, "version": sha256,
                     "verification": origins.EXACT, "verified_at": now}
            if changed:
                after["observed_at"] = now
            entry["current"] = after
            data[artifact_id] = entry
            self._save(data)
            if changed:
                self._record(artifact_id, "current", before, after,
                             "observed", "api", now)
            return entry

    def delete(self, artifact_id: str) -> None:
        """Explicit removal — the ONLY way an entry leaves the store. History
        is never deleted."""
        with self._lock:
            data = self._load()
            if data.pop(artifact_id, None) is not None:
                self._save(data)


# --- pure read-side helpers ------------------------------------------------

def describe(data: dict, *, now: str, stale_s: float) -> list[dict]:
    """Entries with ``version_drift`` and a read-time ``verification``, in
    key order. Never mutates `data`.

    ABSENCE IS NEVER DRIFT — app.facts.detect_drift's rule, and the reason
    its alarms are still worth reading: with no desired version, or no
    current version to compare, there is nothing to disagree about.

    Only EXACT and CONSISTENT can decay to STALE. UNAVAILABLE outranks it —
    "the node is down" is the actionable fact and collapsing it into "we
    have not looked lately" would hide it.

    ONE PLACE TO READ THE VERIFICATION. The stored ``current.verification``
    is REMOVED from the returned copy and replaced by a top-level
    ``verification`` carrying the read-time value. Leaving both would give a
    consumer two fields with the same name and different answers — a reader
    that happened to pick the nested one would never see STALE at all, which
    is precisely the copy-the-wrong-vocabulary bug this codebase has shipped
    before. The stored file is untouched.
    """
    out = []
    for artifact_id in sorted(data):
        entry = data[artifact_id]
        current = dict(entry.get("current") or {})
        desired = entry.get("desired")
        drift = bool(desired and current.get("version") and desired.get("version")
                     and current["version"] != desired["version"])
        verification = current.pop("verification", origins.UNKNOWN)
        if verification in (origins.EXACT, origins.CONSISTENT) and _is_stale(
                current.get("verified_at"), now, stale_s):
            verification = origins.STALE
        out.append({**entry, "current": current,
                    "version_drift": drift, "verification": verification,
                    # Passed through unmodified -- but defaulted here so an
                    # entry written before watch/update existed (live on two
                    # machines) reads back the same shape as one written
                    # after, rather than simply omitting the keys.
                    "watch": entry.get("watch") or [],
                    "update": entry.get("update")})
    return out


def gaps(data: dict) -> list[str]:
    """Artifact ids with no recorded origin — the work queue for what D8
    refuses to guess."""
    return sorted(k for k, v in data.items() if (v or {}).get("origin") is None)


def updates_available(data: dict) -> list[str]:
    """Artifact ids whose rollup says a newer version exists."""
    return sorted(k for k, v in data.items()
                  if ((v or {}).get("update") or {}).get("status") == updates.AVAILABLE)


def _is_stale(verified_at: str | None, now: str, stale_s: float) -> bool:
    if not verified_at:
        return True
    try:
        age = (datetime.fromisoformat(now)
               - datetime.fromisoformat(verified_at)).total_seconds()
    except (TypeError, ValueError):
        return False   # an unparseable stamp is its own problem, not staleness
    return age > stale_s
