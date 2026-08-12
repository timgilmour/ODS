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

EVERYTHING BELOW THE TOP LEVEL IS UNTRUSTED INPUT, AND IS GATED ONCE, AT
``_load()``. ``provenance.json`` is hand-editable and the quarantine above
only proves the document is a mapping — every entry inside it, every field
inside an entry, and every element inside those fields is whatever a text
editor left behind. Four review rounds each generalised one more level of
that (a missing key, a non-dict element, a non-list container, a whole entry
that is a bare string) and each time the level above it was still open. So
the tolerance is no longer a thing each read remembers: ``_load()`` — the
one door stored data comes through — runs every entry through
``_stored_entry``, and the three pure read helpers gate the document they are
handed the same way. Downstream code may then read ``entry["watch"]`` and
``entry["current"]`` as the types ``_blank_entry`` declares, because nothing
else can reach it. The gate normalises SHAPE only: a value whose type is
unreadable becomes the "not recorded" value of its declared type, and every
other value — including keys from a future version — is passed through.

Provenance NEVER actuates (D4/D11). It records desired state as data and
reports drift; app.reconcile stays the only restorer, and convergence — which
would need `docker pull` + container create, or a compose write on sparky —
is a later spec gated on a socket-proxy permission decision.
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from app import origins, provenance_history, updates
from app.store_io import save_json

_log = logging.getLogger(__name__)

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


# --- the gate: stored shapes, normalised once ------------------------------
#
# These are the primitives. `_stored_entry` composes them into the whole
# entry, `_load()` applies it to the whole document, and nothing downstream
# repeats them -- see the module docstring for why the tolerance lives here
# and not at each read.

def _stored_list(value) -> list[dict]:
    """A stored list of objects: a non-list becomes `[]`, and elements that
    are not objects are dropped."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _stored_dict(value) -> dict:
    """A stored object that must exist: `None`, a scalar, or a list all
    become `{}` instead of raising on the first `.get()` against them."""
    return value if isinstance(value, dict) else {}


def _stored_object(value) -> dict | None:
    """A stored object that may legitimately be absent (`origin`,
    `desired`, `update`). An unreadable shape reads as "not recorded",
    which is the honest answer AND the one that puts the artifact back on
    the operator's work queue: `gaps()` lists it, so the fix is visible
    rather than silently papered over with an empty object."""
    return value if isinstance(value, dict) else None


# JSON types that can key a dict or join a set. A source id outside these is
# not an id, whatever a text editor put there.
_ID_TYPES = (str, int, float)


def _keyed_sources(value) -> list[dict]:
    """A stored source list narrowed to the elements that can be KEYED BY
    ID — `record_update` builds a set and a dict out of these ids, and an
    id that is itself a list or an object is unhashable (`TypeError` from
    the set comprehension, not a `.get()` away), while an id-less element
    read as `None` would spuriously match an incoming result that also has
    no id.

    DECLARED AND DERIVED ARE NOT GATED THE SAME WAY, and the difference is
    the same one `_SOURCE_DECLARED`/`_SOURCE_DERIVED` draws one field up.
    `update["sources"]` is DERIVED — a verdict map the next check pass
    rewrites — so an unkeyable verdict is dropped at the gate and nothing is
    lost. `watch` is DECLARED: an operator wrote it and this is its only
    home, so a source whose id merely cannot be USED as a key is kept
    visible rather than deleted, and only the places that turn a watch
    source into a key narrow it through this function.

    THAT PROTECTION IS PARTIAL, AND KNOWINGLY SO. `_ENTRY_SHAPE["watch"]` is
    `_stored_list`, which already drops any element that is not an object at
    all — and since every write re-serialises the whole document, such an
    element IS erased from disk on the next unrelated save. So the rule this
    function implements is not "declared data is never dropped"; it is "an
    element that could still be shown to an operator is not dropped merely
    for being unkeyable". A bare string where a source belongs carries no
    id, no check and no pin: there is nothing to display and nothing to fix.
    Accepted deferral (final review), written down here so this docstring
    stops arguing for a guarantee the level below does not make.
    """
    return [s for s in _stored_list(value)
            if isinstance(s.get("id"), _ID_TYPES) and s.get("id")]


def _stored_current(value) -> dict:
    """`current` is the one field this module INDEXES rather than `.get()`s
    (`entry["current"]["verification"] = ...`), so it is completed against
    `_empty_current()` rather than merely type-checked."""
    return {**_empty_current(), **_stored_dict(value)}


def _stored_update(value) -> dict | None:
    """`update` carries a nested container of its own. `sources` is forced
    to exist so that a caller may write `update["sources"]`; `status` and
    `checked_at` are scalars and stay `.get()`-able absences."""
    update = _stored_object(value)
    if update is None:
        return None
    return {**update, "sources": _keyed_sources(update.get("sources"))}


# Every field of an entry whose TYPE this module depends on, and the gate
# that guarantees it. Keyed by `_blank_entry`'s own vocabulary: a new field
# with a container default belongs here the day it is added, and
# tests/test_provenance.py asserts exactly that.
_ENTRY_SHAPE = {
    "current": _stored_current,
    "origin": _stored_object,
    "desired": _stored_object,
    "watch": _stored_list,
    "update": _stored_update,
}


def _identity_from(artifact_id: str) -> tuple[str | None, str | None]:
    """`kind`/`node` as the id itself declares them, or a pair of `None`
    when the key is not a well-formed id — which a hand-edited file can
    also produce."""
    try:
        kind, node, _ = origins.parse_artifact_id(artifact_id)
    except origins.BadArtifactId:
        return None, None
    return kind, node


def _stored_entry(artifact_id: str, value) -> dict:
    """One entry read back from disk, as the shape `_blank_entry` declares.

    THE ENTRY IS UNTRUSTED TOO -- that is the level three earlier rounds of
    this fix each stopped one short of. A hand-edited file can put a bare
    string where an entry belongs, and `(entry or {}).get(...)` does not
    save you: a truthy non-dict passes the `or {}` untouched and `.get` on a
    string still raises. So the entry is rebuilt on top of a blank one:
    every declared key exists, every container has its declared type, and
    every readable stored value wins over the blank default.

    THE KEY IS THE IDENTITY. `artifact_id`, and the `kind`/`node` derivable
    from it, come from the mapping key -- that is not a guess (D8), it is
    the same derivation `_validate` enforces on every write. All three are
    FORCED over whatever was stored, not merely defaulted: `_validate`
    guarantees the three agree on every write path (below), so a stored pair
    that disagrees can only be a hand edit, and leaving it in place would let
    one entry be served under one key while announcing another id/kind/node
    through /api/provenance and /api/state.

    Forcing `kind`/`node` is conditional on the key actually declaring them,
    because you can only disagree with an id that exists: when the key is not
    a well-formed artifact id, `_identity_from` derives nothing and a stored
    pair is left alone rather than blanked. The gate normalises shape; it
    does not destroy a readable value it has nothing to replace with.

    `role` is not derivable and is never invented; a corrupt entry gets
    `_blank_entry`'s "other", the same unknown `set_watch` uses.

    NORMALISING COSTS THE CORRUPT VALUE ITS PLACE ON DISK: the next write of
    any artifact re-serialises the whole document, so a field that could not
    be read is not preserved. That is the deliberate boundary between this
    and D13's quarantine -- quarantine protects a document that cannot be
    read AT ALL, where refusing to guess is the only safe move; a single
    unreadable field inside a readable document has no content to protect
    (a scalar where an object belongs holds no origin, no version and no
    watch list), and leaving it in place would keep the entry unreadable
    forever. The entry itself is never dropped: `delete()` stays the only
    way one leaves the store.
    """
    kind, node = _identity_from(artifact_id)
    entry = {**_blank_entry(artifact_id, kind, node, "other"),
             **_stored_dict(value), "artifact_id": artifact_id}
    if kind is not None:
        entry["kind"], entry["node"] = kind, node
    for field, gate in _ENTRY_SHAPE.items():
        entry[field] = gate(entry[field])
    return entry


def _stored_document(data) -> dict[str, dict]:
    """The whole `{artifact_id: entry}` mapping, gated. Non-string keys
    cannot survive a JSON round trip and would break `sorted(data)` against
    a mixed document, so they are dropped here rather than at each reader."""
    if not isinstance(data, dict):
        return {}
    return {artifact_id: _stored_entry(artifact_id, entry)
            for artifact_id, entry in data.items()
            if isinstance(artifact_id, str)}


def _by_id(sources) -> list[dict]:
    """A stable, order-independent form of a source list for equality
    comparison. Safe on ANY input -- a raw disk read or an already-validated
    `sources` argument -- so the unchanged-watch guard cannot be the one
    place that crashes on a shape everything else tolerates."""
    return sorted(_stored_list(sources), key=lambda s: str(s.get("id") or ""))


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
        """THE ONLY DOOR STORED DATA COMES THROUGH, and therefore the only
        place that has to know the file is hand-editable. An unreadable
        DOCUMENT is quarantined (D13); a readable document with unreadable
        contents is gated by `_stored_document`, so every method below --
        including ones nobody has written yet -- reads an entry whose fields
        have the types `_blank_entry` declares. No method may read the file
        itself; tests/test_provenance.py asserts that."""
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
        return _stored_document(data)

    def _quarantine(self) -> None:
        """Preserve an unreadable document before starting empty (D13)."""
        stamp = _now_iso().replace(":", "").replace("-", "")
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        try:
            os.replace(self._path, target)
        except OSError as exc:
            # Loud, not raised: raising would turn corrupt-file handling into
            # a crash loop (exactly what this quarantine exists to avoid),
            # but a silent pass defeats the preservation it implements —
            # the operator would never learn the document was unreadable
            # [max-review c5]. The store has no events_path this deep, so the
            # module logger carries it. The corrupt file stays where it is;
            # the next _load() retries the quarantine.
            _log.warning("could not quarantine corrupt %s: %s", self._path, exc)

    def _save(self, data: dict[str, dict]) -> None:
        save_json(self._path, data, indent=1)

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
        saved = self.observe_all(
            [{"artifact_id": artifact_id, "kind": kind, "node": node,
              "role": role, "current": current}], now=now)
        return saved[artifact_id]

    def observe_all(self, readings: list[dict],
                    now: str | None = None) -> dict[str, dict]:
        """Batch ``observe()``: every reading lands under ONE lock/load/save
        instead of one full file round trip per artifact — the collector
        pass calls this once per source sweep [max-review c15]. Each reading
        is ``observe()``'s arguments as a dict (``artifact_id``/``kind``/
        ``node``/``role``/``current``); history is still recorded per
        CHANGED artifact, unchanged readings still write none.

        Every reading is validated before any is applied, so one bad entry
        refuses the whole batch rather than persisting half of it — the
        same posture ``set_watch`` takes with its sources.

        Returns the saved document (a snapshot; the store re-loads on every
        read), so a caller seeding watches from what it just observed can
        consult the stored entries without a per-artifact re-load.
        """
        for reading in readings:
            _validate(reading["artifact_id"], reading["kind"],
                      reading["node"], reading["role"])
        now = now or _now_iso()
        if not readings:
            return self._load()
        with self._lock:
            data = self._load()
            transitions = []
            for reading in readings:
                artifact_id, current = reading["artifact_id"], reading["current"]
                entry = (data.get(artifact_id)
                         or _blank_entry(artifact_id, reading["kind"],
                                         reading["node"], reading["role"]))
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
                entry["role"] = reading["role"]
                data[artifact_id] = entry
                if changed:
                    transitions.append(
                        (artifact_id, None if first else before, after))
            self._save(data)
            for artifact_id, before, after in transitions:
                self._record(artifact_id, "current", before, after,
                             "observed", "watcher", now)
            return data

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
        updates.validate_watch_sources(sources)
        sources = list(sources)
        now = now or _now_iso()
        with self._lock:
            data = self._load()
            entry = data.get(artifact_id)
            before = entry["watch"] if entry is not None else []
            if _by_id(before) == _by_id(sources):
                return
            if entry is None:
                kind, node, _ = origins.parse_artifact_id(artifact_id)
                entry = _blank_entry(artifact_id, kind, node, "other")
            entry["watch"] = sources

            live = {s["id"] for s in sources}
            update = entry["update"]
            if update:
                kept = [s for s in update["sources"] if s["id"] in live]
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

            # Declared types, no per-site type guards: `_load`'s gate has
            # settled the shape of `watch` and of `update["sources"]`
            # already. `_keyed_sources` here is the DECLARED half of that
            # split (see its docstring) -- the stored watch list keeps what
            # the operator wrote, and this is the one place that turns it
            # into keys, so it is the one place that must narrow it to ids
            # that can be keys. An id-less source reaching here as `None`
            # would spuriously match an incoming result that also lacks one.
            live = {s["id"] for s in _keyed_sources(entry["watch"])}
            stored_update = entry["update"] or {}
            previous = {s["id"]: s for s in stored_update.get("sources", [])}

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
            before = stored_update.get("status")
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

    `data` IS UNTRUSTED. In production it arrives from `ProvenanceStore.get()`
    and is already gated, but this is a public pure function reachable with
    any mapping, so it gates its own argument rather than inheriting a
    guarantee from whoever called it. Same for `gaps` and
    `updates_available` — the three of them are the module's read-side
    boundary, exactly as `_load()` is the store's.
    """
    data = _stored_document(data)
    out = []
    for artifact_id in sorted(data):
        entry = data[artifact_id]
        current = dict(entry["current"])
        desired = entry["desired"]
        drift = bool(desired and current.get("version") and desired.get("version")
                     and current["version"] != desired["version"])
        verification = current.pop("verification", origins.UNKNOWN)
        if verification in (origins.EXACT, origins.CONSISTENT) and _is_stale(
                current.get("verified_at"), now, stale_s):
            verification = origins.STALE
        # `watch`/`update` are passed through with no transformation of their
        # own -- the gate above is the only thing that has touched them.
        out.append({**entry, "current": current,
                    "version_drift": drift, "verification": verification})
    return out


def gaps(data: dict) -> list[str]:
    """Artifact ids with no recorded origin — the work queue for what D8
    refuses to guess. An origin that cannot be read is no origin: a corrupt
    entry belongs on the work queue, not silently off it."""
    return sorted(k for k, v in _stored_document(data).items()
                  if v["origin"] is None)


def updates_available(data: dict) -> list[str]:
    """Artifact ids whose rollup says a newer version exists."""
    return sorted(k for k, v in _stored_document(data).items()
                  if (v["update"] or {}).get("status") == updates.AVAILABLE)


def _is_stale(verified_at: str | None, now: str, stale_s: float) -> bool:
    if not verified_at:
        return True
    try:
        age = (datetime.fromisoformat(now)
               - datetime.fromisoformat(verified_at)).total_seconds()
    except (TypeError, ValueError):
        return False   # an unparseable stamp is its own problem, not staleness
    return age > stale_s
