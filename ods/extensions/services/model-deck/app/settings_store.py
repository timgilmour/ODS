"""Model Deck settings store — what things are launched and served with.

Characteristics describe what things ARE; settings describe what they are
launched WITH. Both resolve over machine-derived truth, which is why they
are siblings rather than one store: declared.json asserts facts,
settings.json asserts configuration.

Three scopes, most-specific winning at resolution time (see app.ladder):

* ``engines``       — keyed ``<node>/<engine>``. Node-keyed because the same
  engine on two nodes is two different things.
* ``models``        — keyed by model identity (the checkpoint directory
  name). Node-neutral: it follows the model wherever it runs.
* ``engine_models`` — keyed ``<node>/<engine>|<model>``. The most specific.

Three namespaces, because a real profile sets far more than flags:

* ``args``      — engine argv
* ``env``       — environment (VLLM_USE_FLASHINFER_SAMPLER=1, COMFYUI_PORT)
* ``container`` — a small ALLOWLIST: image, shm_size, ulimits.

Volumes and networking are deliberately absent from the allowlist. The model
mount IS the placement; letting a human edit it here would let placement and
settings disagree about which model is running — precisely the class of bug
this whole layer exists to remove.

``notes`` carries the human rationale that lives in compose comments today
("no --quantization flag: forcing modelopt breaks the load"). Adoption
imports those comments here instead of discarding them, which is one of the
reasons the Deck ships settings documents rather than regenerating compose
files.

Every ``put()`` stamps the touched namespace with an ``updated_ts`` — a
dict ``{namespace: iso}`` on the scope entry, one clock per namespace, not
one per entry and not one per key. PER-NAMESPACE, not entry-level, as of the
Task 7 review round 2026-08-07: an entry-level clock made a settings_drift
"changed" list report keys from namespaces that were never written,
whenever ANY namespace of the same entry was touched later. Task 7's
settings-drift flag compares each namespace's stamp against a placement's
``intent.updated_ts`` (NOT ``last_healthy_ts`` — the latter is re-stamped
on every serving reconcile tick by ``app.arbiter``'s ``note_healthy``,
which would make the flag self-erase within one tick of a placement
actually serving) to say "this namespace was written since the placement's
settings were last (re)recorded" — a display flag only; it never feeds
app.reconcile.

Per-key change history is now kept in the ``journal`` dict: ``journal[namespace]``
is a list of change records ``{"key": name, "old": prev, "new": current, "ts": iso}``,
capped at 50 entries per namespace. Only changes that actually modified a key's value
are journaled (unchanged puts do not append). A ``remove`` list on ``put()`` deletes
keys from a namespace while journaling their removal as ``"new": None``. The journal
is consulted by ``_settings_drift`` (Task 2) to report which keys changed, and by the
UI (Task 5) to understand a removal was requested.

Human/UI-owned. Missing/corrupt reads as empty; writes are atomic; a
rejected put leaves the file untouched. Self-healing is recursive: a
corrupt top-level kind, a corrupt scope entry, or a corrupt namespace or
``notes`` field within an entry each independently reset to ``{}`` rather
than raising or leaking a non-dict value out through ``get()``/``scope()``
— the same posture app.policy applies per-kind, carried one and two levels
deeper. ``notes`` is healed alongside args/env/container rather than left
as an exception: ``put(note=...)`` subscript-assigns into it exactly the
way a put into a namespace does, so a corrupt ``notes`` field is the same
crash vector, not a narrower one.

This is single-process, in-process state only: the Deck runs uvicorn with
its default single worker (see Dockerfile CMD — no ``--workers`` flag), so
there is no cross-process coordination to do. Within that one process,
though, both a router handler and the background watcher thread can reach
this store, and load-modify-save is not atomic across two threads without
help — so mutations are serialized with an internal lock, matching
app.intent's post-2026-08-07 discipline. Validation and ``args``
normalization run *before* the lock is taken (established convention —
app.intent.record does the same with its state check): both are pure
functions of the call's own arguments, not of the file, so there is
nothing there for the lock to protect.

``args`` values are normalized on write to what app.argline's render/parse
round trip actually produces (RULING 2026-08-07, see app.argline module
docstring): a singleton list collapses to its scalar, and an int/float
scalar becomes a string. Without this, a value read back here could
disagree with the same value read back through
``parse_argline(render_argline(...))`` on Python type alone — the raw-==
trap that ruling warns callers off. An empty list has no argline
representation at all — RULING 2026-08-07 (review), overturning an earlier
True-normalization: render_argline({"k": []}) and render_argline({"k":
True}) emit byte-identical argv, so round-trip congruence with app.argline
never distinguished the two, and a bare flag is not what an operator meant
by an empty list. This store's posture is warn-and-DROP the key instead,
matching app.intent's idiom that deleting a key is the only way to say "no
opinion" — the container allowlist below remains the one hard validation
failure in this module; everything else about a value's shape is fixed up
(or, for an empty list, removed with a warning) rather than blocking the
save. A put() whose values are entirely dropped this way still merges and
writes normally: an empty dict merged into an existing namespace is a
no-op on its other keys, not a special case. ``env``/``container`` values
are not argv and are stored as given.

The implementation of this normalization has moved: it is now
``app.argline.normalize_args_map`` (hoisted, review 2026-08-07). A review
of app.ladder found that this store was the ONLY place either axis got
enforced, so a layer that never passes through here — a derived layer such
as checkpoint recommendations parsed straight from generation_config.json
— could hand the ladder a raw int, a shape Tasks 1/2 ruled impossible.
This store now normalizes on write purely by delegating to that function;
its own behavior, including the warn-and-drop posture above, is unchanged.
"""

import threading
from datetime import UTC, datetime
from pathlib import Path

from app.argline import normalize_args_map
from app.store_io import load_json, save_json

KINDS = ("engines", "models", "engine_models")
NAMESPACES = ("args", "env", "container")
_JOURNAL_CAP = 50

# Container-level settings a human may edit. Everything else about the
# container is Deck-managed — see the module docstring.
CONTAINER_ALLOWLIST = ("image", "shm_size", "ulimits")


def empty_store() -> dict:
    """One empty settings-store document — shared with app.sets so a future
    change to the empty shape (e.g. a seeded schema-version field) lands in
    both the live store and set snapshots at once."""
    return {kind: {} for kind in KINDS}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SettingsStore:
    """Scoped configuration, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        data = load_json(self._path)
        if not isinstance(data, dict):
            return empty_store()
        merged = empty_store()
        for kind in KINDS:
            value = data.get(kind)
            if not isinstance(value, dict):
                continue
            merged[kind] = {key: self._heal_entry(entry) for key, entry in value.items()}
        return merged

    @staticmethod
    def _heal_entry(entry) -> dict:
        """A scope entry that is not a dict resets to {} (one level below
        the per-kind reset above). Within a dict entry, args/env/container
        and notes are each individually guarded the same way, so one
        corrupt field doesn't take a healthy sibling down with it — notes
        is included because put(note=...) subscript-assigns into it
        (`entry.setdefault("notes", {})[namespace] = note`), which raises
        an uncaught TypeError against any corrupt present value exactly
        like the namespaces did before this guard existed. ``updated_ts``
        gets the same guard for the same reason (put() subscript-assigns
        into it per namespace) — also covers a pre-migration entry that
        still carries the old entry-level string form (Task 7 review round
        2026-08-07), which would otherwise crash ``_settings_drift``'s
        ``.get(namespace)`` on a str. ``journal`` is guarded two levels deep:
        at the top level, a non-dict journal resets to {}; within a dict
        journal, any per-namespace value that is not a list resets to []
        since put() calls log.extend(changes) on journal[namespace] — any
        corrupt present value would crash with AttributeError on str.extend().
        Any other key is out of this scope and passes through untouched."""
        if not isinstance(entry, dict):
            return {}
        healed = dict(entry)
        for namespace in NAMESPACES:
            if namespace in healed and not isinstance(healed[namespace], dict):
                healed[namespace] = {}
        if "notes" in healed and not isinstance(healed["notes"], dict):
            healed["notes"] = {}
        if "updated_ts" in healed and not isinstance(healed["updated_ts"], dict):
            healed["updated_ts"] = {}
        if "journal" in healed:
            if not isinstance(healed["journal"], dict):
                healed["journal"] = {}
            else:
                # Two-level healing: within a dict journal, each namespace
                # value must be a list, else reset to [].
                healed_journal = dict(healed["journal"])
                for ns_name in NAMESPACES:
                    if ns_name in healed_journal and not isinstance(healed_journal[ns_name], list):
                        healed_journal[ns_name] = []
                healed["journal"] = healed_journal
        return healed

    def _save(self, data: dict) -> None:
        save_json(self._path, data, indent=1, sort_keys=True)

    def get(self) -> dict:
        return self._load()

    def scope(self, kind: str, key: str) -> dict:
        return self._load().get(kind, {}).get(key, {})

    def put(self, kind: str, key: str, namespace: str, values: dict,
            note: str | None = None, remove: list | None = None) -> None:
        """Merge `values` into one namespace of one scope entry, optionally
        removing keys. Per-key merge, never per-blob: setting one flag must
        not discard the others. Optionally delete keys from the namespace
        while journaling the removal. Validates before writing anything.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown scope kind {kind!r}; expected one of {KINDS}")
        if namespace not in NAMESPACES:
            raise ValueError(f"unknown namespace {namespace!r}; expected one of {NAMESPACES}")
        if namespace == "container":
            extra = set(values) - set(CONTAINER_ALLOWLIST)
            if extra:
                raise ValueError(
                    f"container settings {sorted(extra)} are Deck-managed and not editable; "
                    f"editable: {list(CONTAINER_ALLOWLIST)}"
                )
        if namespace == "args":
            # Delegates to app.argline — see module docstring, "The
            # implementation of this normalization has moved". The warning
            # for a dropped empty-list key is emitted from inside there.
            values = normalize_args_map(values)

        if remove:
            overlap = set(remove) & set(values)
            if overlap:
                raise ValueError(
                    f"keys {sorted(overlap)} appear in both 'values' and 'remove'; "
                    "a put must say one thing per key")

        with self._lock:
            data = self._load()
            entry = data[kind].setdefault(key, {})
            ns = entry.setdefault(namespace, {})
            before = dict(ns)
            ns.update(values)
            for name in remove or []:
                ns.pop(name, None)
            now = _now_iso()
            changes = [
                {"key": name, "old": before.get(name), "new": ns.get(name), "ts": now}
                for name in sorted(set(values) | set(remove or []))
                if before.get(name) != ns.get(name)
            ]
            if changes:
                log = entry.setdefault("journal", {}).setdefault(namespace, [])
                log.extend(changes)
                entry["journal"][namespace] = log[-_JOURNAL_CAP:]
            if note is not None:
                entry.setdefault("notes", {})[namespace] = note
            # PER-NAMESPACE write timestamp (Task 7 review round,
            # 2026-08-07 — an earlier entry-level version made
            # settings_drift's "changed" list report namespaces that were
            # never written whenever any OTHER namespace of the same entry
            # was touched later). Per-key history is now kept in the journal
            # for Task 2's drift detection to fold changes; updated_ts remains
            # per-namespace for timestamp compatibility. See app.routers._settings_drift.
            entry.setdefault("updated_ts", {})[namespace] = now
            self._save(data)

    def restore(self, data: dict) -> None:
        """Bulk-REPLACE the entire store from ``data`` — a previously
        captured ``get()`` snapshot (e.g. a config set's ``settings_snapshot``,
        app.sets Task 9). Used by app.sets' ``restore_settings`` apply step
        and nowhere else; a real write, not a merge of the settings themselves
        (but a MERGE of the journals).

        ``data`` gets the same healing a corrupt file read gets (``_load``'s
        per-kind/per-entry guards) — a snapshot is exactly as untrusted as a
        file on disk: it rode through a pydantic JSON round-trip and, for the
        auto-captured ``_previous`` set, may be older than the newest
        healing rule. Every ``args`` namespace is re-normalized through
        ``normalize_args_map``, the same pass ``put()`` applies on write, so
        a snapshot captured before a normalization rule changed doesn't
        persist a stale shape. Every namespace actually present in an entry
        is stamped with a FRESH ``updated_ts`` — a restore is a real write,
        so ``settings_drift`` may honestly flag it, even though nothing here
        reloads the running engine (reload stays a human's call). ``notes``
        is carried through unchanged (human rationale, not a write-clock).
        Each namespace's journal is MERGED with its previous state, not
        replaced: the restore's per-key diff entries are appended to the
        previous journal for that namespace, then capped at _JOURNAL_CAP. This
        allows the next task's drift detection to correctly fold entries since
        a baseline. Untouched namespaces of a surviving entry keep their prior
        journal intact. Entries removed wholesale by the restore have no
        surviving scope entry to carry a journal; a vanished scope has no
        chip to show drift on.
        """
        if not isinstance(data, dict):
            data = {}
        healed = empty_store()
        for kind in KINDS:
            value = data.get(kind)
            if not isinstance(value, dict):
                continue
            healed[kind] = {key: self._heal_entry(entry) for key, entry in value.items()}

        now = _now_iso()
        for kind in KINDS:
            for entry in healed[kind].values():
                if "args" in entry:
                    # heal=True: this is a repair path, not a wire. A
                    # snapshot holding a list-null must be CLEANED, not
                    # refused — refusing aborts the restore_settings apply
                    # step, breaking undo for exactly the legacy data the
                    # renderer tolerates [T9-fix re-review].
                    entry["args"] = normalize_args_map(entry["args"], heal=True)
                stamped = {ns: now for ns in NAMESPACES if ns in entry}
                if stamped:
                    entry["updated_ts"] = stamped

        with self._lock:
            previous = self._load()
            for kind in KINDS:
                for key, entry in healed[kind].items():
                    prev_entry = previous.get(kind, {}).get(key, {})
                    # Start with the previous entry's healed journal, or an
                    # empty dict if there is no previous entry.
                    prev_journal = prev_entry.get("journal", {})
                    if not isinstance(prev_journal, dict):
                        prev_journal = {}
                    healed_prev_journal = {}
                    for ns_name in NAMESPACES:
                        if ns_name in prev_journal and isinstance(prev_journal[ns_name], list):
                            healed_prev_journal[ns_name] = list(prev_journal[ns_name])
                    journal = dict(healed_prev_journal)
                    # Append new changes for each namespace, keeping untouched
                    # namespaces' prior journal intact.
                    for ns_name in NAMESPACES:
                        before = prev_entry.get(ns_name, {}) or {}
                        after = entry.get(ns_name, {}) or {}
                        changes = [
                            {"key": name, "old": before.get(name),
                             "new": after.get(name), "ts": now}
                            for name in sorted(set(before) | set(after))
                            if before.get(name) != after.get(name)
                        ]
                        if changes:
                            if ns_name not in journal:
                                journal[ns_name] = []
                            journal[ns_name].extend(changes)
                            journal[ns_name] = journal[ns_name][-_JOURNAL_CAP:]
                    if journal:
                        entry["journal"] = journal
            self._save(healed)

    def forget(self, kind: str, key: str) -> None:
        with self._lock:
            data = self._load()
            if data.get(kind, {}).pop(key, None) is not None:
                self._save(data)
