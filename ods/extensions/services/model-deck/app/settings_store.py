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
"""

import json
import os
import threading
import warnings
from pathlib import Path

KINDS = ("engines", "models", "engine_models")
NAMESPACES = ("args", "env", "container")

# Container-level settings a human may edit. Everything else about the
# container is Deck-managed — see the module docstring.
CONTAINER_ALLOWLIST = ("image", "shm_size", "ulimits")


def _empty() -> dict:
    return {kind: {} for kind in KINDS}


# Sentinel: this args value carries no argline opinion and must be dropped
# from the key, not stored — see _normalize_args_value and put().
_DROP = object()


def _normalize_args_scalar(value):
    """One value of the two RULING 2026-08-07 axes: int/float becomes str.
    bool is an int subclass in Python, so it is excluded explicitly —
    ``True`` is the bare-flag sentinel, not a number. Anything else
    (str, None, dict, ...) is outside the documented axes and passes
    through unchanged rather than guessed at."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _normalize_args_value(value):
    """Normalize one `args` value to argline's post-round-trip shape, or
    return _DROP if the value carries no argline opinion at all.

    A list of one collapses to its (normalized) element — the other RULING
    2026-08-07 axis. A list of zero is _DROP: RULING 2026-08-07 (review),
    see module docstring — it renders byte-identical to no value at all,
    so keeping it as a stored key would claim an opinion the operator
    never expressed.
    """
    if isinstance(value, list):
        if not value:
            return _DROP
        if len(value) == 1:
            return _normalize_args_scalar(value[0])
        return [_normalize_args_scalar(v) for v in value]
    return _normalize_args_scalar(value)


class SettingsStore:
    """Scoped configuration, persisted to `path`."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return _empty()
        if not isinstance(data, dict):
            return _empty()
        merged = _empty()
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
        like the namespaces did before this guard existed. Any other key
        is out of this scope and passes through untouched."""
        if not isinstance(entry, dict):
            return {}
        healed = dict(entry)
        for namespace in NAMESPACES:
            if namespace in healed and not isinstance(healed[namespace], dict):
                healed[namespace] = {}
        if "notes" in healed and not isinstance(healed["notes"], dict):
            healed["notes"] = {}
        return healed

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp_path, self._path)

    def get(self) -> dict:
        return self._load()

    def scope(self, kind: str, key: str) -> dict:
        return self._load().get(kind, {}).get(key, {})

    def put(self, kind: str, key: str, namespace: str, values: dict,
            note: str | None = None) -> None:
        """Merge `values` into one namespace of one scope entry.

        Per-key merge, never per-blob: setting one flag must not discard the
        others. Validates before writing anything.
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
            normalized = {}
            for k, v in values.items():
                n = _normalize_args_value(v)
                if n is _DROP:
                    warnings.warn(
                        f"settings_store: empty list for args key {k!r} carries "
                        "no argline value (RULING 2026-08-07 review); dropped, "
                        "not stored",
                        stacklevel=2,
                    )
                    continue
                normalized[k] = n
            values = normalized

        with self._lock:
            data = self._load()
            entry = data[kind].setdefault(key, {})
            entry.setdefault(namespace, {}).update(values)
            if note is not None:
                entry.setdefault("notes", {})[namespace] = note
            self._save(data)

    def forget(self, kind: str, key: str) -> None:
        with self._lock:
            data = self._load()
            if data.get(kind, {}).pop(key, None) is not None:
                self._save(data)
