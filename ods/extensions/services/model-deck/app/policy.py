"""Model Deck resource policy store — declaration-driven defaults.

E1 Task 4: Policy defaults now come from the node's declared engines[] (each
engine carries policy_defaults). Tracks arbitration policy for each declared
resource: eviction priority, pinned status (exempt from eviction), and idle-TTL
before Model Deck parks it.

``policy.json`` is a flat mapping of ``{resource: {priority, pinned,
idle_ttl}}``, persisted next to no other state — this module owns the whole
file. Writes are atomic (temp file + ``os.replace``) since the supervisor
may crash mid-write; a missing or corrupt file is treated as absent rather
than raised, and self-heals by materializing and persisting declared defaults
on the next ``get()``.

Malformed-record gating lives HERE, at ``_load()``, and nowhere else
(NodeStore's pattern, app/node_store.py:18-30): a parseable file that is
missing or malformed for a declared resource self-heals the same way as the
whole-file case, one level down. ``_load()`` runs every record through
``_gated()`` and persists the heal immediately if anything changed, so
``get()``/``put()``/``set_auto()`` all consume an already-guaranteed shape
and gate nothing themselves. A missing or malformed declared resource is
replaced by its declared default. Stored rows for undeclared resources (orphaned
rows from hand-edits or older writes) are kept on disk but invisible on read
— they survive re-declaration. A malformed ``_auto`` record (wrong shape, or
a non-bool ``enabled``) is dropped rather than kept, healing to
``auto_enabled()``'s default-True reading instead of crashing the tick.

This is single-process, in-process state only — no CROSS-process locking.
Within the process both stores below DO hold a threading.Lock across every
load-modify-save (T9b): the watcher tick and the HTTP threadpool are real
concurrent writers, and since the boundary gate `_load()` itself persists a
heal, even two concurrent READS are two writers on a partial file. The
supervisor is the sole owner of policy.json.
"""

import threading
from collections.abc import Callable
from pathlib import Path
from app.store_io import load_json, save_json

TenantPolicy = dict[str, int | bool]

_FIELDS = {"priority": int, "pinned": bool, "idle_ttl": int}

# Reserved non-tenant key inside policy.json holding lifecycle automation
# config. Filtered out of get() so callers iterating tenants never see it.
_AUTO_KEY = "_auto"


def _validate_policy(tenant: str, policy: dict) -> None:
    """Raise ValueError if `policy` doesn't have exactly priority/pinned/idle_ttl
    with correct types (bool is not an int for priority/idle_ttl purposes)."""
    extra = set(policy) - set(_FIELDS)
    if extra:
        raise ValueError(f"unknown field(s) for tenant {tenant!r}: {sorted(extra)}")
    missing = set(_FIELDS) - set(policy)
    if missing:
        raise ValueError(f"missing field(s) for tenant {tenant!r}: {sorted(missing)}")

    priority = policy["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError(f"tenant {tenant!r}: priority must be an int, got {priority!r}")

    pinned = policy["pinned"]
    if not isinstance(pinned, bool):
        raise ValueError(f"tenant {tenant!r}: pinned must be a bool, got {pinned!r}")

    idle_ttl = policy["idle_ttl"]
    if isinstance(idle_ttl, bool) or not isinstance(idle_ttl, int):
        raise ValueError(f"tenant {tenant!r}: idle_ttl must be an int, got {idle_ttl!r}")
    if idle_ttl < 0:
        raise ValueError(f"tenant {tenant!r}: idle_ttl must be >= 0, got {idle_ttl!r}")


class PolicyStore:
    """Per-tenant arbitration policy, persisted to `path`."""

    def __init__(self, path: Path, declared_defaults: Callable[[], dict[str, TenantPolicy]] | None = None):
        self._path = path
        # Callable returning declared resource defaults {resource: {priority, pinned, idle_ttl}}.
        # None = no declaration (e.g., legacy code path).
        self._declared_defaults = declared_defaults
        # ONE lock per store around every load-modify-save. The arbiter's
        # 2 s tick (app/arbiter.py) reads policy concurrently with sync HTTP
        # routes, which FastAPI runs on a real threadpool — and _save writes
        # a FIXED .tmp path, so two writers race it and the loser's
        # os.replace raises FileNotFoundError into whichever thread it was:
        # a 500 on /api/status right when someone is looking. Sharpened by
        # the boundary gate (task 3), which made _load() itself a WRITER on
        # any partial/corrupt file — so the heal-write must happen with this
        # lock already held, which is why every public method below takes it
        # and _load/_save stay lock-free internals.
        self._lock = threading.Lock()

    def _gated(self, data: dict) -> dict:
        """Element-level boundary gate (NodeStore._load's pattern), run
        exclusively by `_load()` — see the module docstring. Declared resources
        end up present and valid with their defaults, since decide() and the UI's
        per-resource destructure index them unconditionally. Malformed records are
        replaced by their default; a runtime resource (not declared) survives only
        if it validates — there is no default to heal it to. `_auto` survives only
        as a dict with a real bool `enabled`; a malformed one is dropped, healing
        to `auto_enabled()`'s own default-True reading rather than crashing the
        tick with an AttributeError."""
        gated: dict = {}
        declared = (self._declared_defaults or (lambda: {}))()
        for resource, policy in data.items():
            if resource == _AUTO_KEY:
                if isinstance(policy, dict) and isinstance(policy.get("enabled"), bool):
                    gated[resource] = policy
                continue
            if not isinstance(policy, dict):
                continue
            try:
                _validate_policy(resource, policy)
            except ValueError:
                continue
            gated[resource] = policy
        for resource, default in declared.items():
            if resource not in gated:
                gated[resource] = dict(default)
        return gated

    def _load(self) -> dict[str, TenantPolicy] | None:
        """Missing/corrupt file -> None, the shared "absent" signal every
        caller below falls back on. Otherwise every record is run through
        the boundary gate (`_gated`) and the heal is persisted immediately
        if anything changed — the sole gating site (module docstring); no
        consumer below re-gates its result."""
        data = load_json(self._path)
        if not isinstance(data, dict):
            return None
        gated = self._gated(data)
        if gated != data:
            self._save(gated)        # persist the heal, like the missing-file path
        return gated

    def _save(self, data: dict[str, TenantPolicy]) -> None:
        save_json(self._path, data)

    def get(self) -> dict[str, TenantPolicy]:
        """Full resource->policy mapping, excluding reserved config keys.

        Returns one row per DECLARED resource: declared defaults overlaid by any
        stored override row. Stored rows for undeclared resources are kept on
        disk (e.g., orphaned rows from hand-edits or older writes) but invisible
        on read.

        On first read (file missing or corrupt), materializes the declared
        defaults and persists them before returning. Every other shape concern —
        a partial or malformed file — is already healed by `_load()` (the sole
        boundary gate; see the module docstring).
        """
        declared = (self._declared_defaults or (lambda: {}))()
        with self._lock:
            data = self._load()
            if data is None:
                data = {resource: dict(policy)
                        for resource, policy in declared.items()}
                self._save(data)
                return dict(data)
            # Return only declared resources, using stored values or defaults
            return {k: v for k, v in data.items() if k in declared and k != _AUTO_KEY}

    def put(self, policies: dict[str, TenantPolicy]) -> None:
        """Partial update by resource: replaces the whole record for each resource
        named in `policies`, leaving resources not named untouched.

        Field validation is unchanged and still strict (priority: int not bool;
        pinned: bool; idle_ttl: int >= 0, not bool), and the whole payload is
        validated before anything is written, so a rejected put leaves the
        file untouched.
        """
        if _AUTO_KEY in policies:
            raise ValueError(f"{_AUTO_KEY!r} is reserved; use set_auto()")
        for resource, policy in policies.items():
            _validate_policy(resource, policy)

        # _load() rather than get(): get() filters the reserved _auto key, so
        # reading through it would silently drop the automation setting on
        # every policy write. _load() is the sole boundary gate (module
        # docstring), so a missing/corrupt file, or a partial hand-edited
        # one, is already healed by the time this merge runs.
        declared = (self._declared_defaults or (lambda: {}))()
        with self._lock:
            current = self._load()
            if current is None:
                current = {resource: dict(policy)
                           for resource, policy in declared.items()}
            current.update({resource: dict(policy) for resource, policy in policies.items()})
            self._save(current)

    # --- lifecycle automation toggle ---------------------------------------

    def auto_enabled(self) -> bool:
        """Whether the reconciler may act. Defaults to True: unlike storage
        tiering (whose automation moves bytes and defaults off), lifecycle
        auto-restore only returns a resource to a state the operator already
        chose, and its absence is what let hipfire stay dead for 26 hours."""
        with self._lock:
            data = self._load() or {}
        value = data.get(_AUTO_KEY, {}).get("enabled", True)
        return bool(value)

    def set_auto(self, enabled: bool) -> None:
        """Persist the automation toggle, seeding resource defaults if the file
        has never been written.

        The seeding matters: `_load()` self-heals a file that already
        exists — even a partial or malformed one, resource records included
        (module docstring) — but a file that has never been written
        returns None from `_load()`, the same "absent" signal every other
        consumer falls back on. Writing the toggle first must not be able
        to cost the deck its defaults.
        """
        declared = (self._declared_defaults or (lambda: {}))()
        with self._lock:
            data = self._load()
            if data is None:
                data = {resource: dict(policy)
                        for resource, policy in declared.items()}
            data[_AUTO_KEY] = {"enabled": bool(enabled)}
            self._save(data)


# --- Storage tiering policy -------------------------------------------------

STORAGE_POLICY_DEFAULT = {"auto": False}


class StoragePolicyStore:
    """Auto-tiering mode, persisted to ``storage_policy.json`` — this module's
    second owned file. Same atomic-write/self-heal quality bar as PolicyStore."""

    def __init__(self, path: Path):
        self._path = path
        # Same race, slower cadence: StorageWatcher's 60 s pass heals via
        # get() while HTTP routes read and write the same fixed .tmp path.
        self._lock = threading.Lock()

    def _load(self) -> dict | None:
        data = load_json(self._path)
        if not isinstance(data, dict) or set(data) != {"auto"} or not isinstance(data.get("auto"), bool):
            return None
        return data

    def _save(self, data: dict) -> None:
        save_json(self._path, data)

    def get(self) -> dict:
        with self._lock:
            data = self._load()
            if data is None:
                data = dict(STORAGE_POLICY_DEFAULT)
                self._save(data)
            return data

    def put(self, policy: dict) -> None:
        if set(policy) != {"auto"} or not isinstance(policy.get("auto"), bool):
            raise ValueError('storage policy must be exactly {"auto": <bool>}')
        with self._lock:
            self._save(dict(policy))
