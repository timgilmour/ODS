"""Model Deck tenant policy store.

Tracks the arbitration policy for each known tenant engine (hipfire,
lemonade, comfyui): its eviction priority, whether it's pinned (exempt from
eviction), and its idle-TTL before Model Deck parks it.

``policy.json`` is a flat mapping of ``{tenant: {priority, pinned,
idle_ttl}}``, persisted next to no other state — this module owns the whole
file. Writes are atomic (temp file + ``os.replace``) since the supervisor
may crash mid-write; a missing or corrupt file is treated as absent rather
than raised, and self-heals by materializing and persisting the defaults on
the next ``get()``.

Malformed-record gating lives HERE, at ``_load()``, and nowhere else
(NodeStore's pattern, app/node_store.py:18-30): a parseable file that is
missing or malformed for one tenant self-heals the same way as the
whole-file case, one level down. ``_load()`` runs every record through
``_gated()`` and persists the heal immediately if anything changed, so
``get()``/``put()``/``set_auto()`` all consume an already-guaranteed shape
and gate nothing themselves. A missing or malformed known tenant is
replaced by its default; a runtime tenant (accepted since 1ee64611)
survives only if it validates — there is no default to heal it to. A
malformed ``_auto`` record (wrong shape, or a non-bool ``enabled``) is
dropped rather than kept, healing to ``auto_enabled()``'s default-True
reading instead of crashing the tick.

This is single-process, in-process state only — no cross-process locking.
The supervisor is the sole owner of policy.json.
"""

import json
import os
from pathlib import Path

TenantPolicy = dict[str, int | bool]

DEFAULT_POLICIES: dict[str, TenantPolicy] = {
    "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
    "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
    "comfyui": {"priority": 40, "pinned": False, "idle_ttl": 300},
}

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

    def __init__(self, path: Path):
        self._path = path

    def _gated(self, data: dict) -> dict:
        """Element-level boundary gate (NodeStore._load's pattern), run
        exclusively by `_load()` — see the module docstring. Every
        DEFAULT_POLICIES tenant ends up present and valid, since decide()
        and the UI's per-tenant destructure (ui/src/model/nodes.ts:186)
        index them unconditionally. Malformed tenant records are replaced
        by their default (defaults are seed data); a runtime tenant
        (accepted since 1ee64611) survives only if it validates — there is
        no default to heal it to. `_auto` survives only as a dict with a
        real bool `enabled`; a malformed one (wrong shape, or a non-bool
        `enabled`) is dropped, healing to `auto_enabled()`'s own
        default-True reading rather than crashing the tick with an
        AttributeError."""
        gated: dict = {}
        for tenant, policy in data.items():
            if tenant == _AUTO_KEY:
                if isinstance(policy, dict) and isinstance(policy.get("enabled"), bool):
                    gated[tenant] = policy
                continue
            if not isinstance(policy, dict):
                continue
            try:
                _validate_policy(tenant, policy)
            except ValueError:
                continue
            gated[tenant] = policy
        for tenant, default in DEFAULT_POLICIES.items():
            if tenant not in gated:
                gated[tenant] = dict(default)
        return gated

    def _load(self) -> dict[str, TenantPolicy] | None:
        """Missing/corrupt file -> None, the shared "absent" signal every
        caller below falls back on. Otherwise every record is run through
        the boundary gate (`_gated`) and the heal is persisted immediately
        if anything changed — the sole gating site (module docstring); no
        consumer below re-gates its result."""
        try:
            text = self._path.read_text()
        except OSError:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        gated = self._gated(data)
        if gated != data:
            self._save(gated)        # persist the heal, like the missing-file path
        return gated

    def _save(self, data: dict[str, TenantPolicy]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self._path)

    def get(self) -> dict[str, TenantPolicy]:
        """Full tenant->policy mapping, excluding reserved config keys.

        On first read (file missing or corrupt), materializes the default
        policies and persists them before returning. Every other shape
        concern — a partial or malformed file — is already healed by
        `_load()` (the sole boundary gate; see the module docstring), so
        this method does nothing but filter the reserved `_auto` key back
        out.
        """
        data = self._load()
        if data is None:
            data = {tenant: dict(policy) for tenant, policy in DEFAULT_POLICIES.items()}
            self._save(data)
            return dict(data)
        return {k: v for k, v in data.items() if k != _AUTO_KEY}

    def put(self, policies: dict[str, TenantPolicy]) -> None:
        """Partial update by tenant: replaces the whole record for each tenant
        named in `policies`, leaving tenants not named untouched.

        Tenants outside DEFAULT_POLICIES are accepted: the defaults are seed
        data, not an allowlist. Requiring a code edit to policy a new node or
        engine is exactly the rigidity the lifecycle work removes. Field
        validation is unchanged and still strict (priority: int not bool;
        pinned: bool; idle_ttl: int >= 0, not bool), and the whole payload is
        validated before anything is written, so a rejected put leaves the
        file untouched.
        """
        if _AUTO_KEY in policies:
            raise ValueError(f"{_AUTO_KEY!r} is reserved; use set_auto()")
        for tenant, policy in policies.items():
            _validate_policy(tenant, policy)

        # _load() rather than get(): get() filters the reserved _auto key, so
        # reading through it would silently drop the automation setting on
        # every policy write. _load() is the sole boundary gate (module
        # docstring), so a missing/corrupt file, or a partial hand-edited
        # one, is already healed by the time this merge runs.
        current = self._load()
        if current is None:
            current = {tenant: dict(policy) for tenant, policy in DEFAULT_POLICIES.items()}
        current.update({tenant: dict(policy) for tenant, policy in policies.items()})
        self._save(current)

    # --- lifecycle automation toggle ---------------------------------------

    def auto_enabled(self) -> bool:
        """Whether the reconciler may act. Defaults to True: unlike storage
        tiering (whose automation moves bytes and defaults off), lifecycle
        auto-restore only returns a resource to a state the operator already
        chose, and its absence is what let hipfire stay dead for 26 hours."""
        data = self._load() or {}
        value = data.get(_AUTO_KEY, {}).get("enabled", True)
        return bool(value)

    def set_auto(self, enabled: bool) -> None:
        """Persist the automation toggle, seeding tenant defaults if the file
        has never been written.

        The seeding matters: `_load()` self-heals a file that already
        exists — even a partial or malformed one, tenant records included
        (module docstring) — but a file that has never been written
        returns None from `_load()`, the same "absent" signal every other
        consumer falls back on. Writing the toggle first must not be able
        to cost the deck its defaults.
        """
        data = self._load()
        if data is None:
            data = {tenant: dict(policy) for tenant, policy in DEFAULT_POLICIES.items()}
        data[_AUTO_KEY] = {"enabled": bool(enabled)}
        self._save(data)


# --- Storage tiering policy -------------------------------------------------

STORAGE_POLICY_DEFAULT = {"auto": False}


class StoragePolicyStore:
    """Auto-tiering mode, persisted to ``storage_policy.json`` — this module's
    second owned file. Same atomic-write/self-heal quality bar as PolicyStore."""

    def __init__(self, path: Path):
        self._path = path

    def _load(self) -> dict | None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or set(data) != {"auto"} or not isinstance(data.get("auto"), bool):
            return None
        return data

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._path)

    def get(self) -> dict:
        data = self._load()
        if data is None:
            data = dict(STORAGE_POLICY_DEFAULT)
            self._save(data)
        return data

    def put(self, policy: dict) -> None:
        if set(policy) != {"auto"} or not isinstance(policy.get("auto"), bool):
            raise ValueError('storage policy must be exactly {"auto": <bool>}')
        self._save(dict(policy))
