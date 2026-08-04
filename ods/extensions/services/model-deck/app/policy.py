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

    def _load(self) -> dict[str, TenantPolicy] | None:
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
        return data

    def _save(self, data: dict[str, TenantPolicy]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self._path)

    def get(self) -> dict[str, TenantPolicy]:
        """Full tenant->policy mapping, excluding reserved config keys.

        On first read (file missing or corrupt), materializes the default
        policies and persists them before returning.
        """
        data = self._load()
        if data is None:
            data = {tenant: dict(policy) for tenant, policy in DEFAULT_POLICIES.items()}
            self._save(data)
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
        # every policy write. A missing/corrupt file still seeds the defaults,
        # matching get()'s self-heal, so a put on a fresh deck doesn't leave
        # the untouched tenants unpolicied.
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
        data = self._load() or {}
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
