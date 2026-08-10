"""Model Deck node registry — nodes.json + node_credentials.json.

Topology and credentials are SEPARATE FILES on purpose: nodes.json is safely
readable/backupable (id, label, addresses — no secrets); the credential
sidecar is written 0600 and its values never leave this module except via
credential_for(). The ontology's `credential_ref` is realized as sidecar
membership keyed by node id — nodes.json never names a credential.

`id` is IMMUTABLE IDENTITY: it keys lifecycle intent (<node>/<resource>),
settings scopes (<node>/<engine>), and provenance artifact ids
(oci:<node>:...). There is no rename-id operation; `label` is the editable
display string and must never build a key (app/arbiter.py:1025-1050).

Persistence follows the LocationStore idiom (app/locations.py): atomic
tmp+os.replace writes, corrupt file self-heals to empty. No Settings import
— paths are injected.

Malformed-entry gating lives HERE, at `_load()`, and nowhere else. The
file-level idiom above (an unparseable nodes.json self-heals to an empty
list, never a crash) is applied a second time at the ELEMENT level: any
list element that isn't a dict, or is missing a string `id`/`label`, or
carries an `agent_kind` outside {"local", "node-agent"}, is silently
dropped rather than surfaced. Same quality bar, one level down — a
hand-edited bad element in nodes.json must never take the deck down (it
would otherwise crash `get()`/`list()` callers throughout the app, up to
and including the module-level `app = create_app()` in app/main.py, i.e.
an import-time crash loop). Every consumer of this store — the observer,
the routers, `_build_deck` — is entitled to assume every element in every
list this store returns already has that shape. Do not re-guard for it at
those call sites; if a new shape concern shows up, it belongs here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from app.engines import GuardError

# `\Z`, not `$`: Python's `$` also matches just BEFORE a trailing newline, so
# "sparky\n" satisfied a pattern meant to be fully anchored. The UI mirrors
# this pattern in JS, where `$` does not admit that — the two vocabularies
# have to agree, and an id is a KEY (intent, settings scopes, oci:<id>:
# provenance all attach through it) [max-review c33].
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\Z")
_AGENT_KINDS = {"local", "node-agent"}
_PATCHABLE = {"label", "address", "serving_address"}
_ALLOWED = {"id", "label", "agent_kind", "address", "serving_address"}


def _well_formed(entry: object) -> bool:
    """The element-level shape gate applied by `NodeStore._load()`. Deliberately
    lighter than `_validate()`: only what every downstream consumer indexes
    unconditionally (`entry["id"]`, `entry["label"]`, `entry["agent_kind"]`)
    is checked here. `address`/`serving_address` stay unchecked by design —
    their presence is a per-agent-kind SEMANTIC rule (node-agent needs one,
    local doesn't), not a shape rule, and callers that care already handle
    "absent" (e.g. app/node_observer.py's node-agent-without-address skip)."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("label"), str)
        and entry.get("agent_kind") in _AGENT_KINDS
    )


def _validate(spec: dict) -> None:
    missing = {"id", "label", "agent_kind"} - set(spec)
    extra = set(spec) - _ALLOWED
    if missing or extra:
        raise ValueError(f"node spec: missing {sorted(missing)}, unknown {sorted(extra)}")
    if not isinstance(spec["id"], str) or not _ID_RE.match(spec["id"]):
        raise ValueError(f"node id {spec['id']!r} must be a lowercase slug ([a-z0-9-])")
    if not isinstance(spec["label"], str) or not spec["label"]:
        raise ValueError("label must be a non-empty string")
    if spec["agent_kind"] not in _AGENT_KINDS:
        raise ValueError(f"agent_kind must be one of {sorted(_AGENT_KINDS)}")
    if spec["agent_kind"] == "node-agent" and not spec.get("address"):
        raise ValueError("a node-agent node requires an address")
    for field in ("address", "serving_address"):
        if field in spec and spec[field] is not None and not isinstance(spec[field], str):
            raise ValueError(f"{field} must be a string or null")


class NodeStore:
    """Node topology + write-only credential sidecar, persisted to `path`
    and `credentials_path`."""

    def __init__(self, path: Path, credentials_path: Path):
        self._path = path
        self._creds_path = credentials_path

    # -- persistence (LocationStore idiom) ----------------------------------

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        # Element-level gate (see module docstring): a hand-edited malformed
        # entry is dropped here, silently, same as the file-level self-heal
        # above — every caller downstream may assume the shape holds.
        return [entry for entry in data if _well_formed(entry)]

    def _save(self, data: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._path)

    def _load_creds(self) -> dict:
        try:
            data = json.loads(self._creds_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_creds(self, data: dict) -> None:
        self._creds_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._creds_path.with_suffix(self._creds_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        # chmod the TMP file, before replace: the final path must never
        # exist world-readable, not even for one rename's duration.
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._creds_path)

    # -- API ----------------------------------------------------------------

    def exists(self) -> bool:
        """Whether nodes.json has ever been written — the seed-once gate."""
        return self._path.exists()

    def list(self) -> list[dict]:
        return self._load()

    def get(self, node_id: str) -> dict | None:
        return next((n for n in self._load() if n["id"] == node_id), None)

    def add(self, spec: dict, credential: str | None = None) -> dict:
        spec = dict(spec)
        _validate(spec)
        if spec["agent_kind"] == "local" and self.get("local") is not None:
            raise ValueError("the local node is seeded, not added")
        if spec["agent_kind"] == "local" and spec["id"] != "local":
            raise ValueError("agent_kind 'local' is reserved for the seeded local node")
        if self.get(spec["id"]) is not None:
            raise GuardError(f"node {spec['id']!r} already exists")
        spec["added_ts"] = datetime.now(UTC).isoformat()
        data = self._load()
        data.append(spec)
        self._save(data)
        if credential:
            creds = self._load_creds()
            creds[spec["id"]] = credential
            self._save_creds(creds)
        return spec

    def update(self, node_id: str, patch: dict, credential: str | None = None) -> dict:
        bad = set(patch) - _PATCHABLE
        if bad:
            raise ValueError(f"field(s) not patchable: {sorted(bad)}")
        data = self._load()
        for node in data:
            if node["id"] == node_id:
                merged = {**node, **patch}
                _validate({k: v for k, v in merged.items() if k != "added_ts"})
                node.update(patch)
                self._save(data)
                if credential:
                    creds = self._load_creds()
                    creds[node_id] = credential
                    self._save_creds(creds)
                return node
        raise ValueError(f"unknown node {node_id!r}")

    def remove(self, node_id: str) -> None:
        if node_id == "local":
            raise GuardError("the local node cannot be removed")
        data = [n for n in self._load() if n["id"] != node_id]
        self._save(data)
        creds = self._load_creds()
        if node_id in creds:
            del creds[node_id]
            self._save_creds(creds)
        # Deliberately touches NOTHING keyed by this id: intent, settings
        # scopes, and provenance survive removal (provenance declarations are
        # not re-derivable). Re-adding the same id reattaches everything.

    def credential_for(self, node_id: str) -> str:
        value = self._load_creds().get(node_id, "")
        return value if isinstance(value, str) else ""

    def credential_set(self, node_id: str) -> bool:
        return bool(self.credential_for(node_id))

    def credential_fingerprint(self, node_id: str) -> str | None:
        """A stable, non-reversible stand-in for the credential, so callers
        can answer "is this still the same credential?" without ever holding
        the value. ``None`` when unset — distinct from any real digest.

        Used by the nodes router's ``actuation_stale`` flag, which compares
        the SparkClient's boot-time binding against the registry's current
        state. The credential is part of that binding (main.py passes it as
        ``node_key``), so a rotation goes stale exactly like an address edit
        — but this API is write-only by contract (see the router's module
        docstring), and a digest keeps it that way."""
        value = self.credential_for(node_id)
        if not value:
            return None
        return hashlib.sha256(value.encode()).hexdigest()


def seed_if_missing(store: NodeStore, *, node_label: str, spark_id: str,
                    spark_node_url: str, spark_serving_url: str,
                    spark_node_name: str, spark_node_keys_json: str) -> bool:
    """One-time migration of env-var node config into the registry.

    Runs ONLY while nodes.json does not exist; once it does, env is never
    consulted again — no per-boot merge that could resurrect a removed
    entry. The spark entry's id is the caller-passed `spark_id`, which MUST
    be spark_node_id() (app/observe.py:42-47): every keyed datum — intent,
    settings scopes, oci:<id>: provenance — attaches through that string.

    The keys-json parse mirrors app/main.py's historical tolerance: a
    malformed map degrades to "no credential" (the node still seeds, the
    observer reports it unconfigured), never a crash at first boot.
    """
    if store.exists():
        return False
    store.add({"id": "local", "label": node_label, "agent_kind": "local"})
    if spark_node_url and spark_serving_url:
        try:
            node_keys = json.loads(spark_node_keys_json or "{}")
        except ValueError:
            node_keys = {}
        key = node_keys.get(spark_node_name, "") if isinstance(node_keys, dict) else ""
        store.add({"id": spark_id, "label": spark_id, "agent_kind": "node-agent",
                   "address": spark_node_url, "serving_address": spark_serving_url},
                  credential=key if isinstance(key, str) and key else None)
    return True
