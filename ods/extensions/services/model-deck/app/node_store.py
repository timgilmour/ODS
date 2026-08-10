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
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from app.engines import GuardError

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_AGENT_KINDS = {"local", "node-agent"}
_PATCHABLE = {"label", "address", "serving_address"}
_ALLOWED = {"id", "label", "agent_kind", "address", "serving_address"}


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
        return data if isinstance(data, list) else []

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
