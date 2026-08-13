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
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.engines import GuardError
from app.store_io import load_json, save_json

# `\Z`, not `$`: Python's `$` also matches just BEFORE a trailing newline, so
# "sparky\n" satisfied a pattern meant to be fully anchored. The UI mirrors
# this pattern in JS, where `$` does not admit that — the two vocabularies
# have to agree, and an id is a KEY (intent, settings scopes, oci:<id>:
# provenance all attach through it) [max-review c33].
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\Z")
_AGENT_KINDS = {"local", "node-agent"}
_CONTROLS = {"none", "swap"}
_PATCHABLE = {"label", "address", "serving_address", "control", "engines"}
_ALLOWED = {"id", "label", "agent_kind", "address", "serving_address", "control", "engines"}

# The OLD env-seeded spark node id — frozen MIGRATION DATA, not coupling:
# pre-N1 installs seeded exactly this id (it was app.observe.spark_node_id(),
# deleted in N1), and intent/settings/provenance keys on those installs
# attach through it. Used only by seed_if_missing callers and
# stamp_missing_control; never to build a key for a new node.
LEGACY_SPARK_SEED_ID = "sparky"


def _well_formed(entry: object) -> bool:
    """The element-level shape gate applied by `NodeStore._load()`. Deliberately
    lighter than `_validate()`: only what every downstream consumer indexes
    unconditionally (`entry["id"]`, `entry["label"]`, `entry["agent_kind"]`)
    is checked here. `address`/`serving_address` stay unchecked by design —
    their presence is a per-agent-kind SEMANTIC rule (node-agent needs one,
    local doesn't), not a shape rule, and callers that care already handle
    "absent" (e.g. app/node_observer.py's node-agent-without-address skip).

    The id must also match `_ID_RE`: an id is a KEY (intent, settings
    scopes, `oci:<id>:` provenance all attach through it), and a hand-edited
    id the write-side `_validate()` would refuse — pre-`\\Z`-fix legacy data
    like ``"sparky\\n"`` — is otherwise a ghost row no PATCH can ever touch
    (every patch re-validates the id → 422 forever) [T9 review m-item].
    Dropping it at the load gate matches this module's documented posture:
    the row disappears, and the node can be re-added properly."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and _ID_RE.match(entry["id"]) is not None
        and isinstance(entry.get("label"), str)
        and entry.get("agent_kind") in _AGENT_KINDS
    )


def _usable_url(value: str) -> bool:
    """Enough URL validity that an httpx.Client can be CONSTRUCTED from it —
    the write-side gate that keeps a typo from ever reaching a client
    factory ([[guard-at-the-boundary]]). urlsplit raises on malformed IPv6
    brackets; .port raises on a non-numeric port; scheme/host cover bare
    or relative strings. Deliberately permissive past that — no DNS, no
    reachability."""
    try:
        parts = urlsplit(value)
        parts.port
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _heal_control(entry: dict) -> dict:
    control = entry.get("control")
    if not isinstance(control, str) or control not in _CONTROLS:
        return {**entry, "control": "none"}
    return entry


def _heal_engines(entry: dict) -> dict:
    """Heal invalid engines (guard-at-the-boundary posture).

    Non-local entries must not carry an engines key at all; if present,
    strip it (guard-at-boundary: reject at load, don't brick future patches).
    Local entries with schema-invalid engines heal to [] rather than being
    dropped, preserving the entry."""
    if "engines" not in entry:
        return entry

    # Non-local entries must never have engines key
    if entry.get("agent_kind") != "local":
        return {k: v for k, v in entry.items() if k != "engines"}

    # Local entry: validate schema, heal to [] if invalid
    engines = entry.get("engines")
    try:
        from app.engine_kinds import validate_engines
        validate_engines(engines)
        return entry
    except ValueError:
        # Invalid engines: heal to empty list instead of dropping the entry
        return {**entry, "engines": []}


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
        if field in spec and spec[field] is not None:
            if not isinstance(spec[field], str):
                raise ValueError(f"{field} must be a string or null")
            if not _usable_url(spec[field]):
                raise ValueError(f"{field} must be an absolute http(s) URL")
    if "control" in spec:
        if not isinstance(spec["control"], str) or spec["control"] not in _CONTROLS:
            raise ValueError(f"control must be one of {sorted(_CONTROLS)}")
        if spec["control"] == "swap" and spec["agent_kind"] == "local":
            raise ValueError(
                'the local node cannot be control: "swap" — local actuation '
                "is docker-ctl, not the swap protocol (G1 revisits)")
    if "engines" in spec:
        if spec["agent_kind"] != "local":
            raise ValueError("engines is only valid on the local node entry")
        from app.engine_kinds import validate_engines
        validate_engines(spec["engines"])


class NodeStore:
    """Node topology + write-only credential sidecar, persisted to `path`
    and `credentials_path`."""

    def __init__(self, path: Path, credentials_path: Path):
        self._path = path
        self._creds_path = credentials_path
        # One lock around every MUTATOR's load-modify-save. Reachable from
        # FastAPI's sync-route threadpool (real OS threads), where two
        # concurrent writes to different nodes still read-modify-write the
        # same file — and each of the TWO fixed .tmp paths here (nodes and
        # credentials) is its own race surface whose loser's os.replace
        # raises FileNotFoundError into a route [T9b sweep].
        #
        # Reads (get/list/credential_*) deliberately stay lock-free: add()
        # calls get() internally, so locking reads with a non-reentrant Lock
        # would deadlock. Holding it across the whole mutator body also makes
        # the duplicate-id check atomic with the append it guards.
        self._lock = threading.Lock()

    # -- persistence (LocationStore idiom) ----------------------------------

    def _load(self) -> list[dict]:
        data = load_json(self._path)
        if not isinstance(data, list):
            return []
        # Element-level gate (see module docstring): a hand-edited malformed
        # entry is dropped here, silently, same as the file-level self-heal
        # above — every caller downstream may assume the shape holds.
        # `control` is HEALED rather than dropped (missing/invalid -> "none"):
        # pre-N1 files have no such key, and an entry losing its whole row
        # over a field this increment introduced would be the gate punishing
        # old data for new vocabulary. One gate, here — no per-site guards.
        # `engines` is similarly healed to [] if invalid, preserving the entry.
        return [_heal_engines(_heal_control(entry)) for entry in data if _well_formed(entry)]

    def _save(self, data: list[dict]) -> None:
        save_json(self._path, data)

    def _load_creds(self) -> dict:
        data = load_json(self._creds_path)
        return data if isinstance(data, dict) else {}

    def _save_creds(self, data: dict) -> None:
        # chmod=0o600 lands on the TMP file, before replace: the final path
        # must never exist world-readable, not even for one rename's duration.
        save_json(self._creds_path, data, chmod=0o600)

    # -- API ----------------------------------------------------------------

    def exists(self) -> bool:
        """Whether nodes.json has ever been written — the seed-once gate."""
        return self._path.exists()

    def list(self) -> list[dict]:
        return self._load()

    def get(self, node_id: str) -> dict | None:
        return next((n for n in self._load() if n["id"] == node_id), None)

    def _require_swap_prereqs(self, entry: dict, credential_present: bool) -> None:
        """The three-prerequisite rule for `control: "swap"` (design §2):
        address + serving_address + a credential, present simultaneously.
        Missing fields are NAMED — the 422 is the operator's checklist."""
        missing = [f for f in ("address", "serving_address") if not entry.get(f)]
        if not credential_present:
            missing.append("credential")
        if missing:
            raise ValueError(
                'control: "swap" requires ' + ", ".join(missing)
                + " to be set first")

    def add(self, spec: dict, credential: str | None = None) -> dict:
        spec = dict(spec)
        spec.setdefault("control", "none")
        _validate(spec)
        if spec["control"] == "swap":
            self._require_swap_prereqs(spec, credential_present=bool(credential))
        with self._lock:
            if spec["agent_kind"] == "local" and self.get("local") is not None:
                raise ValueError("the local node is seeded, not added")
            if spec["agent_kind"] == "local" and spec["id"] != "local":
                raise ValueError(
                    "agent_kind 'local' is reserved for the seeded local node")
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
        with self._lock:
            data = self._load()
            for node in data:
                if node["id"] == node_id:
                    merged = {**node, **patch}
                    _validate({k: v for k, v in merged.items() if k != "added_ts"})
                    if merged.get("control") == "swap":
                        self._require_swap_prereqs(
                            merged,
                            credential_present=bool(credential)
                            or self.credential_set(node_id))
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
        with self._lock:
            data = self._load()
            entry = next((n for n in data if n["id"] == node_id), None)
            if entry is not None and entry.get("control") == "swap":
                # A declared-operable node must never lose its credential (the
                # sidecar row dies with the entry) as a side effect of one call;
                # the operator demotes explicitly, then removes (design §2,
                # 08-12 refinement). Any FUTURE credential-clear operation must
                # carry this same gate.
                raise GuardError(
                    f"node {node_id!r} is declared operable (control: \"swap\"); "
                    'set control to "none" first')
            data = [n for n in data if n["id"] != node_id]
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

        Used by app.node_clients' binding_view, which compares a swap node's
        live client binding against the registry's current state to decide
        whether the client needs rebuilding. The credential is part of that
        binding (the factory receives it as ``node_key``), so a rotation
        triggers a rebuild exactly like an address edit — but the nodes
        router's credential API is write-only by contract (see its module
        docstring), and a digest keeps it that way."""
        value = self.credential_for(node_id)
        if not value:
            return None
        return hashlib.sha256(value.encode()).hexdigest()

    def stamp_missing_control(self, swap_id: str) -> bool:
        """One-time migration for pre-N1 files (design §8): stamp `control`
        ON DISK for every well-formed entry that lacks the key. The entry
        whose id is `swap_id` (the old env-seeded spark id) gets `"swap"`
        IFF it has address + serving_address + a stored credential — exactly
        the condition under which the pre-N1 main.py would have bound a
        client; anything less stays `"none"` and the board shows the node
        observe-only (honest, recoverable via the UI toggle).

        Keys on the ABSENCE of `control` in the RAW file, so it runs at most
        once per entry lifetime: after the first write every entry has the
        key, and a later operator demotion is never re-promoted. Reads raw
        (`load_json`, not `_load()`) because `_load()` heals in memory —
        healed entries would look already-stamped. Malformed elements are
        left byte-identical (the migration must not be a destructive pass;
        dropping them stays `_load()`'s job)."""
        with self._lock:
            data = load_json(self._path)
            if not isinstance(data, list):
                return False
            changed = False
            for entry in data:
                if not _well_formed(entry) or "control" in entry:
                    continue
                if (entry["id"] == swap_id
                        and entry.get("address")
                        and entry.get("serving_address")
                        and self.credential_set(entry["id"])):
                    entry["control"] = "swap"
                else:
                    entry["control"] = "none"
                changed = True
            if changed:
                self._save(data)
            return changed


def seed_if_missing(store: NodeStore, *, node_label: str, spark_id: str,
                    spark_node_url: str, spark_serving_url: str,
                    spark_node_name: str, spark_node_keys_json: str) -> bool:
    """One-time migration of env-var node config into the registry.

    Runs ONLY while nodes.json does not exist; once it does, env is never
    consulted again — no per-boot merge that could resurrect a removed
    entry. The spark entry's id is the caller-passed `spark_id`, which MUST
    be `LEGACY_SPARK_SEED_ID`: every keyed datum — intent, settings scopes,
    oci:<id>: provenance — on a pre-N1 install attaches through that string.

    The keys-json parse mirrors app/main.py's historical tolerance: a
    malformed map degrades to "no credential" (the node still seeds, the
    observer reports it unconfigured), never a crash at first boot.

    The seed also migrates the OLD SCHEME's operability declaration (design
    §8): the env-var configuration alone was what main.py used to bind an
    actuation client at boot, so a fresh box booting with that same
    configuration must land in the same state a stamp_missing_control
    upgrade would — control:"swap" when a credential is present, "none"
    otherwise — not the unconditional "none" a bare `add()` defaults to.
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
        credential = key if isinstance(key, str) and key else None
        store.add({"id": spark_id, "label": spark_id, "agent_kind": "node-agent",
                   "address": spark_node_url, "serving_address": spark_serving_url,
                   # The env-var config WAS the old scheme's declaration of
                   # operability (main.py bound a client from exactly this
                   # condition), so the seed migrates the declaration too —
                   # with no credential the node seeds observe-only, honest
                   # and recoverable via the UI toggle (design §8).
                   "control": "swap" if credential else "none"},
                  credential=credential)
    return True


def seed_engines_if_missing(store: "NodeStore", settings, data_dir: Path) -> bool:
    """One-time engines[] seed (E1 spec §1). Keyed on raw-file absence of
    the `engines` key on the local entry — after the first stamp the file
    owns the truth and the env settings are dead (N1 stamp pattern).

    Presence proof: only stamp today's triple when intent.json or
    policy.json shows a pre-E1 install that was actually MANAGING the
    legacy names. Env defaults naming llama-server are not proof
    llama-server exists (coexistence, spec §6). No proof -> stamp []."""
    local = store.get("local")
    if local is None or "engines" in local:
        return False
    legacy = ("lemonade", "comfyui", "hipfire")
    proof = False
    for name in ("intent.json", "policy.json"):
        p = Path(data_dir) / name
        if p.exists():
            try:
                text = p.read_text()
            except OSError:
                continue
            if any(k in text for k in legacy):
                proof = True
                break
    engines = [] if not proof else [
        {"resource": "hipfire", "kind": "hipfire",
         "connection": {"container": settings.hipfire_container},
         "gpu_index": settings.hipfire_gpu_index,
         "policy_defaults": {"priority": 100, "pinned": True, "idle_ttl": 0}},
        {"resource": "lemonade", "kind": "lemonade",
         "connection": {"url": settings.lemonade_url,
                        "metrics_url": settings.lemonade_metrics_url,
                        "container": settings.lemonade_container},
         "gpu_index": settings.lemonade_gpu_index,
         "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
        {"resource": "comfyui", "kind": "comfyui",
         "connection": {"url": settings.comfyui_url},
         "gpu_index": settings.lemonade_gpu_index,
         "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
    ]
    store.update("local", {"engines": engines})
    return True
