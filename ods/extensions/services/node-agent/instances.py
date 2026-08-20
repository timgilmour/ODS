"""Deck-created engine INSTANCES — file-protocol actuation (INST I1).

Mirrors engines.request_engine / swapctl.request_swap exactly: the agent
has no docker access; it validates the SHAPE of the deck's instance
document, then drops <ctl>/instance-req.json for the host-side
instances-helper (instances-helper/instances-helper.sh) to consume. Kinds
are NOT known here — the helper resolves `kind` against its own
templates/kinds.json (the security boundary: a compromised agent can at
most ask for one of the operator's own templates). Nothing here reads a
result back for control flow; instance-status-<resource>.json is forensics
the deck may show, never the source of liveness (the deck observes the
container itself).

No "local" literal anywhere: the same file protocol serves a remote node
declaring control:"instances" later (E2 obligation 4).
"""
import json
import re
import time
import uuid
from pathlib import Path

import nodeconfig

DOC_KEYS = frozenset({"resource", "kind", "gpu_indices", "port", "env"})
VERBS = ("create", "remove", "move")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\Z")


class InstancesDisabled(Exception):
    pass


class InstanceRequestPending(Exception):
    pass


def _ctl_dir() -> Path:
    raw = (nodeconfig.NODE_INSTANCES_CTL_DIR or "").strip()
    if not raw:
        raise InstancesDisabled()
    return Path(raw)


def enabled() -> bool:
    try:
        _ctl_dir()
        return True
    except InstancesDisabled:
        return False


def validate_document(doc: object) -> dict:
    if not isinstance(doc, dict) or set(doc) != DOC_KEYS:
        raise ValueError(f"instance document must have exactly the keys {sorted(DOC_KEYS)}")
    res = doc["resource"]
    if not isinstance(res, str) or not NAME_RE.match(res):
        raise ValueError("resource must match ^[a-z0-9][a-z0-9-]*$")
    if not isinstance(doc["kind"], str) or not doc["kind"]:
        raise ValueError("kind must be a non-empty string")
    gpus = doc["gpu_indices"]
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("gpu_indices must be a non-empty list")
    for g in gpus:
        if isinstance(g, bool) or not isinstance(g, int) or g < 0:
            raise ValueError("gpu_indices entries must be non-negative integers")
    port = doc["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not (1024 <= port <= 65535):
        raise ValueError("port must be an int in 1024-65535")
    env = doc["env"]
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise ValueError("env must be an object")
    if not all(isinstance(v, str) for v in env.values()):
        raise ValueError("env values must be strings")
    return doc


def request_instance(verb: str, document: dict) -> None:
    if verb not in VERBS:
        raise ValueError(f"verb must be one of {list(VERBS)}")
    ctl = _ctl_dir()
    req = ctl / "instance-req.json"
    if req.exists():
        raise InstanceRequestPending(
            f"an instance request is already pending, cannot queue {verb!r} for {document['resource']!r}")
    tmp = ctl / f".instance-req.{uuid.uuid4()}.tmp"
    tmp.write_text(json.dumps({"verb": verb, "document": document, "ts": time.time()}))
    tmp.rename(req)


def read_status(resource: str) -> dict | None:
    path = _ctl_dir() / f"instance-status-{resource}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
