"""Remote inference node registry + read-only metrics poller.

Nodes are configured via ODS_REMOTE_NODES (JSON list of
{"name", "display_name"?, "url", "key_env"}). Authentication keys are delivered
via ODS_REMOTE_NODE_KEYS (a JSON object mapping node name → bearer key), with
key_env as the fallback mechanism. Keys are never inline secrets in the node
definitions. Absent/empty config → feature dormant.
Terminology: distinct from routers/node.py (local snapshot) and from the
remote-provider/peer inference-routing machinery.

Status semantics: the GPU probe (/v1/node/gpu) alone governs a node's
status ("online"/"offline"/"error"); the serving probe is auxiliary and any
failure on it degrades only to serving=None, never the node's status. A node
that answers the GPU probe but reports its own collector failure in that
response's nullable `error` field stays "online" (it is reachable) and carries
that message through, so `error` is not exclusive to status="error".
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from models import IndividualGPU, RemoteNodeServing, RemoteNodeStatus

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0
NODE_TIMEOUT_SECONDS = 2.0

_STATE: dict[str, RemoteNodeStatus] = {}


@dataclass(frozen=True)
class RemoteNodeConfig:
    name: str
    url: str
    key: str
    display_name: str | None = None


def _load_node_keys_map() -> dict:
    """Parse ODS_REMOTE_NODE_KEYS, the preferred key-delivery mechanism: a
    single JSON object {"<node name>": "<bearer key>"}. This exists because
    compose services enumerate their environment explicitly, so an
    admin-chosen key_env var name is never forwarded into the container
    without a per-node compose edit -- one map var sidesteps that.
    Malformed input (bad JSON / not an object) logs a warning and is
    treated as empty, mirroring load_remote_nodes' own malformed-config
    handling; it must never crash the registry.
    """
    raw = os.environ.get("ODS_REMOTE_NODE_KEYS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
    except (ValueError, AssertionError):
        logger.warning("ODS_REMOTE_NODE_KEYS is not a JSON object; ignoring")
        return {}
    return parsed


def _resolve_node_key(name: str, key_env: str, keys_map: dict) -> str:
    """Resolution order: (1) keys_map[name] if present and non-empty,
    (2) key_env lookup (fallback, kept working), (3) empty string. Keys
    are never read from the ODS_REMOTE_NODES topology JSON itself."""
    mapped = keys_map.get(name)
    if mapped:
        return str(mapped)
    return os.environ.get(key_env, "") if key_env else ""


_KEYISH_FIELD_RE = re.compile(r"(key|secret|token|password|pass|credential)",
                              re.IGNORECASE)


def _sanitize_entry_for_log(entry: object) -> object:
    """Redact key-ish values before an entry reaches the log.

    Keys are never *read* from ODS_REMOTE_NODES, but an admin may still have
    inlined one there, and logging the raw entry with %r wrote it verbatim into
    the log. Field names are kept so the message stays diagnosable; only the
    values go. (`key_env` is a variable *name*, not key material, so it is left
    readable -- that is the whole point of the field.)
    """
    if not isinstance(entry, dict):
        return entry
    return {
        k: (v if k == "key_env" or not _KEYISH_FIELD_RE.search(str(k))
            else "[redacted]")
        for k, v in entry.items()
    }


def load_remote_nodes() -> list[RemoteNodeConfig]:
    raw = os.environ.get("ODS_REMOTE_NODES", "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
        assert isinstance(entries, list)
    except (ValueError, AssertionError):
        logger.warning("ODS_REMOTE_NODES is not a JSON list; ignoring")
        return []
    keys_map = _load_node_keys_map()
    nodes: list[RemoteNodeConfig] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") \
                or not entry.get("url"):
            logger.warning("Skipping malformed remote node entry: %r",
                           _sanitize_entry_for_log(entry))
            continue
        name = str(entry["name"])
        key = _resolve_node_key(name, entry.get("key_env", ""), keys_map)
        if not key:
            # The likeliest misconfiguration by far, and otherwise silent until
            # it surfaces as an opaque 401 on the node's dashboard card.
            logger.warning(
                "Remote node %r has no bearer key: neither ODS_REMOTE_NODE_KEYS"
                "[%r] nor key_env %r resolved to a value. The node will report"
                " an HTTP 401 error until a key is supplied.",
                name, name, entry.get("key_env", "") or None)
        nodes.append(RemoteNodeConfig(
            name=name, url=str(entry["url"]).rstrip("/"),
            key=key, display_name=entry.get("display_name")))
    return nodes


def get_remote_node_statuses() -> list[RemoteNodeStatus]:
    return [_STATE[cfg.name] for cfg in load_remote_nodes()
            if cfg.name in _STATE]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _carry(cfg: RemoteNodeConfig, status: str, error: str | None) -> RemoteNodeStatus:
    prev = _STATE.get(cfg.name)
    return RemoteNodeStatus(
        name=cfg.name, display_name=cfg.display_name,
        platform=prev.platform if prev else "unknown",
        status=status, last_seen=prev.last_seen if prev else None,
        gpus=[], serving=None, error=error)


async def _poll_node_once(cfg: RemoteNodeConfig,
                          client: httpx.AsyncClient) -> RemoteNodeStatus:
    headers = {"Authorization": f"Bearer {cfg.key}"}
    # GPU and serving probes run concurrently so a node's wall-clock poll
    # bound is ~one timeout, not two sequential ones. The GPU probe alone
    # governs status/error; the serving probe is auxiliary (see module
    # docstring) and any failure on it just yields serving=None.
    gpu_result, serving_result = await asyncio.gather(
        client.get(f"{cfg.url}/v1/node/gpu", headers=headers),
        client.get(f"{cfg.url}/v1/node/serving", headers=headers),
        return_exceptions=True)

    if isinstance(gpu_result, BaseException):
        if isinstance(gpu_result, (httpx.TransportError, asyncio.TimeoutError)):
            return _carry(cfg, "offline", None)
        return _carry(cfg, "error", f"gpu probe failed: {gpu_result!r}")

    if gpu_result.status_code != 200:
        return _carry(cfg, "error",
                      f"node returned HTTP {gpu_result.status_code}")

    try:
        gpu_body = gpu_result.json()
        gpus = [IndividualGPU(**g) for g in gpu_body.get("gpus", [])]
        platform = str(gpu_body.get("backend", "unknown"))
        # "Node up, collector failing": the agent answered, so the node stays
        # online, but its own collector message is carried through so the card
        # can explain an empty GPU list instead of dead-ending on a blank body.
        # The field is additive -- older agents omit it entirely.
        collector_error = gpu_body.get("error") or None
    except (ValueError, TypeError) as exc:  # bad JSON / bad shape / pydantic
        return _carry(cfg, "error", f"malformed node response: {exc}")

    serving = None
    if not isinstance(serving_result, BaseException) \
            and serving_result.status_code == 200:
        try:
            serving = RemoteNodeServing(**serving_result.json())
        except (ValueError, TypeError):
            serving = None  # auxiliary probe; never demotes node status

    return RemoteNodeStatus(
        name=cfg.name, display_name=cfg.display_name,
        platform=platform, status="online", last_seen=_now_iso(),
        gpus=gpus, serving=serving,
        error=str(collector_error) if collector_error else None)


async def poll_all_nodes_once(client: httpx.AsyncClient) -> None:
    cfgs = load_remote_nodes()
    current_names = {cfg.name for cfg in cfgs}
    for stale_name in set(_STATE) - current_names:
        del _STATE[stale_name]
    results = await asyncio.gather(
        *(_poll_node_once(cfg, client) for cfg in cfgs),
        return_exceptions=True)
    for cfg, result in zip(cfgs, results):
        if isinstance(result, BaseException):
            logger.warning("remote node %s poll crashed: %r", cfg.name, result)
            _STATE[cfg.name] = _carry(cfg, "error", repr(result))
        else:
            _STATE[cfg.name] = result


async def poll_remote_nodes_forever(
        interval: float = POLL_INTERVAL_SECONDS) -> None:
    if not load_remote_nodes():
        logger.info("No remote nodes configured; poller idle-exits")
        return
    async with httpx.AsyncClient(timeout=NODE_TIMEOUT_SECONDS) as client:
        while True:
            try:
                await poll_all_nodes_once(client)
            except Exception:  # never die
                logger.exception("remote node poll cycle failed")
            await asyncio.sleep(interval)
