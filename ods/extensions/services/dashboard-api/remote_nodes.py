"""Remote inference node registry + read-only metrics poller.

Nodes are configured via ODS_REMOTE_NODES (JSON list of
{"name", "display_name"?, "url", "key_env"}). Keys are env-var *names*,
never inline secrets. Absent/empty config → feature dormant.
Terminology: distinct from routers/node.py (local snapshot) and from the
remote-provider/peer inference-routing machinery.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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
    nodes: list[RemoteNodeConfig] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") \
                or not entry.get("url"):
            logger.warning("Skipping malformed remote node entry: %r", entry)
            continue
        key = os.environ.get(entry.get("key_env", ""), "")
        nodes.append(RemoteNodeConfig(
            name=str(entry["name"]), url=str(entry["url"]).rstrip("/"),
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
    try:
        gpu_resp = await client.get(f"{cfg.url}/v1/node/gpu", headers=headers)
        serving_resp = await client.get(f"{cfg.url}/v1/node/serving",
                                        headers=headers)
        if gpu_resp.status_code != 200:
            return _carry(cfg, "error",
                          f"node returned HTTP {gpu_resp.status_code}")
        gpu_body = gpu_resp.json()
        serving = None
        if serving_resp.status_code == 200:
            serving = RemoteNodeServing(**serving_resp.json())
        return RemoteNodeStatus(
            name=cfg.name, display_name=cfg.display_name,
            platform=str(gpu_body.get("backend", "unknown")),
            status="online", last_seen=_now_iso(),
            gpus=[IndividualGPU(**g) for g in gpu_body.get("gpus", [])],
            serving=serving, error=None)
    except (httpx.TransportError, asyncio.TimeoutError):
        return _carry(cfg, "offline", None)
    except (ValueError, TypeError) as exc:  # bad JSON / bad shape
        return _carry(cfg, "error", f"malformed node response: {exc}")


async def poll_all_nodes_once(client: httpx.AsyncClient) -> None:
    cfgs = load_remote_nodes()
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
