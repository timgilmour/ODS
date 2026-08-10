"""Shared node-agent HTTP plumbing + the observe-only client.

ONE protocol implementation for the node-agent wire (auth header, error
taxonomy) — SparkClient extends this base for its actuation surface
(swap/settings/harvest), NodeAgentClient stays read-only for the node
observer and the Nodes screen's test-connection probe. Two independent
copies of the same protocol is the "two tasks agree on a name, disagree on
its meaning" failure class this codebase has already shipped repeatedly;
this base is the structural answer.

NodeAgentUnreachable vs EngineError: "could not reach it" and "reached it,
it answered badly" must not collapse (the undetermined-vs-unavailable rule)
— the observer maps them to `offline` vs `error` respectively. The subclass
relationship keeps every existing `except EngineError` catch working.
"""

from __future__ import annotations

import httpx

from app.engines import EngineError

_TIMEOUT = httpx.Timeout(5.0)


class NodeAgentUnreachable(EngineError):
    """Transport-level failure: the node-agent did not answer at all."""


class NodeAgentHTTP:
    def __init__(self, node_url: str, node_key: str,
                 transport: httpx.BaseTransport | None = None,
                 timeout: httpx.Timeout = _TIMEOUT) -> None:
        self._node = httpx.Client(
            base_url=node_url.rstrip("/"),
            headers={"Authorization": f"Bearer {node_key}"},
            timeout=timeout,
            transport=transport,
        )

    def _node_get(self, path: str) -> dict:
        try:
            resp = self._node.get(path)
        except httpx.TransportError as exc:
            raise NodeAgentUnreachable(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def close(self) -> None:
        self._node.close()


class NodeAgentClient(NodeAgentHTTP):
    """Read-only observation: /info, /gpu, /serving. No verbs, by design —
    a client with no write methods cannot become a second actuator."""

    def info(self) -> dict:
        return self._node_get("/v1/node/info")

    def gpu(self) -> dict:
        return self._node_get("/v1/node/gpu")

    def serving(self) -> dict:
        return self._node_get("/v1/node/serving")
