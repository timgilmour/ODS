"""sglang-omni remote engine client — status/up/down over the node-agent's
declared-engine channel (extensions/services/node-agent/app.py's
/v1/node/engine/{resource}/{status,up,down} routes; engines.py's
engine_status()/request_engine(), Tasks 1-3 on this branch).

Wire contract:
  GET  /v1/node/engine/{resource}/status
      -> 200 {"reachable": bool, "healthy": bool, "busy_requests": int|None}
      -> 404 `resource` is not declared in the node's engines.json
  POST /v1/node/engine/{resource}/up
  POST /v1/node/engine/{resource}/down
      -> 202 {"accepted": true} -- the node-agent queued the request for
         the host-side swap-helper to act on; nothing here observes the
         result
      -> 404 undeclared, 409 a request is already pending
         (engines.EngineRequestPending), 503 swap-ctl disabled
         (swapctl.SwapCtlDisabled -- app.py's exception handler)

Subclasses NodeAgentHTTP exactly as SparkClient does (its own bearer-authed
httpx.Client, the same base_url/auth/timeout/error-taxonomy plumbing)
rather than composing over a shared instance: NodeAgentClient's own
docstring is explicit that gaining a write method would make it "a second
actuator", so an actuation surface is added by SUBCLASSING the base, never
by reaching into an existing read-only client's internals.

Thin and dumb by design (deck-reconciler-fights-unloads, 2026-08-07): no
retries, no polling for the request to land, no interpretation of the
swap-helper's result files. up()/down() fire one request and return -- the
deck's own reconciler is the only thing allowed to pace retries against a
pending or booting engine; a client that retried here would fight it the
same way the old unload path did.
"""

import httpx

from app.engines import EngineError, guarded_send
from app.engines.node_agent import NodeAgentHTTP

_TIMEOUT = httpx.Timeout(5.0)
_ACCEPTED = 202


class SglangOmniClient(NodeAgentHTTP):
    def __init__(
        self,
        node_url: str,
        node_key: str,
        resource: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(node_url, node_key, transport=transport, timeout=_TIMEOUT)
        self._resource = resource

    def status(self) -> dict:
        """GET .../status -> the wire dict verbatim ({"reachable",
        "healthy", "busy_requests"}). Any failure (404 unknown resource,
        any other non-2xx, or a transport-level connection failure) is
        re-raised naming `resource` -- the node-agent's own response body
        never does (the name lives only in the URL it was sent to, and
        "unknown engine" alone doesn't say which one). The concrete
        exception TYPE from _node_get is preserved (NodeAgentUnreachable
        stays NodeAgentUnreachable, a plain EngineError stays a plain
        EngineError); only the message gains resource context."""
        try:
            return self._node_get(f"/v1/node/engine/{self._resource}/status")
        except EngineError as exc:
            raise type(exc)(
                f"sglang-omni engine {self._resource!r} status: {exc}") from exc

    def up(self) -> None:
        """POST .../up. A 202 means the node-agent queued the request for
        the swap-helper to act on; this call does not wait for it to land
        -- see the module docstring."""
        self._request("up")

    def down(self) -> None:
        """POST .../down. Same shape as up()."""
        self._request("down")

    def _request(self, verb: str) -> None:
        resp = guarded_send(lambda: self._node.post(
            f"/v1/node/engine/{self._resource}/{verb}"))
        if resp.status_code != _ACCEPTED:
            raise EngineError(
                f"sglang-omni engine {self._resource!r} {verb} request not "
                f"accepted (status {resp.status_code}): {resp.text}")
