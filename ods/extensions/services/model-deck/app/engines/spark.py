"""Spark (remote DGX) swap client — single-slot engine, lifecycle only.

The Spark serves exactly one vLLM model at a time on its own box, so there
is no VRAM arbitration to do here and this client is deliberately NOT part
of the World snapshot or the arbiter's math. It exists so the deck is the
one control plane: see the current profile, list the available ones, and
swap — through the node-agent's file-protocol swap control (the agent has
no docker access; a host-side helper validates and runs swap.sh).

Guard order in swap(), mirroring hipfire.py's park():
  1. litellm guard — refuse if litellm's `default` route points at the
     Spark's serving endpoint; a swap would break every caller that never
     asked for a spark model by name. EngineError from the route-table
     read propagates unchanged (can't see the table -> don't swap).
     force NEVER skips this guard.
  2. busy guard — refuse while the live model has running or waiting
     requests (vLLM /metrics). endpoint_ok False skips this check (a dead
     endpoint has nothing to interrupt — that's the recovery path). A
     metrics failure while the endpoint is up is EngineError, not "not
     busy". force=True skips this guard only.
The node-agent's own 409 (pending request / helper mid-swap) surfaces as
BusyError; its 4xx validation answers surface as EngineError.
"""

from urllib.parse import urlparse

import httpx

from app.engines import BusyError, EngineError, GuardError
from app.engines.litellm import LiteLLMClient

_TIMEOUT = httpx.Timeout(5.0)

_BUSY_METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting")


class SparkClient:
    def __init__(
        self,
        node_url: str,
        node_key: str,
        serving_url: str,
        litellm: LiteLLMClient,
        node_transport: httpx.BaseTransport | None = None,
        serving_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._node = httpx.Client(
            base_url=node_url.rstrip("/"),
            headers={"Authorization": f"Bearer {node_key}"},
            timeout=_TIMEOUT,
            transport=node_transport,
        )
        self._serving = httpx.Client(
            base_url=serving_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=serving_transport,
        )
        self._serving_host = urlparse(serving_url).netloc
        self._litellm = litellm

    # -- reads ------------------------------------------------------------

    def _node_get(self, path: str) -> dict:
        try:
            resp = self._node.get(path)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def status(self) -> dict:
        profiles = self._node_get("/v1/node/profiles")
        serving = self._node_get("/v1/node/serving")
        return {
            "profiles": profiles.get("profiles", []),
            "swap_status": profiles.get("swap_status"),
            "serving": serving,
        }

    def busy_requests(self) -> int:
        """Sum of running+waiting requests from vLLM's /metrics."""
        try:
            resp = self._serving.get("/metrics")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        total = 0.0
        for line in resp.text.splitlines():
            if line.startswith(_BUSY_METRICS):
                try:
                    total += float(line.rsplit(None, 1)[-1])
                except ValueError:
                    raise EngineError(f"unparseable metric line: {line!r}")
        return int(total)

    # -- swap -------------------------------------------------------------

    def _default_route_targets_spark(self) -> bool:
        for entry in self._litellm.model_info():
            if entry["model_name"] == "default":
                api_base = entry["litellm_params"].get("api_base", "")
                return self._serving_host in api_base
        return False

    def swap(self, profile: str, force: bool = False) -> dict:
        if self._default_route_targets_spark():
            raise GuardError(
                "litellm's default route targets the spark serving endpoint; "
                "swapping would break default-route callers (force does not "
                "override this)")

        serving = self._node_get("/v1/node/serving")
        if serving.get("endpoint_ok") and not force:
            n = self.busy_requests()
            if n > 0:
                raise GuardError(
                    f"spark serving has {n} in-flight request(s); "
                    "retry later or use force")

        try:
            resp = self._node.post("/v1/node/swap", json={"profile": profile})
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code == 409:
            raise BusyError(resp.text)
        if not resp.is_success:
            raise EngineError(resp.text)
        body = resp.json()
        return {"id": body.get("id"), "profile": profile}
