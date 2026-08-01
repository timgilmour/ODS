"""
Host-agent engine client — delegates model activation to the ODS host agent.

Talks to the host agent's REST API over an httpx.Client, authenticated with
`Authorization: Bearer <key>` (same scheme as Lemonade/litellm). A
`transport=` kwarg lets tests inject httpx.MockTransport instead of hitting
the network.

Unlike the other engine clients in this package, activate() uses a wide
600 s read timeout: activation can involve the host agent loading a large
model, and the caller is expected to wait for it synchronously rather than
poll. connect/write/pool stay at the package's usual short bounds so a dead
or unreachable agent still fails fast on connection setup.

activate() raises BusyError (not an EngineError subclass) on HTTP 409: the
host agent holds its own activation lock and is already busy with another
request. It raises EngineError on any other non-2xx response (with the
response text) or on an httpx.TransportError. There is no retry logic here
by design — the host agent owns its own lock/rollback semantics, and a
retry from this layer could race with that.

lifecycle() is a different kind of call: it's the never-raising busy probe
used by the arbiter/control guards to check whether the host agent has an
activation in flight before issuing a competing command. It polls
GET /v1/model/status with a short 2 s timeout (unlike activate's wide 600 s
read timeout, since this is a cheap status check, not a long-running
operation) and always returns a dict — any transport/HTTP/JSON failure is
treated as idle, since an unreachable agent cannot have an activation in
flight and the deck must keep working when the agent is down.
"""

import httpx

from app.engines import BusyError, EngineError

_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
_LIFECYCLE_TIMEOUT = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)

_IDLE = {"active": False, "operation": None, "target": None}


class HostAgent:
    def __init__(
        self,
        base_url: str,
        key: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=_TIMEOUT,
            transport=transport,
        )

    def activate(self, model_id: str) -> dict:
        try:
            resp = self._client.post("/v1/model/activate", json={"model_id": model_id})
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code == 409:
            raise BusyError(resp.text)
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()

    def lifecycle(self) -> dict:
        try:
            resp = self._client.get("/v1/model/status", timeout=_LIFECYCLE_TIMEOUT)
        except httpx.TransportError:
            return dict(_IDLE)
        if not resp.is_success:
            return dict(_IDLE)
        try:
            data = resp.json()
        except ValueError:
            return dict(_IDLE)
        return {
            "active": bool(data.get("lifecycleActive")),
            "operation": data.get("activeOperation"),
            "target": data.get("activeTarget"),
        }
