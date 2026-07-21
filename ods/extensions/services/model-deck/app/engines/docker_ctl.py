"""
Docker control client — socket-proxy wrapper for container park/resume.

Talks to the Docker Engine API through a tecnativa/docker-socket-proxy
sidecar (default http://docker-ctl:2375) over a 5 s httpx.Client. The proxy
accepts unversioned paths, so no `/v1.4x` prefix is needed. A `transport=`
kwarg lets tests inject httpx.MockTransport instead of touching the network.

`allowlist` is OUR enforcement, independent of (and in addition to) the
proxy's own API-class narrowing: stop()/start() raise GuardError, naming the
rejected container, if `name` is not in `allowlist` — checked BEFORE any
HTTP call is made, so a disallowed name never reaches the socket at all.

Real wire shapes this is coded against:
  GET  /containers/{name}/json  -> 200 {"State": {"Running": bool, ...}, ...}
                                    404 if the container doesn't exist
  POST /containers/{name}/stop  -> 204 (stopped) or 304 (already stopped)
  POST /containers/{name}/start -> 204 (started) or 304 (already running)

running()/stop()/start() raise EngineError on any other non-2xx response
(with the response text) or on an httpx.TransportError. 304 is treated as
success (the container was already in the requested state), not an error.
"""

import httpx

from app.engines import EngineError, GuardError

_TIMEOUT = 5.0
_ALREADY_DONE = 304


class DockerCtl:
    def __init__(
        self,
        base_url: str,
        allowlist: list[str],
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._allowlist = allowlist
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=transport,
        )

    def running(self, name: str) -> bool:
        try:
            resp = self._client.get(f"/containers/{name}/json")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        return resp.json()["State"]["Running"]

    def stop(self, name: str) -> None:
        self._guard(name)
        self._lifecycle_post(name, "stop")

    def start(self, name: str) -> None:
        self._guard(name)
        self._lifecycle_post(name, "start")

    def _guard(self, name: str) -> None:
        if name not in self._allowlist:
            raise GuardError(f"container {name!r} is not in the park allowlist")

    def _lifecycle_post(self, name: str, action: str) -> None:
        try:
            resp = self._client.post(f"/containers/{name}/{action}")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code == _ALREADY_DONE or resp.is_success:
            return
        raise EngineError(resp.text)
