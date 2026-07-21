"""
Hipfire park/resume client — container lifecycle guarded by litellm route
awareness.

hipfire is the big always-loaded model engine; parking it (stopping its
container) frees VRAM for other work, but only when litellm's `default`
route isn't currently pointed at it — parking it out from under an
in-flight default route would silently break inference for every caller
that didn't ask for hipfire by name. This client composes DockerCtl (which
container to stop/start) with LiteLLMClient (is it safe to stop) and a
direct health-check GET (is it done coming back up).

status() distinguishes:
  "parked"  — dockerctl.running(container) is False (health is not checked
              at all in this case: a parked container that isn't running
              has nothing listening at health_url).
  "loading" — container is running but the health GET didn't return 200
              (503 is the documented case; any other non-200 also reads as
              still-loading rather than a hard failure).
  "running" — container is running and health GET returned 200.

A transport-level failure reaching health_url while the container is known
to be running propagates as EngineError (that's not "loading", it's "we
can't tell" — different from the ordinary loading path).

park() checks litellm.default_targets_hipfire() BEFORE calling
dockerctl.stop(); if that check itself raises EngineError, it propagates
unchanged and the container is never touched — fail safe: if we can't see
the route table, we don't park. If the check returns True, GuardError is
raised (not an EngineError subclass) and stop() is never called.

resume() only starts the container; it does not poll status() — callers
that need to know when hipfire has finished loading call status() in a
loop themselves.

A `transport=` kwarg lets tests inject httpx.MockTransport for this
client's own internal health-check httpx.Client. DockerCtl and
LiteLLMClient are passed in fully constructed, each carrying its own
transport seam independently.
"""

import httpx

from app.engines import EngineError, GuardError
from app.engines.docker_ctl import DockerCtl
from app.engines.litellm import LiteLLMClient

_TIMEOUT = 5.0
_HEALTHY = 200


class HipfireClient:
    def __init__(
        self,
        health_url: str,
        dockerctl: DockerCtl,
        container: str,
        litellm: LiteLLMClient,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._health_url = health_url
        self._dockerctl = dockerctl
        self._container = container
        self._litellm = litellm
        self._client = httpx.Client(timeout=_TIMEOUT, transport=transport)

    def status(self) -> str:
        if not self._dockerctl.running(self._container):
            return "parked"
        try:
            resp = self._client.get(self._health_url)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        return "running" if resp.status_code == _HEALTHY else "loading"

    def park(self) -> None:
        if self._litellm.default_targets_hipfire():
            raise GuardError(
                f"refusing to park {self._container!r}: "
                "litellm's default route currently targets hipfire"
            )
        self._dockerctl.stop(self._container)

    def resume(self) -> None:
        self._dockerctl.start(self._container)
