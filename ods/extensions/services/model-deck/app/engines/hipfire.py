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

park() then refuses while a hipfire conversation is live (the 2026-07-21
incident: a set apply stopped/recreated the engine under an in-flight
hermes chat; the single-slot conversation cache makes that minutes of
re-prefill). "Live" means either of:
  - the daemon's /stats reports queue_depth > 0 (a request holds the
    single admission slot right now), or
  - requests_served was seen to change within the last
    `activity_window_s` seconds (the conversation is warm — the user is
    between turns, not gone). The tracker is fed by every stats() call
    (the watcher's World snapshot polls it each tick). On the FIRST
    observation after a deck start, requests_served == 0 proves the
    daemon has never served (not busy), while requests_served > 0 is
    unknowable-recency traffic and counts as activity now — conservative
    until the window elapses. activity_window_s = 0 disables the
    recency rule.
`force=True` skips the busy guard (an operator overriding for an
abandoned conversation) but NEVER the litellm route guard — parking the
default route breaks every caller, force or not. A stats transport
failure while the container is running propagates as EngineError and
nothing is stopped (same fail-safe stance as the route check).

resume() only starts the container; it does not poll status() — callers
that need to know when hipfire has finished loading call status() in a
loop themselves. A parked container needs no busy check: nothing can be
in flight.

A `transport=` kwarg lets tests inject httpx.MockTransport for this
client's own internal health/stats httpx.Client. DockerCtl and
LiteLLMClient are passed in fully constructed, each carrying its own
transport seam independently. `clock=` (monotonic) is the activity
tracker's time seam.
"""

import time

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
        stats_url: str | None = None,
        activity_window_s: float = 600.0,
        clock=time.monotonic,
    ) -> None:
        self._health_url = health_url
        self._stats_url = stats_url or health_url.replace("/health", "/stats")
        self._dockerctl = dockerctl
        self._container = container
        self._litellm = litellm
        self._activity_window_s = activity_window_s
        self._clock = clock
        self._served_last: int | None = None
        self._last_activity_time: float | None = None
        self._client = httpx.Client(timeout=_TIMEOUT, transport=transport)

    def status(self) -> str:
        if not self._dockerctl.running(self._container):
            return "parked"
        try:
            resp = self._client.get(self._health_url)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        return "running" if resp.status_code == _HEALTHY else "loading"

    def stats(self) -> dict:
        try:
            resp = self._client.get(self._stats_url)
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if resp.status_code != _HEALTHY:
            raise EngineError(f"GET {self._stats_url} -> {resp.status_code}")
        body = resp.json()
        self._note_activity(body.get("requests_served"))
        return body

    def _note_activity(self, served) -> None:
        if not isinstance(served, int):
            return
        if self._served_last is None:
            # First-ever observation: 0 proves the daemon has never served
            # (no activity to record); >0 is traffic of unknowable recency,
            # so conservatively count it as activity now.
            if served > 0:
                self._last_activity_time = self._clock()
        elif served != self._served_last:
            self._last_activity_time = self._clock()
        self._served_last = served

    def ensure_not_busy(self, action: str) -> None:
        if not self._dockerctl.running(self._container):
            return  # parked: nothing can be in flight
        stats = self.stats()
        queue_depth = stats.get("queue_depth")
        if isinstance(queue_depth, int) and queue_depth > 0:
            raise GuardError(
                f"refusing to {action}: hipfire request in flight "
                f"(queue_depth={queue_depth})"
            )
        if self._last_activity_time is not None and self._activity_window_s > 0:
            age = self._clock() - self._last_activity_time
            if age < self._activity_window_s:
                raise GuardError(
                    f"refusing to {action}: hipfire served a request {age:.0f}s "
                    f"ago (activity window {self._activity_window_s:.0f}s; "
                    "pass force=true to override)"
                )

    def park(self, force: bool = False) -> None:
        if self._litellm.default_targets_hipfire():
            raise GuardError(
                f"refusing to park {self._container!r}: "
                "litellm's default route currently targets hipfire"
            )
        if not force:
            self.ensure_not_busy(f"park {self._container!r}")
        self._dockerctl.stop(self._container)

    def resume(self) -> None:
        self._dockerctl.start(self._container)
