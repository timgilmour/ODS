"""
ComfyUI engine client — queue status and VRAM free/unload.

Talks to ComfyUI's REST API (default http://comfyui:8188) over a 5 s
httpx.Client. ComfyUI has no auth, unlike Lemonade/litellm. A `transport=`
kwarg lets tests inject httpx.MockTransport instead of hitting the network.

Real response shapes this is coded against:
  GET  /queue -> {"queue_running": [...], "queue_pending": [...]}
  POST /free   body {"unload_models": true, "free_memory": true}

queue_len()/free() raise EngineError on a non-2xx response (with the
response text) or on an httpx.TransportError.

free() additionally enforces a guard, at the wire, against freeing VRAM
out from under a running or queued generation: it calls queue_len()
immediately before POSTing /free and raises GuardError (not an
EngineError subclass) if the queue is non-empty. If the guard's own
queue_len() call fails at the transport level, that failure propagates
as EngineError and /free is never sent — fail safe: if we can't see the
queue, we don't free.
"""

import httpx

from app.engines import EngineError, GuardError

_TIMEOUT = 5.0


class ComfyClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=_TIMEOUT,
            transport=transport,
        )

    def queue_len(self) -> int:
        try:
            resp = self._client.get("/queue")
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
        data = resp.json()
        return len(data["queue_running"]) + len(data["queue_pending"])

    def free(self) -> None:
        if self.queue_len() > 0:
            raise GuardError("ComfyUI queue is not empty; refusing to free VRAM")
        try:
            resp = self._client.post(
                "/free", json={"unload_models": True, "free_memory": True}
            )
        except httpx.TransportError as exc:
            raise EngineError(str(exc)) from exc
        if not resp.is_success:
            raise EngineError(resp.text)
