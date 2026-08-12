"""
Model Deck engine clients — shared exceptions.

Every client in this package (LemonadeClient, ComfyClient, LiteLLMClient)
raises EngineError on a non-2xx HTTP response or an httpx transport-level
failure, so callers can catch one type regardless of which engine they
called. LemonadeClient.activity() is the sole exception: it returns None
on any failure instead of raising, because callers use it to gate an
idle-TTL timer and a crash there must not take down the arbiter loop.

GuardError is intentionally NOT an EngineError subclass. It signals a
different kind of failure: the call reached the engine fine, but a safety
guard refused to proceed (e.g. ComfyClient.free() sees a non-empty queue).
Callers that want to distinguish "engine is broken" from "guard tripped,
try again later" need these to be unrelated exception types.

BusyError is similarly NOT an EngineError subclass. It signals that the
host agent reached the request fine but refused it with HTTP 409 because
it already holds its own model-activation lock. Callers that want to
distinguish "host agent is broken" from "host agent is busy, try again
later" need these to be unrelated exception types.
"""


from collections.abc import Callable

import httpx


class EngineError(Exception):
    """An engine HTTP call failed: non-2xx response or transport error."""


class GuardError(Exception):
    """A safety guard refused an engine operation (e.g. ComfyUI queue non-empty)."""


class BusyError(Exception):
    """The host agent returned HTTP 409: it is already busy with another activation."""


def guarded_send(send: Callable[[], "httpx.Response"]) -> "httpx.Response":
    """The transport half of the wire idiom: run one httpx call, mapping
    httpx.TransportError to EngineError. For callers whose STATUS handling
    is their own vocabulary (hipfire's healthy/loading split) — everyone
    else wants engine_request below."""
    try:
        return send()
    except httpx.TransportError as exc:
        raise EngineError(str(exc)) from exc


def engine_request(send: Callable[[], "httpx.Response"], *,
                   also_ok: tuple[int, ...] = ()) -> "httpx.Response":
    """The engine-client wire idiom, in ONE place instead of hand-copied
    per call site [max-review c9]: httpx.TransportError becomes
    EngineError(str(exc)), a non-2xx response becomes EngineError carrying
    the response text. ``also_ok`` admits status codes with call-specific
    meaning past the raise for the caller to interpret — docker's 304
    already-in-that-state, the 409s hostagent.activate and spark.swap turn
    into BusyError. A diagnosability change to this mapping now lands at
    every engine at once instead of drifting per copy."""
    resp = guarded_send(send)
    if resp.is_success or resp.status_code in also_ok:
        return resp
    raise EngineError(resp.text)
