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
"""


class EngineError(Exception):
    """An engine HTTP call failed: non-2xx response or transport error."""


class GuardError(Exception):
    """A safety guard refused an engine operation (e.g. ComfyUI queue non-empty)."""
