"""
Post-move engine notification hooks — make the engine SEE a file that just
arrived in its store.

lemonade (verified live, v10.2.0): registers store GGUFs only at startup;
no rescan endpoint exists. Hook = container restart via DockerCtl
(`ods-llama-server` is already on the park allowlist). DEFERRED with a
returned warning when a model is loaded — we never yank a loaded model to
register a file; the operator retries after unload or the idle TTL clears
the way.

comfyui (verified live): /api/models/{type} lists files per request — no
action needed. engine "none": nothing to notify.

stop() resilience (LIVE-VERIFIED): ods-llama-server ignores SIGTERM, so
even with DockerCtl's extended stop timeout the container can still be
mid-SIGKILL when the HTTP call times out client-side and raises
EngineError, while the container goes on to actually stop. If stop()
raises, the container is very likely just still stopping — wait out
Docker's grace period, then attempt start() anyway (the normal next step
regardless of how stop() finished). If start() then succeeds, the goal
state ("container restarting") is reached and we return None same as the
happy path. If start() also raises, that EngineError propagates — a real
double failure, not a timing race, should surface as a failure.
"""

import time

from app.engines import EngineError


def notify_engine(location: dict, deck: dict) -> str | None:
    if location["engine"] != "lemonade":
        return None
    if deck["lemonade"].status()["loaded"]:
        return ("lemonade has a model loaded — restart deferred; the new file "
                "registers after the next lemonade restart")
    container = deck["settings"].lemonade_container
    try:
        deck["dockerctl"].stop(container)
    except EngineError:
        time.sleep(10)
    deck["dockerctl"].start(container)
    return None
