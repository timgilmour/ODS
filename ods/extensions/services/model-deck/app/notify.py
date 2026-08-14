"""
Post-move engine notification hooks — make the engine SEE a file that just
arrived in its store.

lemonade (verified live, v10.2.0): registers store GGUFs only at startup;
no rescan endpoint exists. Hook = container restart via DockerCtl
(`ods-llama-server` is already on the park allowlist). DEFERRED per-resource
with a returned warning when THAT resource has a model loaded — we never
yank a loaded model to register a file; the operator retries after unload or
the idle TTL clears the way.

E1 Task 9 (T6 review class): the old code reached its single container via
`deck["settings"].lemonade_container` and a single `deck["lemonade"]`
boot-time alias — both engine-name assumptions that silently ignore every
resource but the one seeded at boot. This now iterates every resource in
the LIVE declaration (`deck["node_store"].get("local")["engines"]`, read
fresh on every call — never a boot-time copy, same posture as
`app.routers.build_world_snapshot`/`_declared_kind`) whose declared KIND
matches the destination location's `engine` field, and asks
`app.engine_kinds.ENGINE_KINDS[kind]` to do the per-kind work
(`restart_container(entry)` for the container name, `uses_gguf(obs)` for
"is a model loaded right now") — no engine name appears here. Two
lemonade-kind resources declared at once each restart their OWN container
independently; a resource with nothing loaded restarts even while a
SIBLING resource defers.

comfyui (verified live): /api/models/{type} lists files per request — no
action needed. Its adapter's `restart_container` returns None, so the loop
below skips it as a structural no-op rather than a special case; engine
"none" never matches any declared kind, so the loop simply iterates zero
entries.

stop() resilience (LIVE-VERIFIED): ods-llama-server ignores SIGTERM, so
even with DockerCtl's extended stop timeout the container can still be
mid-SIGKILL when the HTTP call times out client-side and raises
EngineError, while the container goes on to actually stop. If stop()
raises, the container is very likely just still stopping — wait out
Docker's grace period, then attempt start() anyway (the normal next step
regardless of how stop() finished). If start() then succeeds, the goal
state ("container restarting") is reached for that resource and it
contributes no warning, same as the happy path. If start() also raises,
that EngineError propagates — a real double failure, not a timing race,
should surface as a failure (and, per the house "let it crash" policy,
abandons any remaining resources still queued in the loop rather than
guessing at a partial-failure summary).
"""

import time

from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError


def notify_engine(location: dict, deck: dict) -> str | None:
    local = deck["node_store"].get("local")
    engines = local.get("engines", []) if local is not None else []
    deferred: list[str] = []
    for entry in engines:
        if entry["kind"] != location["engine"]:
            continue
        adapter = ENGINE_KINDS[entry["kind"]]
        container = adapter.restart_container(entry)
        if container is None:
            continue
        resource = entry["resource"]
        client = deck["local_clients"].client_for(resource)
        status = client.status()
        obs = {"state": "loaded" if status["loaded"] else "unloaded",
               "model": status["loaded"]}
        if adapter.uses_gguf(obs) is not None:
            deferred.append(resource)
            continue
        try:
            deck["dockerctl"].stop(container)
        except EngineError:
            time.sleep(10)
        deck["dockerctl"].start(container)
    if deferred:
        return "; ".join(
            f"{resource} has a model loaded — restart deferred; the new "
            "file registers after the next restart"
            for resource in deferred
        )
    return None
