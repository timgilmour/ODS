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
contributes no warning, same as the happy path.

E1 Task 9 review fix — a restart failure (start() still raising after the
stop-retry, or a plain start() failure) must not fail INVISIBLY, and one
resource's failure must not silently swallow whether its SIBLINGS ever got
attempted. Chosen semantic: ISOLATE per resource, like
`app.engine_kinds`' `execute_unload`/`execute_load` and `app.arbiter`'s
`_execute_restore` already do for every other per-resource actuation loop
in this codebase (a raise from one action must not abort the rest) — NOT
`app.sets.apply`'s halt-on-first-step precedent, which fits a single
operator-authored PLAN whose later steps may depend on earlier ones; these
are independent resources sharing nothing but a destination store, so a
failed restart on one has no bearing on whether a sibling's restart can
still succeed. So: every declared entry still gets its OWN restart
attempt regardless of an earlier entry's failure (no rollback, no
transaction across resources — Let It Crash: a resource that already
restarted successfully STAYS restarted even if a sibling then fails), but
the overall call still ends by raising (never swallowing a failure into a
benign warning return) once every entry has been attempted, so the
caller's existing failure path (an app-wide EngineError->502 handler, or a
job's post-move failure) still fires. Each failure is logged as its own
resource+container-scoped event (`notify-restart-failed`) BEFORE moving on
— the same "resource" field app.sets.apply's failing-step event and
app.engine_kinds' unload-failed/load-failed events already carry, for the
identical reason: two same-kind resources failing identically must stay
distinguishable in the log, not collapse into one anonymous line. When
more than one entry fails, the FIRST failure encountered is what
ultimately propagates (deterministic, and every failure — first or not —
already has its own logged event regardless).

Final-review item 3a (E1): the per-resource catch around `_restart` above
now names `GuardError` alongside `EngineError` — a container outside
`settings.park_allowlist` makes `DockerCtl.stop()`/`start()` raise
`GuardError` (app/engines/docker_ctl.py:197-199), and `GuardError` is deliberately
NOT an `EngineError` subclass (app/engines/__init__.py:30-38: "Callers that
want to distinguish 'engine is broken' from 'guard tripped' need these to
be unrelated exception types"). An `EngineError`-only catch here let that
refusal escape the loop entirely — aborting every SIBLING's restart attempt
and skipping its own `notify-restart-failed` event — the exact isolation
failure this whole fix exists to prevent, just for a different exception
type than the one it was written against.
"""

import time

from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError, GuardError
from app.events import log_event


def notify_engine(location: dict, deck: dict) -> str | None:
    local = deck["node_store"].get("local")
    engines = local.get("engines", []) if local is not None else []
    deferred: list[str] = []
    failure: EngineError | GuardError | None = None
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
            _restart(deck, container)
        except (EngineError, GuardError) as exc:
            log_event(deck["events_path"], "notify-restart-failed",
                      {"resource": resource, "container": container,
                       "error": str(exc)})
            if failure is None:
                failure = exc
            continue
    if failure is not None:
        raise failure
    if deferred:
        return "; ".join(
            f"{resource} has a model loaded — restart deferred; the new "
            "file registers after the next restart"
            for resource in deferred
        )
    return None


def _restart(deck: dict, container: str) -> None:
    try:
        deck["dockerctl"].stop(container)
    except EngineError:
        time.sleep(10)
    deck["dockerctl"].start(container)
