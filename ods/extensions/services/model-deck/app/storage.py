"""
Model Deck storage tiering — pure planning core (+ StorageWatcher shell).

``plan_move()`` and ``storage_decide()`` are PURE functions (no I/O, no
Settings): unit/location/world dicts in, plan/action dicts or GuardError
out. Mirrors app.arbiter's decide()/Watcher split — every storage guard
lives here and is unit-tested without threads or filesystems.

Guard philosophy (spec invariants 12-17): a loaded model, the litellm
default-route model (NEVER overridable — archiving the auto-reload target
would break the idle-burn fix one tier down), an unavailable source or
destination, a readonly destination, and an underspaced destination are all
refused with GuardError → HTTP 409.
"""

from app.engines import GuardError

EXTRA_PREFIX = "extra."


def _strip(name: str | None) -> str | None:
    return None if name is None else name.removeprefix(EXTRA_PREFIX)


def unit_in_use(unit: dict, world: dict) -> str | None:
    """Reason string when `unit` is in active use by an engine, else None.
    The default-route check is separate (never overridable) — see plan_move."""
    if unit["type"] == "gguf":
        if _strip(world["tenants"]["lemonade"]["model"]) == unit["name"]:
            return f"model {unit['name']!r} is currently loaded in lemonade"
        if _strip(world["default_route"]) == unit["name"]:
            return f"model {unit['name']!r} is the litellm default route — never movable"
    if unit["type"] == "comfy":
        comfy = world["tenants"]["comfyui"]
        if comfy["state"] != "idle" or (comfy["queue"] or 0) > 0:
            return f"ComfyUI is {comfy['state']} (queue {comfy['queue']}) — comfy files locked"
    return None


def plan_move(unit: dict, dest: dict, world: dict, active_unit_ids,
              dest_free_bytes: int | None, slack_bytes: int) -> dict:
    if unit["state"] == "unavailable":
        raise GuardError(f"unit {unit['id']!r} is on an unavailable location")
    if unit["state"] == "moving" or unit["id"] in active_unit_ids:
        raise GuardError(f"unit {unit['id']!r} already has a move in flight")

    reason = unit_in_use(unit, world)
    if reason:
        raise GuardError(reason)
    if unit["type"] == "gguf" and _strip(world["default_route"]) == unit["name"]:
        raise GuardError(
            f"model {unit['name']!r} is the litellm default route — never movable")

    if dest["name"] == unit["location"]:
        raise GuardError("destination equals source location")
    if not dest["available"]:
        raise GuardError(f"destination {dest['name']!r} is unavailable (marker missing — drive unmounted?)")
    if dest["readonly"]:
        raise GuardError(f"destination {dest['name']!r} is readonly")
    if dest_free_bytes is None or dest_free_bytes < unit["size"] + slack_bytes:
        raise GuardError(
            f"insufficient space on {dest['name']!r}: need {unit['size'] + slack_bytes}, have {dest_free_bytes}")

    return {"unit_id": unit["id"], "src_location": unit["location"],
            "dest_location": dest["name"], "bytes": unit["size"]}
