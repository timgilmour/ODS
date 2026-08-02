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
    """Reason string if unit is in active use by an engine, else None.

    Covers three cases: gguf currently loaded in lemonade, gguf being litellm's
    default route (never overridable — plan-level invariant), and comfy units
    while ComfyUI is busy or queue unknown. Callers (plan_move, storage_decide)
    treat non-None reason as GuardError refusal."""
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


def storage_decide(units: list[dict], locations: list[dict], world: dict,
                   slack_bytes: int) -> list[dict]:
    """Pure watermark-eviction rules. DELIBERATE ASYMMETRY vs VRAM healing:
    partial relief still helps a disk, so this archives what it can and
    reports the remainder as a shortfall — it never feasibility-vetoes."""
    by_name = {loc["name"]: loc for loc in locations}
    default_route = _strip(world["default_route"])
    actions: list[dict] = []
    planned_to_dest: dict[str, int] = {}

    for loc in locations:
        if loc["role"] != "hot" or not loc["available"] or not loc["watermark_gb"]:
            continue
        watermark = int(loc["watermark_gb"] * 1e9)
        needed = watermark - loc["free_bytes"]
        if needed <= 0:
            continue

        dest = by_name.get(loc["archive_to"] or "")
        candidates = []
        if dest is not None and dest["available"] and not dest["readonly"]:
            candidates = [
                u for u in units
                if u["location"] == loc["name"] and u["state"] == "resident"
                and not u["pinned"] and unit_in_use(u, world) is None
                and not (u["type"] == "gguf" and u["name"] == default_route)
            ]
            candidates.sort(key=lambda u: (u["last_used"] is not None,
                                           u["last_used"] or 0.0, u["mtime"]))
        for u in candidates:
            if needed <= 0:
                break
            already = planned_to_dest.get(dest["name"], 0)
            if dest["free_bytes"] - already < u["size"] + slack_bytes:
                continue                    # dest can't take this one; try smaller
            actions.append({"type": "archive", "unit_id": u["id"],
                            "dest": dest["name"], "bytes": u["size"]})
            planned_to_dest[dest["name"]] = already + u["size"]
            needed -= u["size"]
        if needed > 0:
            actions.append({"type": "shortfall", "location": loc["name"],
                            "missing_bytes": needed})
    return actions
