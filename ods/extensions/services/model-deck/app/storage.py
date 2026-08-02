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

import threading

from app.engines import GuardError
from app.events import log_event
from app.sets import apply_in_progress

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

    # FAIL CLOSED: with litellm unreachable, world["default_route"] is None
    # because we couldn't ask — not because there is no default route. The
    # default-route guard is "never overridable", so an unverifiable one must
    # refuse rather than silently pass. (Non-gguf units can't be a route.)
    if unit["type"] == "gguf" and world.get("routes_known", True) is False:
        raise GuardError("litellm unreachable — cannot verify default route")

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
    # Same fail-closed rule as plan_move: an unverifiable default route
    # disqualifies every gguf candidate (silently — the watcher's job is to
    # stay quiet until it can act safely; plan_move is where an operator who
    # asked for a specific move gets told why).
    routes_known = world.get("routes_known", True) is not False
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
                and not (u["type"] == "gguf" and not routes_known)
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


class StorageWatcher:
    """Slow-cadence auto-tiering loop (60 s default): scan → describe →
    storage_decide → enqueue (auto) or suggest (manual). Never starts work
    while a set apply or a move job is in flight; ticks catch-all so the
    loop survives (arbiter.Watcher idiom)."""

    _DEDUP_KINDS = frozenset({"storage_suggestion", "storage_shortfall",
                              "storage-tick-error", "storage_skip"})

    def __init__(self, settings, location_store, catalog, storage_policy_store,
                 job_queue, world_fn, events_path):
        self._settings = settings
        self._locations = location_store
        self._catalog = catalog
        self._policy = storage_policy_store
        self._queue = job_queue
        self._world_fn = world_fn
        self._events_path = events_path
        self._interval = getattr(settings, "storage_watch_interval", 60.0)
        self._slack = getattr(settings, "storage_slack_bytes", 2_000_000_000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_event_keys: dict = {}  # {kind: key, ...}

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run,
                                        name="model-deck-storage-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            if self._stop.wait(self._interval):
                break

    def tick(self) -> None:
        if apply_in_progress() or self._queue.active():
            return
        try:
            units = self._catalog.scan()
            locations = self._locations.describe()
            world = self._world_fn()
            actions = storage_decide(units, locations, world, self._slack)
            auto = self._policy.get()["auto"]
            units_by_id = {u["id"]: u for u in units}
            locs_by_name = {loc["name"]: loc for loc in locations}
            suggestions: dict[str, int] = {}
            for action in actions:
                if action["type"] == "archive":
                    if auto:
                        self._enqueue(action, units_by_id, locs_by_name, world)
                    else:
                        loc = units_by_id[action["unit_id"]]["location"]
                        suggestions[loc] = suggestions.get(loc, 0) + 1
                elif action["type"] == "shortfall":
                    self._log("storage_shortfall", {"location": action["location"],
                                                    "missing_bytes": action["missing_bytes"]})
            for loc_name, count in sorted(suggestions.items()):
                self._log("storage_suggestion", {"location": loc_name, "candidates": count})
        except Exception as exc:  # noqa: BLE001 — supervisor loop survival
            self._log("storage-tick-error", {"error": str(exc)})

    def _enqueue(self, action, units_by_id, locs_by_name, world) -> None:
        unit = units_by_id[action["unit_id"]]
        dest = locs_by_name[action["dest"]]
        try:
            plan = plan_move(unit, dest, world, frozenset(),
                             dest["free_bytes"], self._slack)
        except GuardError as exc:
            self._log("storage_skip", {"unit": unit["id"], "reason": str(exc)})
            return
        self._queue.submit(plan, label="watermark archive")

    def _log(self, kind: str, detail: dict) -> None:
        detail_key = tuple(sorted(detail.items()))
        if kind in self._DEDUP_KINDS:
            # Extract entity: location or unit (the stable subject of the event)
            entity = detail.get("location") or detail.get("unit") or ""
            dedup_key = (kind, entity)
            last_key = self._last_event_keys.get(dedup_key)
            if detail_key == last_key:
                return
            self._last_event_keys[dedup_key] = detail_key
        log_event(self._events_path, kind, detail)
