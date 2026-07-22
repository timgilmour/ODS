"""
Model Deck config sets — schema, slug store, diff planner, apply engine.

A *config set* is a named, user-authored bundle of desired state that a human
saves once and re-applies with one click ("Chat mode", "Image session", ...).
It has two halves:

* **durable** — the litellm default route (which model requests go to by
  default). Changing it means asking the ODS host agent to *activate* a model,
  a heavy, exclusive operation.
* **ephemeral** — per-tenant load/park intent (lemonade loaded/unloaded,
  comfyui freed or left alone, hipfire running/parked). Omitting a tenant's
  subsection means "don't touch it".

plus an optional **policy_overrides** blob handed verbatim to the arbiter's
``PolicyStore.put`` (its own validation is the gate).

This is the ONE module in the service that uses pydantic: config sets are
user-authored JSON, so they get schema validation at the door. Everything else
in Model Deck consumes already-trusted internal dicts.

Three layers, cleanly separated for testability:

* ``ConfigSet`` + ``SetStore`` — the schema and its on-disk CRUD (slugged
  filenames, atomic writes, same temp+os.replace idiom as registry/policy).
* ``plan_apply(cfgset, world)`` — a PURE diff of a set against one world
  snapshot, emitting only the steps that actually change reality, in a fixed
  safety order (evictions first, loads last). No I/O, no clients.
* ``apply(cfgset, ...)`` — the imperative shell. Serialized under a module
  lock (two applies never interleave real evictions/activations on a live
  box). It FIRST snapshots pre-apply reality as the ``_previous`` revert set,
  then executes the plan step by step, halting on the first failure with an
  exact report, logging every step.

Why _previous is captured first and unconditionally: apply performs real,
partially-irreversible actions (unloading models, parking containers,
activating a new default route). If an apply half-completes and the user wants
out, the ``_previous`` set is their one-click way back to the state the box was
in the instant before. Its durable half records the old default route but with
``activate_model_id=None`` (world snapshots don't carry the catalog id needed to
re-activate), so a durable revert may warn "unavailable" — the ephemeral revert
always works.
"""

import os
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from app.engines import BusyError, EngineError, GuardError
from app.events import log_event

# Reserved on-disk slug for the auto-captured pre-apply snapshot. Written only
# by apply(); user sets are forbidden from slugging to it (or to "previous",
# which is what its display name collapses to).
RESERVED_SLUG = "_previous"
PREVIOUS_NAME = "· previous"
_RESERVED_SLUGS = frozenset({"_previous", "previous"})

_PREVIOUS_NOTES = (
    "Auto-snapshot of the box's state captured just before the last apply, so "
    "you can revert in one click. Note: the durable (default-route) revert may "
    "be unavailable — the snapshot records the previous model's name but not "
    "the catalog id the host agent needs to re-activate it."
)


# ===========================================================================
# Schema (pydantic — user-authored JSON)
# ===========================================================================


class Durable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # litellm-side name (e.g. "extra.Qwen3.5-27B-Q4_K_M.gguf") — used for
    # DIFFING against world.default_route.
    default_route_model: str
    # ODS catalog id the host agent needs for ACTIVATE. The UI populates it;
    # may be None (then a durable change can only warn, not act).
    activate_model_id: str | None = None


class LemonadeEphemeral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["loaded", "unloaded"]


class ComfyuiEphemeral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["free", "leave"]
    reserve_gb: int = 24  # informational (UI budgeting), not enforced at apply


class HipfireEphemeral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["running", "parked"]


class Ephemeral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # An omitted subsection means "don't touch that tenant".
    lemonade: LemonadeEphemeral | None = None
    comfyui: ComfyuiEphemeral | None = None
    hipfire: HipfireEphemeral | None = None


class ConfigSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    notes: str = ""
    durable: Durable | None = None
    ephemeral: Ephemeral | None = None
    # Passed to PolicyStore.put verbatim at apply time; its validation is the gate.
    policy_overrides: dict | None = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must be non-empty")
        return value


# ===========================================================================
# Slug + store
# ===========================================================================


def slugify(name: str) -> str:
    """Lowercase; alnum runs joined by single dashes; leading/trailing dashes
    trimmed (e.g. "Image session" -> "image-session"). May return "" for a
    name with no alphanumerics at all."""
    return "-".join(re.findall(r"[a-z0-9]+", name.lower()))


class SetStore:
    """CRUD over ``<slug>.json`` files in ``dir``. Atomic writes (temp +
    os.replace), same idiom as registry/policy."""

    def __init__(self, dir: Path):  # noqa: A002 - matches the brief's signature
        self._dir = dir

    def _path(self, slug: str) -> Path:
        return self._dir / f"{slug}.json"

    def _write(self, slug: str, cfgset: ConfigSet) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(slug)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(cfgset.model_dump_json(indent=2))
        os.replace(tmp, path)

    def save(self, cfgset: ConfigSet) -> str:
        """Persist a user set under the slug of its name; returns the slug.

        Rejects (ValueError) a name that produces an empty slug or one that
        collides with the reserved ``_previous``/``previous`` revert slot.
        """
        slug = slugify(cfgset.name)
        if not slug:
            raise ValueError(f"name {cfgset.name!r} produces an empty slug")
        if slug in _RESERVED_SLUGS:
            raise ValueError(
                f"slug {slug!r} is reserved for the auto-captured revert snapshot"
            )
        self._write(slug, cfgset)
        return slug

    def save_previous(self, cfgset: ConfigSet) -> str:
        """Write the pre-apply revert snapshot under the reserved slug.
        Bypasses save()'s reserved-slug guard by design — apply() is the sole
        caller."""
        self._write(RESERVED_SLUG, cfgset)
        return RESERVED_SLUG

    def get(self, slug: str) -> ConfigSet | None:
        """The set stored under ``slug``, or None if there is none."""
        try:
            text = self._path(slug).read_text()
        except FileNotFoundError:
            return None
        return ConfigSet.model_validate_json(text)

    def list(self) -> list[ConfigSet]:
        """All stored sets, sorted by name (includes ``_previous`` if present)."""
        if not self._dir.exists():
            return []
        sets = [
            ConfigSet.model_validate_json(path.read_text())
            for path in self._dir.glob("*.json")
        ]
        return sorted(sets, key=lambda cfgset: cfgset.name)

    def delete(self, slug: str) -> None:
        """Remove the set stored under ``slug`` (no-op if absent)."""
        self._path(slug).unlink(missing_ok=True)


# ===========================================================================
# plan_apply — pure diff
# ===========================================================================


def plan_apply(cfgset: ConfigSet, world: dict) -> list[dict]:
    """Diff ``cfgset`` against ``world`` -> ordered list of step dicts.

    PURE: no I/O, no client calls. Emits only steps that change reality.
    Order: evictions (unload/free) -> park -> activate -> resume -> load ->
    policy_patch, with ``warn`` steps interleaved where they are generated.
    """
    tenants = world["tenants"]
    lem_world = tenants["lemonade"]
    comfy_world = tenants["comfyui"]
    hip_world = tenants["hipfire"]

    eph = cfgset.ephemeral
    durable = cfgset.durable

    lem_desired = eph.lemonade.state if eph and eph.lemonade else None
    comfy_desired = eph.comfyui.state if eph and eph.comfyui else None
    hip_desired = eph.hipfire.state if eph and eph.hipfire else None

    steps: list[dict] = []

    # --- Evictions (unload / free) -----------------------------------------
    if lem_desired == "unloaded" and lem_world["state"] == "loaded":
        steps.append({"step": "unload_lemonade", "model": lem_world["model"]})

    if comfy_desired == "free":
        # Only free when we can confirm the queue is empty (== 0). A busy
        # queue OR an unknown/None queue -> skip and be honest about it; we
        # never yank VRAM out from under a running generation.
        if comfy_world["queue"] == 0:
            steps.append({"step": "free_comfyui"})
        else:
            steps.append({"step": "warn", "reason": "comfyui-busy-skipped"})

    # --- Park + Activate ---------------------------------------------------
    # Normally park comes BEFORE activate (free the GPU, then re-point the
    # route). But when the activate MOVES the default route off hipfire and
    # the route currently targets hipfire, parking first would yank the GPU
    # out from under the still-default hipfire model — so park AFTER activate
    # in that one case. (hipfire.park()'s own guard also refuses to park while
    # it serves the default route; ordering activate first is what lets the
    # park succeed.)
    park_step = None
    if hip_desired == "parked" and hip_world["state"] in ("running", "loading"):
        park_step = {"step": "park_hipfire"}

    activate_step = None
    if durable is not None and durable.default_route_model != world["default_route"]:
        if durable.activate_model_id is not None:
            activate_step = {"step": "activate", "model_id": durable.activate_model_id}
        else:
            activate_step = {"step": "warn", "reason": "durable-revert-unavailable"}

    route_on_hipfire = (
        hip_world["model"] is not None and world["default_route"] == hip_world["model"]
    )
    park_after_activate = (
        park_step is not None
        and activate_step is not None
        and activate_step["step"] == "activate"
        and route_on_hipfire
    )

    if park_after_activate:
        steps.append(activate_step)
        steps.append(park_step)
    else:
        if park_step is not None:
            steps.append(park_step)
        if activate_step is not None:
            steps.append(activate_step)

    # --- Resume ------------------------------------------------------------
    if hip_desired == "running" and hip_world["state"] == "parked":
        steps.append({"step": "resume_hipfire"})

    # --- Load --------------------------------------------------------------
    if lem_desired == "loaded" and lem_world["state"] == "unloaded":
        model = (
            durable.default_route_model if durable is not None else world["default_route"]
        )
        if model is None:
            steps.append({"step": "warn", "reason": "no-model-to-load"})
        else:
            steps.append({"step": "load_lemonade", "model": model})

    # --- Policy patch (always, not diffed) ---------------------------------
    if cfgset.policy_overrides is not None:
        steps.append({"step": "policy_patch", "policies": cfgset.policy_overrides})

    return steps


# ===========================================================================
# apply — imperative shell (serialized)
# ===========================================================================

# Non-reentrant: a second apply blocks until the first releases. Two applies
# must never interleave real evictions/parks/activations on a live box.
_apply_lock = threading.Lock()


def apply_in_progress() -> bool:
    """True if an apply currently holds the module lock, WITHOUT acquiring it.

    A non-blocking peek the arbiter watcher uses to yield a tick to an
    in-flight set apply — the two must never interleave real evictions/loads
    on the live box.
    """
    return _apply_lock.locked()

# The set of exceptions that halt an apply mid-plan (vs. crashing it). Each is
# a known, meaningful "this step could not proceed" signal from a client.
_HALT_EXCEPTIONS = (GuardError, EngineError, BusyError, ValueError)


def apply(
    cfgset: ConfigSet,
    *,
    world: dict,
    lemonade,
    comfy,
    hipfire,
    hostagent,
    policy_store,
    store: SetStore,
    events_path: Path,
    heal_suppressor=None,
    force: bool = False,
) -> dict:
    """Execute ``cfgset`` against the live box, serialized under a module lock.

    ``heal_suppressor`` (optional; None tolerated) is the arbiter's shared
    ``HealSuppressor``: an ``unload_lemonade`` step arms it so contention
    healing can't revert this deliberate unload, and a ``load_lemonade`` step
    clears it. None (e.g. in unit tests without the arbiter) simply skips that
    coordination.

    ``force=True`` skips the hipfire conversation-guard (both the pre-veto
    and the per-step rechecks) for an operator overriding an abandoned
    conversation; it does NOT skip park()'s litellm route guard.

    Returns an ApplyReport dict:
        {"completed": [<step>, ...], "failed": <step>|None,
         "error": <str>|None, "warnings": [<reason>, ...]}
    """
    with _apply_lock:
        return _run_apply(
            cfgset,
            world=world,
            lemonade=lemonade,
            comfy=comfy,
            hipfire=hipfire,
            hostagent=hostagent,
            policy_store=policy_store,
            store=store,
            events_path=events_path,
            heal_suppressor=heal_suppressor,
            force=force,
        )


def _run_apply(
    cfgset,
    *,
    world,
    lemonade,
    comfy,
    hipfire,
    hostagent,
    policy_store,
    store,
    events_path,
    heal_suppressor=None,
    force=False,
) -> dict:
    steps = plan_apply(cfgset, world)

    # Veto BEFORE any mutation (and before the _previous snapshot — a refused
    # apply changes nothing, so there is nothing to revert): a plan that would
    # park hipfire or flip the durable route (litellm restart severs in-flight
    # streams; a hipfire-direction activate recreates the container outright)
    # is refused while a hipfire conversation is live. GuardError propagates
    # to the route -> 409. The per-step rechecks below cover the gap between
    # this veto and the step actually running.
    if not force and any(s["step"] in ("park_hipfire", "activate") for s in steps):
        try:
            hipfire.ensure_not_busy(f"apply set {cfgset.name!r}")
        except _HALT_EXCEPTIONS:
            log_event(events_path, "apply-vetoed", {"name": cfgset.name})
            raise

    log_event(events_path, "apply-start", {"name": cfgset.name})

    # FIRST mutation: capture pre-apply reality as the one-click revert set,
    # before any step touches the box.
    store.save_previous(_previous_set(world))

    report: dict = {"completed": [], "failed": None, "error": None, "warnings": []}

    for step in steps:
        name = step["step"]

        if name == "warn":
            report["warnings"].append(step["reason"])
            log_event(events_path, "apply-warn", {"reason": step["reason"]})
            continue

        try:
            _execute_step(
                step,
                lemonade,
                comfy,
                hipfire,
                hostagent,
                policy_store,
                heal_suppressor,
                force=force,
            )
        except _HALT_EXCEPTIONS as exc:
            report["failed"] = step
            report["error"] = str(exc)
            log_event(
                events_path,
                "apply-end",
                {"outcome": "failed", "step": name, "error": str(exc)},
            )
            return report

        log_event(events_path, name, {k: v for k, v in step.items() if k != "step"})
        report["completed"].append(step)

    log_event(events_path, "apply-end", {"outcome": "ok"})
    return report


def _execute_step(
    step, lemonade, comfy, hipfire, hostagent, policy_store, heal_suppressor=None,
    force=False,
) -> None:
    name = step["step"]
    if name == "unload_lemonade":
        lemonade.unload(step["model"])
        # Deliberate unload: arm suppression so the arbiter doesn't heal it back.
        if heal_suppressor is not None:
            heal_suppressor.note_deck_unload()
    elif name == "load_lemonade":
        lemonade.load(step["model"])
        # Deliberate load: the model is wanted resident, so clear suppression.
        if heal_suppressor is not None:
            heal_suppressor.clear()
    elif name == "free_comfyui":
        comfy.free()
    elif name == "park_hipfire":
        hipfire.park(force=force)
    elif name == "resume_hipfire":
        hipfire.resume()
    elif name == "activate":
        # Recheck at execution time: a request may have landed on hipfire
        # since the pre-veto (activation restarts litellm, severing in-flight
        # streams; a hipfire-direction activate recreates the container).
        if not force:
            hipfire.ensure_not_busy(f"activate {step['model_id']!r}")
        hostagent.activate(step["model_id"])
    elif name == "policy_patch":
        # Merge each field-partial per-tenant override onto the current stored
        # values before writing, so {"comfyui": {"priority": 90}} keeps the
        # other two fields intact. put() still validates the merged records
        # (an unknown tenant merges onto {} and is rejected there).
        current = policy_store.get()
        merged = {}
        for tenant, override in step["policies"].items():
            base = dict(current.get(tenant, {}))
            base.update(override)
            merged[tenant] = base
        policy_store.put(merged)
    else:  # pragma: no cover - plan_apply is the sole producer of steps
        raise AssertionError(f"unknown step {name!r}")


def _previous_set(world: dict) -> ConfigSet:
    """Build the ``· previous`` revert snapshot from pre-apply world reality.

    ephemeral mirrors current load/park state (comfyui is always "leave" — a
    freed VRAM cache can't be meaningfully un-freed); durable records the old
    default route with activate_model_id=None (world carries no catalog id).
    """
    tenants = world["tenants"]
    lem_state = "loaded" if tenants["lemonade"]["state"] == "loaded" else "unloaded"
    hip_state = (
        "running" if tenants["hipfire"]["state"] in ("running", "loading") else "parked"
    )
    default_route = world["default_route"]
    durable = (
        {"default_route_model": default_route, "activate_model_id": None}
        if default_route
        else None
    )
    return ConfigSet(
        name=PREVIOUS_NAME,
        notes=_PREVIOUS_NOTES,
        durable=durable,
        ephemeral={
            "lemonade": {"state": lem_state},
            "comfyui": {"state": "leave"},
            "hipfire": {"state": hip_state},
        },
    )
