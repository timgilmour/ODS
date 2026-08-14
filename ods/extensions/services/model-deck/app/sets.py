"""
Model Deck config sets — schema, slug store, diff planner, apply engine.

A *config set* is a named, user-authored bundle of desired state that a human
saves once and re-applies with one click ("Chat mode", "Image session", ...).
It has two halves:

* **durable** — the litellm default route (which model requests go to by
  default). Changing it means asking the ODS host agent to *activate* a model,
  a heavy, exclusive operation.
* **ephemeral** — per-RESOURCE load/park intent (E1 Task 8: generalized past
  the old fixed lemonade/comfyui/hipfire triple — any number of any-kind
  DECLARED resources, keyed by resource name). Omitting a resource's entry
  means "don't touch it".

plus an optional **policy_overrides** blob handed verbatim to the arbiter's
``PolicyStore.put`` (its own validation is the gate).

This is the ONE module in the service that uses pydantic: config sets are
user-authored JSON, so they get schema validation at the door. Everything else
in Model Deck consumes already-trusted internal dicts.

Three layers, cleanly separated for testability:

* ``ConfigSet`` + ``SetStore`` — the schema and its on-disk CRUD (slugged
  filenames, atomic writes, same temp+os.replace idiom as registry/policy).
  ``SetStore`` also upgrades a pre-Task-8 on-disk file to the current shape
  on LOAD (``upgrade_legacy_set``, pure) — ``save`` always writes the new
  shape, so a file is upgraded at most once.
* ``plan_apply(cfgset, world)`` — a PURE diff of a set against one world
  snapshot, emitting only the steps that actually change reality, in a fixed
  safety order (evictions first, loads last). No I/O, no clients.
* ``apply(cfgset, ...)`` — the imperative shell. Serialized under
  ``app.actuation.LOCK`` (task 6) — the ONE process-wide actuation lock also
  held by the watcher tick's actuation phase and the pull-through completion
  hook, so no two of those three ever interleave real evictions/loads on a
  live box, not just two applies against each other. It FIRST snapshots
  pre-apply reality as the ``_previous`` revert set, then records the set's
  DECLARED goals as intent (``_record_goal_intents``, before any step
  actuates), then executes the plan step by step, halting on the first
  failure with an exact report, logging every step. Actuation resolves each
  step's client per-resource, live, through ``local_clients.client_for``
  (E1 Task 8, same conversion Task 7 made for app.routers.control) — never a
  boot-time per-engine alias, which would silently keep acting on a resource's
  OLD connection after a live declaration edit (Task 10).

Why _previous is captured first and unconditionally: apply performs real,
partially-irreversible actions (unloading models, parking containers,
activating a new default route). If an apply half-completes and the user wants
out, the ``_previous`` set is their one-click way back to the state the box was
in the instant before. Its durable half records the old default route but with
``activate_model_id=None`` (world snapshots don't carry the catalog id needed to
re-activate), so a durable revert may warn "unavailable" — the ephemeral revert
always works.

E1 Task 8 generalization, in one paragraph: the old schema had exactly three
fixed sub-sections (``lemonade``/``comfyui``/``hipfire``, each with its own
state vocabulary) and steps named ``<verb>_<kind>`` (``unload_lemonade``,
``free_comfyui``, ...). Both were kind-name literals baked into the shape
itself — spec §8 forbids that outside ``app.engine_kinds``. The new schema is
``Ephemeral = {resources: {resource: ResourceDesired}}`` and steps are
``{"step": <verb>, "resource": <name>, ...}``: verb-generic, resource-keyed,
kind-agnostic. Validity (which ``desired`` values and the optional ``model``
field a resource's declared KIND actually accepts) is checked by VERB
membership in ``app.engine_kinds.ENGINE_KINDS[kind].human_verbs()`` — never a
hardcoded kind name — see ``Ephemeral``'s validator below. The three legacy
field names survive in exactly one place, ``upgrade_legacy_set`` (the
disclosed residue spec §8's Global Constraints allow), which converts an
old-shape on-disk file to the new shape on load.
"""

import copy
import json
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from app import actuation
from app.engine_kinds import ENGINE_KINDS
from app.engines import BusyError, EngineError, GuardError
from app.events import log_event
from app.observe import local_key
from app.settings_store import KINDS, NAMESPACES, empty_store
from app.store_io import write_atomic

# Reserved on-disk slug for the auto-captured pre-apply snapshot. Written only
# by apply(); user sets are forbidden from slugging to it (or to "previous",
# which is what its display name collapses to).
# Lemonade namespaces store GGUFs as "extra.<filename>"; catalog units carry
# the bare filename.
_EXTRA_PREFIX = "extra."

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


# desired -> the human_verbs() a resource's declared kind must expose at
# least one of, for that desired value to be meaningful for it. Module-level
# and VERB-keyed on purpose (spec §8): this is the one place validity is
# decided, and it never spells a kind name to do it — only the verb tokens
# app.engine_kinds.ENGINE_KINDS[...].human_verbs() already returns.
_DESIRED_VERBS: dict[str, frozenset[str]] = {
    "loaded": frozenset({"load", "resume"}),
    "unloaded": frozenset({"unload"}),
    "parked": frozenset({"park"}),
    "freed": frozenset({"free"}),
}


class ResourceDesired(BaseModel):
    model_config = ConfigDict(extra="forbid")

    desired: Literal["loaded", "unloaded", "parked", "freed"]
    # The model a "loaded" goal means, when the set knows it. Stamped by
    # _previous_set (revert must reload what WAS loaded, not the route
    # default); None = pre-existing behavior (durable/default-route pick).
    # Meaningful only for a load-verb (lemonade-kind) resource — see
    # Ephemeral's kind validator below for why any other kind carrying one
    # is refused rather than silently ignored.
    model: str | None = None


class Ephemeral(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # resource -> desired state. An ABSENT resource means "don't touch it"
    # (E1 Task 8: replaces the old fixed lemonade/comfyui/hipfire
    # sub-sections — any number of any-kind DECLARED resources now, keyed by
    # name, not by kind).
    resources: dict[str, ResourceDesired] = {}

    @model_validator(mode="after")
    def _validate_against_declared_kinds(self, info) -> "Ephemeral":
        """``desired``/``model`` validity depends on the resource's DECLARED
        KIND — static pydantic schema alone cannot know that. The live
        declaration is supplied by the CALLER as validation context
        (``{"kinds": {resource: kind}}``); today the only caller that has
        one is ``app.routers.sets.create_set`` (design choice recorded in
        the E1 Task 8 commit: the kinds DATA comes from the router, which is
        the one seam holding a real declaration to check against, but the
        cross-check itself runs HERE, inside pydantic, so a caller with a
        declaration gets a real ``pydantic.ValidationError`` -> 422 the
        app-wide ``ValueError`` handler already maps, not a second ad hoc
        gate).

        No context — every ``plan_apply``/``_record_goal_intents``/pure-diff
        unit test construction in this file, ``_previous_set`` (built from
        already-consistent live world state), and every set loaded through
        ``upgrade_legacy_set`` (predates this check entirely) — skips this
        validator outright. A resource present in ``resources`` but ABSENT
        from a SUPPLIED ``kinds`` mapping is refused too: declaring desire
        for something the caller can't identify the kind of is exactly the
        ambiguous input ``literal-declared-inputs`` says to refuse, not
        guess past.

        Kind-agnostic on purpose (spec §8, no engine-kind-name literal
        outside app.engine_kinds): validity is VERB membership in
        ``ENGINE_KINDS[kind].human_verbs()`` via the module-level
        ``_DESIRED_VERBS`` table (verb tokens only) — never a hardcoded kind
        name. ``model`` is accepted only when the kind supports "load"
        (today: lemonade-kind alone) — the brief's "model only for
        lemonade-kind" rule, expressed the same verb-generic way.
        """
        kinds = (info.context or {}).get("kinds") if info.context else None
        if kinds is None:
            return self
        for resource, rd in self.resources.items():
            kind = kinds.get(resource)
            if kind is None or kind not in ENGINE_KINDS:
                raise ValueError(f"{resource!r} is not a currently declared engine")
            verbs = ENGINE_KINDS[kind].human_verbs()
            if not (_DESIRED_VERBS[rd.desired] & verbs):
                raise ValueError(
                    f"{resource!r} ({kind}) cannot be declared {rd.desired!r}"
                )
            if rd.model is not None and "load" not in verbs:
                raise ValueError(f"{resource!r} ({kind}) does not accept a model")
        return self


class ConfigSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    notes: str = ""
    durable: Durable | None = None
    ephemeral: Ephemeral | None = None
    # Passed to PolicyStore.put verbatim at apply time; its validation is the gate.
    policy_overrides: dict | None = None
    # The ENTIRE settings store (app.settings_store.SettingsStore.get()'s
    # shape), captured at save time (design decision 6, Tim, 2026-08-07) --
    # a real ConfigSet has no placements, so "reproducible" means the whole
    # tenant-state recipe, not a per-scope subset. None for every set saved
    # before this field existed (and the reserved ``_previous`` slot, when
    # no settings_store was wired into the apply that captured it) -- those
    # plan and apply exactly as they did pre-Task-9 (see plan_apply below).
    settings_snapshot: dict | None = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must be non-empty")
        return value


# ===========================================================================
# Legacy (pre-Task-8) shape upgrade — pure, LOAD-time only
# ===========================================================================

# The three legacy field names — every set saved before E1 Task 8 used
# exactly these three fixed fields, so resource == kind name is a FACT about
# every old file, not an assumption. This is the ONE allowed engine-kind-name
# residue outside app.engine_kinds (spec §8's Global Constraints) — it
# appears nowhere else in this module.
_LEGACY_KIND_NAMES = ("lemonade", "comfyui", "hipfire")

# (legacy field name, its old "state" value) -> the new "desired" value.
# ("comfyui", "leave") has no entry: omission itself IS "leave" in the new
# schema, so that entry is DROPPED, never mapped — see upgrade below.
_LEGACY_STATE_TO_DESIRED: dict[tuple[str, str], str] = {
    ("lemonade", "loaded"): "loaded",
    ("lemonade", "unloaded"): "unloaded",
    ("comfyui", "free"): "freed",
    ("hipfire", "running"): "loaded",
    ("hipfire", "parked"): "parked",
}


def upgrade_legacy_set(raw: dict) -> dict:
    """Pure function: a stored set's raw dict, upgraded to the current
    ``ephemeral: {resources: {...}}`` shape if it is still in the pre-Task-8
    per-kind shape (top-level ``lemonade``/``comfyui``/``hipfire`` keys
    inside ``ephemeral``). Already-current-shape or ephemeral-absent input
    passes through UNCHANGED — this function's own job is exactly the outer
    per-kind -> per-resource restructuring (plus the old ``state`` field's
    rename to ``desired`` and its per-kind value translation); a raw dict
    whose per-resource entries already carry ``desired`` has nothing left
    for it to do to them.

    Called on LOAD only (``SetStore._scan``/``get``); ``SetStore.save``
    always writes the new shape, so a set is upgraded at most once — the
    NEXT save silently drops the legacy shape for good.

    ``raw`` is untrusted disk content — a top level that parsed as JSON but
    isn't an object (``"hello"``, ``[1, 2]``) is passed straight through
    UNCHANGED rather than crashing on ``.get`` ([c44] re-opened, T8 review
    I1): this function's job is reshaping a dict, not validating one: a
    non-dict is exactly as "not the legacy per-kind shape" as a dict with
    no ephemeral key, and letting it through here means
    ``ConfigSet.model_validate`` is what raises the named ``ValidationError``
    below (``SetStore.get``/``_scan`` already turn that into "unreadable",
    same as before this function existed) — not a bare ``AttributeError``
    escaping this one.
    """
    if not isinstance(raw, dict):
        return raw
    eph = raw.get("ephemeral")
    if not isinstance(eph, dict) or "resources" in eph or not (
        set(eph) & set(_LEGACY_KIND_NAMES)
    ):
        return raw

    resources: dict = {}
    for kind in _LEGACY_KIND_NAMES:
        entry = eph.get(kind)
        if entry is None:
            continue
        entry = dict(entry)
        # reserve_gb was comfyui-only, informational (UI budgeting), never
        # enforced at apply — dropped, not carried into ResourceDesired.
        entry.pop("reserve_gb", None)
        state = entry.pop("state", None)
        if state is not None:
            desired = _LEGACY_STATE_TO_DESIRED.get((kind, state))
            if desired is None:
                # ("comfyui", "leave"): omission IS "leave" now.
                continue
            entry["desired"] = desired
        resources[kind] = entry

    new_eph = {k: v for k, v in eph.items() if k not in _LEGACY_KIND_NAMES}
    new_eph["resources"] = resources
    return {**raw, "ephemeral": new_eph}


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
        # Narrow but real [T9b sweep]: _write uses a fixed `.tmp` path PER
        # SLUG, so two HTTP threads writing the SAME set name race that path
        # and the loser's os.replace raises FileNotFoundError into a route.
        # Different slugs cannot collide — the tmp name carries the slug.
        #
        # Covers save(), replace() and delete() — every path that writes a
        # slug file from an HTTP thread.
        #
        # save_previous() deliberately does NOT take it: apply() is its sole
        # caller and already holds app.actuation.LOCK, so it is structurally
        # serialized. Taking this lock there would be harmless but would
        # imply a race that cannot happen.
        self._lock = threading.Lock()

    def _path(self, slug: str) -> Path:
        return self._dir / f"{slug}.json"

    def _write(self, slug: str, cfgset: ConfigSet) -> None:
        write_atomic(self._path(slug), cfgset.model_dump_json(indent=2))

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
        with self._lock:
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
        path = self._path(slug)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            # Present but UNREADABLE (permissions, non-UTF-8 bytes) is the
            # same "not missing" case as invalid-JSON below — a distinct
            # exception type, but the same named-ValueError contract.
            raise ValueError(f"stored set {slug!r} could not be read: {exc}") from exc
        try:
            raw = upgrade_legacy_set(json.loads(text))
            return ConfigSet.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            # Present-but-invalid ≠ missing: a hand-edited file, or a set
            # saved by a newer build read under a rolled-back image
            # (extra='forbid'). Named so the router's 422 tells the operator
            # WHICH file, and so delete can catch it and still remove it.
            raise ValueError(f"stored set {slug!r} failed validation: {exc}") from exc

    def _scan(self) -> tuple[list[ConfigSet], list[str]]:
        """One pass over every stored ``*.json`` -> (parseable ConfigSets,
        slugs that aren't). The SAME pass backs both list() and
        unreadable() so the two can never disagree about which files parsed
        between one call and the next — two independent directory scans
        could see a file change (or a concurrent write land) in between.
        A read failure (bad permissions, non-UTF-8 bytes) counts as
        unreadable exactly like a validation failure; either way it isn't a
        usable ConfigSet."""
        if not self._dir.exists():
            return [], []
        good: list[ConfigSet] = []
        bad: list[str] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                raw = upgrade_legacy_set(json.loads(path.read_text()))
                good.append(ConfigSet.model_validate(raw))
            except (ValidationError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                bad.append(path.stem)
        return good, bad

    def list(self) -> list[ConfigSet]:
        """All parseable stored sets, sorted by name. Invalid files are
        SKIPPED, not fatal — one bad file must not blank every healthy set
        (and the listing is what the recovery UI needs). unreadable() names
        the skipped ones."""
        good, _ = self._scan()
        return sorted(good, key=lambda cfgset: cfgset.name)

    # Return type quoted: the `list()` method above already bound the name
    # `list` in this class's namespace, shadowing the builtin for any
    # annotation evaluated after it — a bare `list[str]` here would try to
    # subscript that method object instead of the builtin.
    def unreadable(self) -> "list[str]":
        """Slugs of stored files list() skipped — the other half of the
        same _scan() pass, so it can never disagree with what list()
        actually returned."""
        _, bad = self._scan()
        return bad

    def replace(self, slug: str, cfgset: ConfigSet) -> str:
        """Overwrite the set ALREADY stored at ``slug`` — an update, never a
        create. Exists for adopt: it must write back to the slug it read
        from; deriving from the NAME would turn '· previous' into the
        reserved slug save() refuses [c50]."""
        # Same lock as save()/delete(): replace() writes through the same
        # fixed per-slug tmp path, so two adopts of one slug — or an adopt
        # racing a save — hit the identical FileNotFoundError. The
        # exists() check joins it for the same reason register()'s
        # duplicate check does: a guard outside the lock it protects is not
        # a guard.
        with self._lock:
            if not self._path(slug).exists():
                raise ValueError(f"no set stored at slug {slug!r}")
            self._write(slug, cfgset)
        return slug

    def delete(self, slug: str) -> None:
        """Remove the set stored under ``slug`` (no-op if absent)."""
        with self._lock:
            self._path(slug).unlink(missing_ok=True)


# ===========================================================================
# Settings-snapshot diffing (Task 9) — one stripped-view helper, two callers:
# diff_snapshot (the settings-diff route) and plan_apply's own differ-check.
# ===========================================================================


def _stripped(store: dict) -> dict:
    """A settings-store snapshot with ``updated_ts``/``notes`` excluded —
    the one comparison view both ``diff_snapshot`` and ``plan_apply`` use, so
    a set never looks "different" for reasons a human never asked about: an
    older snapshot's stale write-clock, or a note added after the fact,
    must never themselves count as drift. Pure dict reshaping, no I/O."""
    return {
        kind: {
            scope_key: {
                namespace: dict(entry.get(namespace) or {})
                for namespace in NAMESPACES
                if namespace in entry
            }
            for scope_key, entry in (store or {}).get(kind, {}).items()
        }
        for kind in KINDS
    }


def diff_snapshot(snapshot: dict | None, current: dict) -> dict:
    """Diff a set's captured ``settings_snapshot`` against the live
    settings store -> ``{"changed": [...], "added": [...], "removed": [...]}``.

    ``snapshot=None`` (an old set, saved before this field existed, or the
    ``_previous`` slot when no settings_store was wired into the apply that
    captured it) diffs as empty — there is nothing to compare TO, which is
    the honest answer, not a suppressed positive. Whether to *label* that
    "no snapshot" for a caller is the route's job (``has_snapshot``), not
    this pure function's.

    Each entry:
    * ``changed``: ``{"scope", "key", "snapshot", "current"}`` — present on
      both sides with different values.
    * ``added``/``removed``: ``{"scope", "key"}`` — present on only one side
      (``added`` = only in ``current``, ``removed`` = only in ``snapshot``).

    ``scope`` is ``"<kind>/<key>"`` (e.g. ``"engines/boxa/vllm"``); ``key``
    is the C1 qualified form ``"<namespace>:<name>"`` (e.g. ``"args:x"`` —
    see ``app.routers._settings_drift``'s ``"namespace:key"`` convention,
    routers/__init__.py:182) so same-named keys in different namespaces
    never collide. Compares through ``_stripped`` — see its docstring.
    """
    empty = {"changed": [], "added": [], "removed": []}
    if snapshot is None:
        return empty

    snap = _stripped(snapshot)
    curr = _stripped(current)

    changed: list[dict] = []
    added: list[dict] = []
    removed: list[dict] = []

    for kind in KINDS:
        scope_keys = sorted(set(snap[kind]) | set(curr[kind]))
        for scope_key in scope_keys:
            scope = f"{kind}/{scope_key}"
            snap_entry = snap[kind].get(scope_key, {})
            curr_entry = curr[kind].get(scope_key, {})
            for namespace in NAMESPACES:
                snap_ns = snap_entry.get(namespace, {})
                curr_ns = curr_entry.get(namespace, {})
                for name in sorted(set(snap_ns) | set(curr_ns)):
                    key = f"{namespace}:{name}"
                    if name in snap_ns and name in curr_ns:
                        if snap_ns[name] != curr_ns[name]:
                            changed.append({
                                "scope": scope, "key": key,
                                "snapshot": snap_ns[name], "current": curr_ns[name],
                            })
                    elif name in curr_ns:
                        added.append({"scope": scope, "key": key})
                    else:
                        removed.append({"scope": scope, "key": key})

    return {"changed": changed, "added": added, "removed": removed}


def adopt_selective(snapshot: dict | None, current: dict, keys: list[dict]) -> dict:
    """Selective adopt (``POST /api/sets/{slug}/adopt``, ``mode="selective"``):
    take ONLY the named diff entries — each ``{"scope", "key"}`` pair, the
    exact shape ``diff_snapshot`` emits — from the live settings store into
    a copy of ``snapshot``; everything else in the snapshot is left exactly
    as it was. ``snapshot=None`` (adopting into a set that never had one)
    starts from an empty store.

    A requested key that no longer exists in ``current`` (a "removed" diff
    entry) is DROPPED from the result — adopting means "make the snapshot
    agree with current," and current no longer has an opinion. A malformed
    entry (missing/mistyped ``scope``/``key``, an unknown scope kind or
    namespace) is refused loudly (``ValueError`` -> 422 via the app-wide
    handler): adopting a key the caller asked for and getting silence
    instead is worse than an explicit error.
    """
    result = copy.deepcopy(snapshot) if snapshot is not None else empty_store()
    for entry in keys:
        if not isinstance(entry, dict) or "scope" not in entry or "key" not in entry:
            raise ValueError(f"adopt key entries need 'scope' and 'key': {entry!r}")
        scope, qualified = entry["scope"], entry["key"]
        if not isinstance(scope, str) or "/" not in scope:
            raise ValueError(f"malformed scope {scope!r}")
        kind, scope_key = scope.split("/", 1)
        if kind not in KINDS:
            raise ValueError(f"unknown scope kind {kind!r} in {scope!r}")
        if not isinstance(qualified, str) or ":" not in qualified:
            raise ValueError(f"malformed key {qualified!r}")
        namespace, name = qualified.split(":", 1)
        if namespace not in NAMESPACES:
            raise ValueError(f"unknown namespace {namespace!r} in {qualified!r}")

        current_entry = current.get(kind, {}).get(scope_key, {})
        current_ns = current_entry.get(namespace, {})
        if name in current_ns:
            dest_entry = result.setdefault(kind, {}).setdefault(scope_key, {})
            dest_entry.setdefault(namespace, {})[name] = current_ns[name]
            ts = current_entry.get("updated_ts", {}).get(namespace)
            if ts is not None:
                dest_entry.setdefault("updated_ts", {})[namespace] = ts
        else:
            dest_ns = result.get(kind, {}).get(scope_key, {}).get(namespace, {})
            dest_ns.pop(name, None)
    return result


# ===========================================================================
# plan_apply — pure diff
# ===========================================================================


def _park_capable_resources(world: dict) -> list[str]:
    """Every DECLARED resource whose kind supports the "park" verb (today:
    hipfire-kind) — the generalization of "the hipfire client" the pre-E1
    busy-guard checked directly by a fixed boot-time alias. Kind-agnostic
    (spec §8): membership comes from ``ENGINE_KINDS[...].human_verbs()``,
    never a hardcoded kind name. An empty/partial world (obligation 1, T3
    review) simply yields an empty list — nothing to guard, not a crash."""
    return [resource for resource, tenant in world["tenants"].items()
            if "park" in ENGINE_KINDS[tenant["engine"]].human_verbs()]


def plan_apply(cfgset: ConfigSet, world: dict, settings_now: dict | None = None) -> list[dict]:
    """Diff ``cfgset`` against ``world`` (and, for a set carrying a
    settings snapshot, against ``settings_now``) -> ordered list of step
    dicts, each ``{"step": <verb>, "resource": <name>, ...}`` (``activate``/
    ``policy_patch``/``restore_settings`` carry no resource — they are
    box-wide, not per-tenant).

    PURE: no I/O, no client calls. Emits only steps that change reality.
    Order: every eviction verb (unload/park/free) first, then activate, then
    every load/resume, then restore_settings, then policy_patch, with
    ``warn`` steps interleaved where they are generated — same ordering
    rationale as before Task 8 (evictions free VRAM/GPU before anything new
    claims it; activate flips the route between the eviction and load
    halves; see the I4 park-after-activate exception below for the one
    documented reordering).

    A resource named in ``cfgset.ephemeral.resources`` but NOT currently in
    ``world["tenants"]`` (undeclared, or removed from the declaration since
    the set was saved) is silently skipped — obligation 1 (T3 review): an
    empty or partial declared world must plan cleanly, never KeyError. This
    is the diffing-function half of that contract; ``_record_goal_intents``
    and ``apply`` (via ``local_clients``) make the same choice for the same
    reason.

    ``settings_now`` (Task 9) is the caller's own snapshot of the live
    settings store — this function does no I/O, so it cannot fetch it
    itself, exactly like ``world`` above. ``None`` (a caller that never
    wired up a settings_store) skips the restore_settings check entirely,
    the same as an old set with no ``settings_snapshot`` — see the callers'
    docstrings (``app.routers.sets``, ``apply`` below) for how it's sourced.
    """
    tenants = world["tenants"]
    eph = cfgset.ephemeral
    durable = cfgset.durable
    resources = eph.resources if eph is not None else {}

    steps: list[dict] = []

    # --- Activate (decided first — the eviction pass needs to know whether
    # it's happening, for the I4 park-after-activate exception below) ------
    activate_step = None
    if durable is not None and durable.default_route_model != world["default_route"]:
        if durable.activate_model_id is not None:
            activate_step = {"step": "activate", "model_id": durable.activate_model_id}
        else:
            activate_step = {"step": "warn", "reason": "durable-revert-unavailable"}
    activating = activate_step is not None and activate_step["step"] == "activate"

    # --- Evictions (unload / free / park) -----------------------------------
    # Fixed verb priority — unload, then free, then park — same as the pre-
    # Task-8 code's own fixed block sequence (an unload check, then a free
    # check, then the park+activate dance), now generalized past one
    # resource per verb: EVERY unload-verb resource's step (declared order
    # among ties), THEN every free-verb resource's, THEN every park-verb
    # resource's. This is NOT simply "declared order" — a set that declares
    # a park-verb resource before an unload-verb one still emits the unload
    # first, exactly like the fixed pre-Task-8 blocks did (proven by
    # test_preview_no_exec_and_estimate_arithmetic's resume-before-load
    # sibling case below, which pins the same fixed-priority contract for
    # the load/resume bucket).
    #
    # One documented exception, preserved verbatim:
    #
    # I4: normally park comes BEFORE activate (free the GPU, then re-point
    # the route). But when the activate MOVES the default route off a
    # park-verb resource (hipfire-kind today) and the route currently
    # targets THAT resource, parking first would yank the GPU out from under
    # the still-default model — so THAT resource's park runs AFTER activate
    # instead (its own guard also refuses to park while it serves the
    # default route; ordering activate first is what lets the park
    # succeed). Decided per-resource: a park-verb resource NOT currently
    # serving the default route keeps the normal park-before-activate order
    # even while another one is deferred.
    unloads: list[dict] = []
    frees: list[dict] = []
    parks: list[dict] = []
    deferred_parks: list[dict] = []

    for resource, rd in resources.items():
        tenant = tenants.get(resource)
        if tenant is None:
            continue  # obligation 1: undeclared/removed — nothing to diff
        verbs = ENGINE_KINDS[tenant["engine"]].human_verbs()

        if rd.desired == "unloaded" and "unload" in verbs and tenant["state"] == "loaded":
            unloads.append({"step": "unload", "resource": resource, "model": tenant["model"]})

        elif rd.desired == "freed" and "free" in verbs:
            # Only free when we can confirm the queue is empty (== 0). A busy
            # queue OR an unknown/None queue -> skip and be honest about it; we
            # never yank VRAM out from under a running generation.
            if tenant["queue"] == 0:
                frees.append({"step": "free", "resource": resource})
            else:
                # "busy-skipped", not "comfyui-busy-skipped" (T8 review I3):
                # the reason string is a kind-agnostic catalogued message
                # (spec §8 — no engine-kind-name literal outside
                # app.engine_kinds), same as every step verb already is;
                # WHICH resource is carried in the detail's own "resource"
                # key, not baked into the reason text.
                frees.append(
                    {"step": "warn", "reason": "busy-skipped", "resource": resource}
                )

        elif rd.desired == "parked" and "park" in verbs and tenant["state"] in ("running", "loading"):
            park_step = {"step": "park", "resource": resource}
            route_on_this = (
                tenant.get("model") is not None and world["default_route"] == tenant["model"]
            )
            if activating and route_on_this:
                deferred_parks.append(park_step)
            else:
                parks.append(park_step)

    steps.extend(unloads)
    steps.extend(frees)
    steps.extend(parks)
    if activate_step is not None:
        steps.append(activate_step)
    steps.extend(deferred_parks)

    # --- Resume / Load -------------------------------------------------------
    # Fixed verb priority again — every resume-verb resource's step, THEN
    # every load-verb resource's — mirroring the pre-Task-8 code's own fixed
    # "Resume" block before its "Load" block.
    #
    # Model precedence (lemonade-style, "load" verb): ephemeral-explicit >
    # durable > world default. A set that names the exact model it wants
    # loaded (stamped by _previous_set, or authored directly) wins outright
    # — falling through to durable/world default here is what silently
    # reloaded the wrong model on a "· previous" revert [c45].
    resumes: list[dict] = []
    loads: list[dict] = []

    for resource, rd in resources.items():
        tenant = tenants.get(resource)
        if tenant is None:
            continue  # obligation 1
        verbs = ENGINE_KINDS[tenant["engine"]].human_verbs()

        if rd.desired == "loaded" and "resume" in verbs and tenant["state"] == "parked":
            resumes.append({"step": "resume", "resource": resource})

        elif rd.desired == "loaded" and "load" in verbs and tenant["state"] == "unloaded":
            model = rd.model
            if model is None:
                model = (
                    durable.default_route_model if durable is not None else world["default_route"]
                )
            if model is None:
                loads.append({"step": "warn", "reason": "no-model-to-load", "resource": resource})
            else:
                loads.append({"step": "load", "resource": resource, "model": model})

    steps.extend(resumes)
    steps.extend(loads)

    # --- Restore settings (Task 9) ------------------------------------------
    # Independent of engine state: a settings write never itself restarts
    # anything (reload stays human), so its position relative to the
    # evict/park/activate/load steps above is cosmetic. Placed alongside the
    # always-emitted policy_patch below: both are config-only writes, never
    # diffed against ``world``. Emitted only when BOTH sides are knowable
    # (snapshot exists, settings_now was supplied) AND they actually differ
    # — an old set (no snapshot) or a caller with no settings_store wired in
    # (settings_now=None) plans exactly as before this field existed.
    if (
        cfgset.settings_snapshot is not None
        and settings_now is not None
        and _stripped(cfgset.settings_snapshot) != _stripped(settings_now)
    ):
        steps.append({"step": "restore_settings", "settings": cfgset.settings_snapshot})

    # --- Policy patch (always, not diffed) ---------------------------------
    if cfgset.policy_overrides is not None:
        steps.append({"step": "policy_patch", "policies": cfgset.policy_overrides})

    return steps


# ===========================================================================
# apply — imperative shell (serialized)
# ===========================================================================


def apply_in_progress() -> bool:
    """True if ``app.actuation.LOCK`` is currently held, WITHOUT acquiring it.

    Delegates to ``app.actuation.in_progress()`` (task 6: the lock this used
    to peek moved to app.actuation, shared by the watcher tick's actuation
    phase and the pull-through completion hook too, not just apply()). Kept
    as a re-export under its original name — ``app.storage`` imports it by
    this name (a non-blocking peek: the storage watcher yields a pass
    without needing to acquire anything).
    """
    return actuation.in_progress()

# The set of exceptions that halt an apply mid-plan (vs. crashing it). Each is
# a known, meaningful "this step could not proceed" signal from a client.
_HALT_EXCEPTIONS = (GuardError, EngineError, BusyError, ValueError)

# Plan steps that touch the surface the ODS host agent's own lifecycle ops
# (activation snapshots + readiness proofs) assume nobody else is mutating.
# free/policy_patch/warn/restore_settings are deliberately absent — freeing
# VRAM helps an in-flight activation, and neither a policy patch nor a
# settings restore mutates engine state. Verb-generic (E1 Task 8: the pre-E1
# set was {"unload_lemonade","load_lemonade","park_hipfire","activate",
# "resume_hipfire"} — this is that same set with the kind names stripped,
# since a resource's kind is no longer spellable in a step name).
_HOST_AGENT_GUARDED_STEPS = frozenset({"unload", "load", "park", "activate", "resume"})


def _record_goal_intents(cfgset, steps, world, intent_store) -> None:
    """Record the set's DECLARED tenant goals as intent — every accepted
    goal, not every executed step, BEFORE any step actuates.

    The step-derived recording this replaces had a hole: a goal the world
    already appears to satisfy plans no step, so nothing was recorded, and
    a stale 'loaded' intent (crashed tenant) survived the operator's own
    set — the reconciler then fought the apply [max-review #2]. Recording
    the goal itself closes it: intent is the operator's declared desire,
    which an apply ACCEPTS whether or not it needs to act.

    Deliberate consequences, all matching existing lifecycle semantics:
    * recorded up-front — a step that later fails leaves intent at the
      declared goal and the reconciler converges toward it (the documented
      restore-on-failure path, arbiter.py:746-747);
    * a "freed" (comfyui-kind) goal records nothing (routers/sets.py's old
      table documented why: a free leaves the server observing 'loaded'; an
      'unloaded' intent would derive permanent 'unexpected');
    * a "loaded" goal on a resume-verb (hipfire-kind) resource records
      model=None ('loaded, no opinion which model');
    * a "loaded" goal on a load-verb (lemonade-kind) resource with NO
      determinable model records nothing — the plan already warned
      no-model-to-load, and a model-less loaded intent is unrestorable;
    * model resolution mirrors plan_apply's own documented LOAD precedence
      (ephemeral-explicit > planned step > observed world) — checking the
      DECLARED ``rd.model`` FIRST, not last, matters even when no step
      plans: if the world already reads "loaded" with some OTHER model
      resident (no swap step exists to reconcile identity, only state), the
      declared model still wins the recorded intent. This is deliberate, not
      a bug: the operator's stated desire is what intent means (this
      function's own opening claim), so recording anything but the
      declaration would silently downgrade "I asked for X" into "you have
      Y, fine". The consequence is a 'drifted' status (intent X, observed Y)
      rather than a fight — reconcile.py acts on 'down' alone (app/reconcile
      .py:31), so 'drifted' is report-only and never triggers a restore; a
      human decides, exactly like the sibling 'unmanaged'/'unexpected'
      statuses reconcile.py already leaves alone.

    A resource in ``cfgset.ephemeral.resources`` but absent from
    ``world["tenants"]`` (obligation 1, T3 review — undeclared/removed since
    the set was saved) is skipped: its KIND can't be determined, and
    declaring intent for a kind you can't identify is exactly the ambiguous
    input this codebase refuses rather than guesses past.

    Kind-agnostic on purpose (spec §8): dispatch is VERB membership in
    ``ENGINE_KINDS[kind].human_verbs()``, never a hardcoded kind name.
    """
    if intent_store is None:
        return
    eph = cfgset.ephemeral
    if eph is None:
        return
    tenants = world["tenants"]
    for resource, rd in eph.resources.items():
        tenant = tenants.get(resource)
        if tenant is None:
            continue
        kind = tenant["engine"]
        verbs = ENGINE_KINDS[kind].human_verbs()

        if rd.desired == "unloaded" and "unload" in verbs:
            intent_store.record(local_key(resource), state="unloaded",
                                model=None, engine=kind)
        elif rd.desired == "parked" and "park" in verbs:
            intent_store.record(local_key(resource), state="unloaded",
                                model=None, engine=kind)
        elif rd.desired == "loaded" and "load" in verbs:
            model = rd.model
            if model is None:
                model = next(
                    (s["model"] for s in steps
                     if s["step"] == "load" and s.get("resource") == resource),
                    None,
                )
            if model is None and tenant["state"] == "loaded":
                model = tenant.get("model")
            if model is not None:
                intent_store.record(local_key(resource), state="loaded",
                                    model=model, engine=kind)
        elif rd.desired == "loaded" and "resume" in verbs:
            intent_store.record(local_key(resource), state="loaded",
                                model=None, engine=kind)
        # "freed" (or any other desired/kind combination matching none of
        # the branches above): never recorded — see docstring.


def apply(
    cfgset: ConfigSet,
    *,
    world: dict,
    local_clients,
    hostagent=None,
    policy_store,
    store: SetStore,
    events_path: Path,
    heal_suppressor=None,
    catalog=None,
    force: bool = False,
    settings_now: dict | None = None,
    settings_store=None,
    intent_store=None,
) -> dict:
    """Execute ``cfgset`` against the live box, serialized under
    ``app.actuation.LOCK`` — the one process-wide actuation lock (task 6),
    also held by the watcher tick's actuation phase and the pull-through
    completion hook.

    ``local_clients`` (E1 Task 8; same conversion Task 7 made for
    ``app.routers.control``) is ``app.local_clients.LocalClients`` (or an
    equivalent ``.client_for(resource)``): every unload/load/free/park/
    resume step resolves ITS OWN resource's client through it, live, at the
    moment that step executes — never a boot-time per-engine alias, which
    would silently keep acting on a resource's OLD connection after a live
    declaration edit (Task 10). ``apply`` no longer takes ``lemonade``/
    ``comfy``/``hipfire`` params for this reason.

    ``heal_suppressor`` (optional; None tolerated) is the arbiter's shared
    ``HealSuppressor``: an ``unload`` step arms it so contention healing
    can't revert this deliberate unload, and a ``load`` step clears it.
    None (e.g. in unit tests without the arbiter) simply skips that
    coordination.

    ``catalog`` (optional; None tolerated) is the storage catalog: a
    ``load`` step records the model as used, so the storage watcher's LRU
    eviction order sees loads made through a set apply.

    ``force=True`` skips the park-capable-resource conversation-guard (both
    the pre-veto and the per-step rechecks) for an operator overriding an
    abandoned conversation; it does NOT skip park()'s litellm route guard.

    ``settings_now`` (Task 9; optional, None tolerated) is the caller's own
    fresh read of the live settings store — plan_apply's pure input, exactly
    like ``world`` above (this function does no I/O of its own to get it).
    ``settings_store`` (optional, None tolerated) is the CLIENT that
    executes a ``restore_settings`` step, deliberately kept separate from
    ``settings_now``: reading "what's current" (to decide whether to plan a
    restore) and writing "the restore itself" are different capabilities, so
    a caller can supply one without the other. If the plan ends up
    containing ``restore_settings`` anyway and no ``settings_store`` was
    given, the step fails loudly (ValueError) rather than silently
    no-opping — see ``_execute_step``.

    ``intent_store`` (Task 5; optional, None tolerated for unit tests without
    the arbiter's lifecycle wired in) records the set's DECLARED goals as
    intent, inside this lock and before the first step actuates — see
    ``_record_goal_intents``.

    Returns an ApplyReport dict:
        {"completed": [<step>, ...], "failed": <step>|None,
         "error": <str>|None, "warnings": [<reason>, ...]}
    """
    with actuation.LOCK:
        return _run_apply(
            cfgset,
            world=world,
            local_clients=local_clients,
            hostagent=hostagent,
            policy_store=policy_store,
            store=store,
            events_path=events_path,
            heal_suppressor=heal_suppressor,
            catalog=catalog,
            force=force,
            settings_now=settings_now,
            settings_store=settings_store,
            intent_store=intent_store,
        )


def _run_apply(
    cfgset,
    *,
    world,
    local_clients,
    hostagent=None,
    policy_store,
    store,
    events_path,
    heal_suppressor=None,
    catalog=None,
    force=False,
    settings_now=None,
    settings_store=None,
    intent_store=None,
) -> dict:
    steps = plan_apply(cfgset, world, settings_now=settings_now)
    park_capable = _park_capable_resources(world)

    # Veto BEFORE any mutation (and before the _previous snapshot — a refused
    # apply changes nothing, so there is nothing to revert): a plan that would
    # park a park-verb (hipfire-kind) resource or flip the durable route
    # (litellm restart severs in-flight streams; moving the route recreates
    # a container outright) is refused while that resource's conversation is
    # live. GuardError propagates to the route -> 409. The per-step rechecks
    # below cover the gap between this veto and the step actually running.
    #
    # E1 Task 8 generalization: the pre-E1 guard checked exactly ONE fixed
    # hipfire client. Today's guarded set is every park-capable resource a
    # "park" step actually targets, PLUS — if an "activate" step is
    # planned — every DECLARED park-capable resource (moving the default
    # route can affect any of them, whether or not this set's ephemeral
    # touches it; that's the pre-E1 behavior too, just never generalized
    # past a single resource before now).
    guarded_resources = {s["resource"] for s in steps if s["step"] == "park"}
    if any(s["step"] == "activate" for s in steps):
        guarded_resources.update(park_capable)
    if not force and guarded_resources:
        for resource in sorted(guarded_resources):
            try:
                local_clients.client_for(resource).ensure_not_busy(
                    f"apply set {cfgset.name!r}"
                )
            except _HALT_EXCEPTIONS:
                log_event(events_path, "apply-vetoed", {"name": cfgset.name})
                raise

    # Pre-veto, not per-step: plan order runs evictions FIRST, so hitting the
    # agent's own 409 at the activate step would leave a half-applied set.
    if (
        not force
        and hostagent is not None
        and any(s["step"] in _HOST_AGENT_GUARDED_STEPS for s in steps)
    ):
        lifecycle = hostagent.lifecycle()
        if lifecycle["active"]:
            log_event(
                events_path,
                "apply-vetoed",
                {
                    "name": cfgset.name,
                    "reason": "host-agent-busy",
                    "operation": lifecycle["operation"],
                },
            )
            raise BusyError(
                f"host agent is busy ({lifecycle['operation'] or 'model lifecycle'}); "
                "retry after it finishes or use force=true"
            )

    log_event(events_path, "apply-start", {"name": cfgset.name})

    # FIRST mutation: capture pre-apply reality as the one-click revert set,
    # before any step touches the box. settings_now rides along so reverting
    # to "· previous" restores settings too (Task 9).
    store.save_previous(_previous_set(world, settings_now))

    # Record the DECLARED goals as intent, still inside the lock, before the
    # first step actuates (Task 5) — see _record_goal_intents.
    _record_goal_intents(cfgset, steps, world, intent_store)

    report: dict = {"completed": [], "failed": None, "error": None, "warnings": []}

    for step in steps:
        name = step["step"]

        if name == "warn":
            report["warnings"].append(step["reason"])
            # "resource" carried through when plan_apply attached one (T8
            # review M2) — a resource-less warn (e.g. durable-revert-
            # unavailable) still logs cleanly via .get, no KeyError.
            log_event(
                events_path,
                "apply-warn",
                {"reason": step["reason"], "resource": step.get("resource")},
            )
            continue

        try:
            _execute_step(
                step,
                local_clients,
                hostagent,
                policy_store,
                park_capable,
                heal_suppressor=heal_suppressor,
                catalog=catalog,
                force=force,
                settings_store=settings_store,
            )
        except _HALT_EXCEPTIONS as exc:
            report["failed"] = step
            report["error"] = str(exc)
            # "resource" included (T8 review M1): two same-kind resources
            # failing the same verb with the same error text must stay
            # distinguishable in the event log — a bare verb alone is
            # exactly the invisibility class this branch has already ruled
            # forbidden (app.engine_kinds' execute_unload/execute_load
            # folding "resource" into their own failure details for the
            # identical reason). box-wide steps (activate/policy_patch/
            # restore_settings) carry no resource — .get keeps this a
            # clean None for those, never a KeyError.
            log_event(
                events_path,
                "apply-end",
                {
                    "outcome": "failed",
                    "step": name,
                    "resource": step.get("resource"),
                    "error": str(exc),
                },
            )
            return report

        # restore_settings carries the whole settings store as its payload —
        # log a compact scope count instead of dumping it verbatim into
        # events.jsonl.
        detail = (
            {"scopes": sum(len(v) for v in step["settings"].values())}
            if name == "restore_settings"
            else {k: v for k, v in step.items() if k != "step"}
        )
        log_event(events_path, name, detail)
        report["completed"].append(step)

    log_event(events_path, "apply-end", {"outcome": "ok"})
    return report


def _execute_step(
    step, local_clients, hostagent, policy_store, park_capable_resources,
    heal_suppressor=None, catalog=None, force=False, settings_store=None,
) -> None:
    name = step["step"]
    if name == "unload":
        local_clients.client_for(step["resource"]).unload(step["model"])
        # Deliberate unload: arm suppression so the arbiter doesn't heal it back.
        if heal_suppressor is not None:
            heal_suppressor.note_deck_unload()
    elif name == "load":
        local_clients.client_for(step["resource"]).load(step["model"])
        # Deliberate load: the model is wanted resident, so clear suppression.
        if heal_suppressor is not None:
            heal_suppressor.clear()
        # ...and it is a real use of the model: feed the storage catalog's LRU
        # bookkeeping (lemonade names GGUFs "extra.<file>"; units are bare).
        if catalog is not None:
            catalog.note_used_gguf(step["model"].removeprefix(_EXTRA_PREFIX))
    elif name == "free":
        local_clients.client_for(step["resource"]).free()
    elif name == "park":
        local_clients.client_for(step["resource"]).park(force=force)
    elif name == "resume":
        local_clients.client_for(step["resource"]).resume()
    elif name == "activate":
        # Recheck at execution time: a request may have landed on a
        # park-capable resource since the pre-veto (activation restarts
        # litellm, severing in-flight streams; moving the route recreates a
        # container outright). Same guarded-resource set as the pre-veto
        # (every DECLARED park-capable resource) — precomputed once in
        # _run_apply and threaded through, since it doesn't change mid-apply.
        if not force:
            for resource in park_capable_resources:
                local_clients.client_for(resource).ensure_not_busy(
                    f"activate {step['model_id']!r}"
                )
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
    elif name == "restore_settings":
        # Fail LOUDLY, never a silent no-op: if the plan says restore but no
        # settings_store was wired into apply(), that's a real misconfiguration
        # a caller needs to know about, not something to quietly skip past.
        if settings_store is None:
            raise ValueError(
                "restore_settings step has no settings_store wired into apply(); "
                "cannot restore settings"
            )
        settings_store.restore(step["settings"])
    else:  # pragma: no cover - plan_apply is the sole producer of steps
        raise AssertionError(f"unknown step {name!r}")


def _previous_set(world: dict, settings_now: dict | None = None) -> ConfigSet:
    """Build the ``· previous`` revert snapshot from pre-apply world reality.

    Generalized past the old fixed triple (E1 Task 8, obligation 1 T3
    review): iterates every DECLARED resource in ``world["tenants"]`` and
    stamps ITS desired state by VERB membership, kind-agnostic. A load-verb
    (lemonade-kind) resource mirrors its current load/unload state, stamped
    with the ACTUAL loaded model (None when unloaded) so reverting reloads
    what WAS resident rather than falling through plan_apply's durable/
    world-default fallback — which, on the ``· previous`` set itself (no
    durable section), meant the default ROUTE model, not necessarily the
    one that was loaded [c45]. A park-verb (hipfire-kind) resource mirrors
    running/parked. A free-verb (comfyui-kind) resource is OMITTED entirely
    — a freed VRAM cache can't be meaningfully un-freed, the same as the old
    fixed shape's unconditional ``"comfyui": {"state": "leave"}``; omission
    IS "leave" in the new schema. durable records the old default route
    with activate_model_id=None (world carries no catalog id).

    ``settings_now`` (Task 9) is stamped straight into this set's own
    ``settings_snapshot``, so the one-click revert restores settings too:
    applying "· previous" later diffs its snapshot (pre-apply settings)
    against whatever is live BY THEN, and proposes a restore_settings step
    if the original apply changed anything. ``None`` (no settings_store
    wired into the apply that produced this snapshot) keeps the pre-Task-9
    shape exactly — settings_snapshot stays None, same as an old set.

    Note: this ``model`` field and the per-file isolation in ``SetStore``
    ship in the SAME commit, so an image OLDER than this one has neither.
    Rolling back to a build that already CONTAINS the isolation fix (this
    commit or later) degrades a mismatched ``_previous.json`` to a skipped
    set (``list()``/``unreadable()``), not a downed API. Rolling back PAST
    this commit — to a build without the isolation fix — still downs the
    whole sets API on a new-shape ``_previous.json`` (extra='forbid' with
    no ``model`` field), exactly as [c44] describes, until the file is
    either removed from disk by hand or overwritten by that older build's
    own next apply: ``save_previous()`` unconditionally overwrites
    ``_previous.json`` with whatever shape the RUNNING build writes, so one
    more apply on the old build self-heals it.
    """
    tenants = world["tenants"]
    resources: dict = {}
    for resource, tenant in tenants.items():
        verbs = ENGINE_KINDS[tenant["engine"]].human_verbs()
        if "unload" in verbs:
            loaded = tenant["state"] == "loaded"
            resources[resource] = {
                "desired": "loaded" if loaded else "unloaded",
                "model": tenant["model"] if loaded else None,
            }
        elif "park" in verbs:
            running = tenant["state"] in ("running", "loading")
            resources[resource] = {"desired": "loaded" if running else "parked"}
        # else (e.g. free-verb/comfyui-kind): omitted — see docstring.

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
        ephemeral={"resources": resources},
        settings_snapshot=settings_now,
    )
