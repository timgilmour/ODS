"""Engine-kind registry — THE one module that knows local engine names.

E1 (spec §2) ships this as an internal protocol with exactly three kinds;
E2 turns it into the pluggable descriptor registry. Spec §8 binds: no
engine name may appear in app/ outside this module (allowed residues are
listed in the plan's Global Constraints).

Task 1 shipped the declaration half: kind names + connection schemas +
validate_engines. Task 3 (this module's other half) adds the adapters:
`observe`/`active`/`arbiter_verbs`/`human_verbs`/`demand`, moved VERBATIM
from app/state.py's three `_snapshot_*` methods (see each adapter's
`observe` for the incident-history comments that travelled with them), and
`build_client` (review fix, T3 round 2): the per-kind CONSTRUCTOR call
`app.local_clients.LocalClients` used to hold directly (kind-name
`if`/`elif` dispatch) — moved here because per-kind constructor knowledge
is exactly what this module exists to hold; `LocalClients` now calls
`ENGINE_KINDS[kind].build_client(connection, settings)` and holds no
engine-kind-name literal anywhere. `idle_action`/`reclaimable` (arbiter
generalization) and the actuator methods (execute_unload/execute_free/
execute_load) are Task 5/6's additions to these same classes — this
module is the one place every later task's per-kind logic lands, per its
own docstring above.

hipfire's ``observe`` needs the RESOURCE name (to look up its model in
litellm's route table, which the pre-E1 code keyed by the literal
"hipfire"). Delegated mechanism choice (Task 3 brief): `ctx` carries
`{"registry", "routes", "resource"}` — `resource` added by `World.snapshot`
per iteration (a fresh per-resource ctx dict, not a mutated shared one) —
rather than a separate `observe(client, mem, now, resource, ctx)`
parameter, so every adapter's `observe` keeps the identical 4-arg
signature whether or not that particular kind needs the resource name.

`uses_gguf(obs)` and `restart_container(entry)` (Task 9) are this same
per-kind-knowledge pattern applied to the storage feature: `app.storage`'s
in-use scan (a GGUF unit currently loaded by ANY declared lemonade-kind
resource, never a fixed `world["tenants"]["lemonade"]` index) and
`app.notify`'s moved-in-GGUF restart hook (which container to bounce, and
whether THAT resource's own client currently has something loaded worth
protecting) both iterate the live declaration and ask the matching
adapter, rather than assuming a single resource literally named
"lemonade" is the only GGUF server that could exist."""

from app.engines import EngineError, GuardError
from app.engines.comfyui import ComfyClient
from app.engines.docker_ctl import DockerCtl
from app.engines.hipfire import HipfireClient
from app.engines.lemonade import LemonadeClient
from app.engines.litellm import LiteLLMClient
from app.observe import local_key
from app.registry import HIPFIRE_FOOTPRINT

_OPENAI_PREFIX = "openai/"
_EXTRA_PREFIX = "extra."

# Engine failures carry raw HTTP bodies (`EngineError(resp.text)`), and a
# multi-KB body re-logged per 2 s tick is exactly the trim-thrash input
# app/events.py's byte-denominated trim bounds [T10]. Bound the DETAIL too:
# 500 chars keeps a failed-load line well under events' 2500 B/line
# hysteresis ratio even when the body is unstable tick to tick (timestamps
# in an error page defeat dedup, so truncation is the backstop dedup needs).
# Moved here VERBATIM from app/arbiter.py (E1 Task 6) alongside the two
# actuator methods that are now its only callers.
_ERROR_TEXT_MAX = 500


def _error_text(exc: BaseException) -> str:
    text = str(exc)
    if len(text) <= _ERROR_TEXT_MAX:
        return text
    return text[:_ERROR_TEXT_MAX] + " …[truncated]"


# VRAM overhead slack subtracted when estimating a "free"-verb kind's (today:
# comfyui) reclaimable bytes from raw GPU usage (fragmentation, driver/
# runtime overhead, small tenants) — moved here VERBATIM from app/arbiter.py
# (Task 5) alongside the usage-gap estimate that consumes it, since both are
# now per-kind (`_ComfyAdapter.reclaimable`), not arbiter-module globals.
_SLACK_BYTES = 1024**3  # 1 GiB

# hipfire runs as a sibling container on the compose network; its health
# endpoint is <container>:11435/health (config/ports.json + manifest.yaml).
# Owned here (review fix, T3 round 2 — was duplicated in local_clients.py):
# the hipfire adapter is the one place that builds a HipfireClient from a
# declared connection now. app.main._build_deck's own copy of this port
# number, for the pre-E1 shared actuation instance, is untouched — a
# separate, coexistence-era construction this task doesn't migrate.
_HIPFIRE_PORT = 11435

# kind -> {connection field -> required?}
KNOWN_KINDS: dict[str, dict[str, bool]] = {
    "lemonade": {"url": True, "metrics_url": True, "container": True},
    "comfyui": {"url": True},
    "hipfire": {"container": True},
}

_POLICY_FIELDS = {"priority": int, "pinned": bool, "idle_ttl": int}


def _bad(reason: str) -> ValueError:
    return ValueError(reason)


def validate_engines(engines: object) -> None:
    """Raise ValueError (one-line reason) unless `engines` is a valid
    declaration list. Refuse, never coerce ([[literal-declared-inputs]])."""
    if not isinstance(engines, list):
        raise _bad("engines must be a list")
    seen: set[str] = set()
    for e in engines:
        if not isinstance(e, dict):
            raise _bad("engine entry must be an object")
        extra = set(e) - {"resource", "kind", "connection", "gpu_index",
                          "policy_defaults"}
        if extra:
            raise _bad(f"engine entry has extra field(s): {sorted(extra)}")
        resource = e.get("resource")
        if (not isinstance(resource, str) or not resource
                or "/" in resource or resource != resource.strip()):
            raise _bad("resource must be a non-empty string without '/'")
        if resource in seen:
            raise _bad(f"duplicate resource {resource!r}")
        seen.add(resource)
        kind = e.get("kind")
        if kind not in KNOWN_KINDS:
            raise _bad(f"unknown kind {kind!r} (known: {sorted(KNOWN_KINDS)})")
        schema = KNOWN_KINDS[kind]
        conn = e.get("connection")
        if not isinstance(conn, dict):
            raise _bad(f"{resource}: connection must be an object")
        extra_conn = set(conn) - set(schema)
        if extra_conn:
            raise _bad(f"{resource}: connection has extra field(s): "
                       f"{sorted(extra_conn)}")
        for field, required in schema.items():
            if required and not (isinstance(conn.get(field), str) and conn[field]):
                raise _bad(f"{resource}: connection.{field} is required "
                           f"for kind {kind!r}")
        gpu = e.get("gpu_index")
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise _bad(f"{resource}: gpu_index must be a non-negative integer")
        pol = e.get("policy_defaults")
        if not isinstance(pol, dict) or set(pol) != set(_POLICY_FIELDS):
            raise _bad(f"{resource}: policy_defaults must have exactly "
                       f"{sorted(_POLICY_FIELDS)}")
        for field, typ in _POLICY_FIELDS.items():
            v = pol[field]
            if typ is int and (isinstance(v, bool) or not isinstance(v, int)):
                raise _bad(f"{resource}: policy_defaults.{field} must be int")
            if typ is bool and not isinstance(v, bool):
                raise _bad(f"{resource}: policy_defaults.{field} must be bool")


def _strip_prefix(name: str | None, prefix: str) -> str | None:
    if name is None:
        return None
    return name.removeprefix(prefix)


class _LemonadeAdapter:
    """lemonade-kind: load/unload a GGUF, idle-clock tracked by llama.cpp's
    monotonic activity counters."""

    def observe(self, client, mem: dict, now: float, ctx: dict) -> dict:
        # Moved VERBATIM from app.state.World._snapshot_lemonade (app/state.py,
        # pre-Task-3 lines 137-183) — comments included, they carry incident
        # history. `self._lemonade_last_*` reads/writes become `mem` dict
        # reads/writes (keys: last_value, last_activity_time, last_loaded).

        # A deck-authored load in flight: llama-server health reports nothing
        # loaded while weights stream in, which upstream must not read as dead
        # (derive_status turns 'loading' into the inert 'warming'). Idle-clock
        # bookkeeping is deliberately skipped: the loaded-value transition on
        # the first post-load snapshot resets the activity clock as usual.
        if client.load_in_flight():
            return {"state": "loading", "model": None, "footprint": None, "idle_s": None}

        try:
            status = client.status()
        except EngineError:
            return {"state": "unknown", "model": None, "footprint": None, "idle_s": None}

        loaded = status["loaded"]
        activity = client.activity()  # never raises

        # A load transition is activity: llama.cpp's counters restart at 0 on
        # every load, so a fresh model can report the same counter value the
        # previous one ended with and the value-change check alone would let
        # it inherit a stale idle clock (and be evicted on the next tick).
        if loaded != mem.get("last_loaded"):
            mem["last_activity_time"] = now
        mem["last_loaded"] = loaded

        if activity is not None:
            if mem.get("last_value") is None or activity != mem.get("last_value"):
                mem["last_activity_time"] = now
            mem["last_value"] = activity
            idle_s = now - mem["last_activity_time"]
        else:
            idle_s = None

        footprint = None
        if loaded:
            key = _strip_prefix(loaded, _EXTRA_PREFIX)
            registry = ctx["registry"]
            try:
                footprint = registry.footprint(key)
            except FileNotFoundError:
                footprint = None

        return {
            "state": "loaded" if loaded else "unloaded",
            "model": loaded,
            "footprint": footprint,
            "idle_s": idle_s,
        }

    def active(self, obs: dict) -> bool:
        return obs["state"] == "loaded"

    def uses_gguf(self, obs: dict) -> str | None:
        """The loaded model name when this resource is currently serving a
        GGUF, else None — the same ``obs["state"] == "loaded"`` read
        ``active()`` makes just above. E1 Task 9: the one place that knows
        "a lemonade-kind resource being loaded means a specific GGUF file
        is in use" — app.storage's in-use scan and app.notify's
        moved-in-GGUF restart hook both read this instead of assuming a
        single resource named "lemonade" is the only place a GGUF could be
        loaded."""
        return obs["model"] if obs["state"] == "loaded" else None

    def arbiter_verbs(self) -> frozenset:
        return frozenset({"unload"})

    def human_verbs(self) -> frozenset:
        return frozenset({"load", "unload"})

    def demand(self) -> bool:
        return True

    def idle_action(self, obs: dict, policy: dict, gpu: dict | None, co_footprints: int) -> dict | None:
        # Moved VERBATIM from app.arbiter._decide_idle_release's pre-E1
        # rule 1 (arbiter.py:251-262) — comments included, they carry
        # incident history. `gpu`/`co_footprints` are unused for this kind
        # (kept for adapter-interface uniformity with comfy's idle_action,
        # same rationale as `observe`'s per-kind-unused `ctx` fields — see
        # this module's docstring).
        if (
            obs["state"] == "loaded"
            and not policy["pinned"]
            and policy["idle_ttl"] > 0
            and obs["idle_s"] is not None
            and obs["idle_s"] >= policy["idle_ttl"]
        ):
            # NOTE: the default-route model is intentionally NOT guarded
            # here — idle release on it is the idle-GPU-burn fix (reload
            # ~4 s). This exception applies ONLY to idle release, never to
            # contention (see app.arbiter._eviction_candidates).
            return {"type": "unload", "model": obs["model"]}
        return None

    def reclaimable(self, obs: dict, gpu: dict | None, co_footprints: int) -> int | None:
        """Known footprint while loaded, else None (unquantifiable, so not a
        usable eviction/idle candidate) — moved VERBATIM from the
        eligibility half of app.arbiter._eviction_candidates's pre-E1
        lemonade branch (arbiter.py:369-374). `gpu`/`co_footprints` are
        unused for this kind (see idle_action's docstring above)."""
        if obs["state"] == "loaded" and obs.get("footprint"):
            return obs["footprint"]
        return None

    def build_client(self, connection: dict, settings):
        # Moved from app.local_clients._build_client (review fix, T3 round
        # 2) — same constructor call app.main._build_deck used to make
        # once. `settings.lemonade_key` is a deck-level shared credential,
        # not part of the per-resource declared connection.
        return LemonadeClient(connection["url"], settings.lemonade_key,
                              metrics_url=connection["metrics_url"])

    def restart_container(self, entry: dict) -> str | None:
        """The container to bounce so a newly-arrived GGUF registers
        (lemonade only rescans its store at startup — app.notify's module
        docstring) — the declared resource's OWN ``connection.container``,
        never a single settings-level alias (E1 Task 9: two lemonade-kind
        resources restart their own containers independently, each on its
        own declared connection)."""
        return entry["connection"]["container"]

    def execute_unload(self, watcher, resource: str, client, model: str,
                        actuated: set[str]) -> None:
        """Moved VERBATIM from Watcher._execute's unload branch (E1 Task 6,
        pre-move: app/arbiter.py's old `_execute`, ~lines 789-874) — every
        comment travels, they carry incident history. `watcher` is the
        calling Watcher instance: this block always needed cross-action
        bookkeeping that lives THERE (`_log`'s dedup memos, the shared
        `HealSuppressor`, the shared `IntentStore`) — previously reached as
        bare `self.*` from inside Watcher itself, now reached the same way
        through the instance handed in. `LOCAL_LEMONADE_KEY` (a single
        hardcoded key) is now `local_key(resource)` — the same
        per-resource generalization every other site in this branch makes.
        `actuated` is `_execute`'s own per-tick set, passed in so the
        "added before the engine call" invariant below stays structural
        (see its own comment) rather than becoming a caller's
        after-the-fact add that a future refactor could accidentally skip.
        """
        key = local_key(resource)
        # Whoever actuates, records — and records FIRST, so a tick that
        # lands mid-unload derives 'parked', never 'down' (2026-08-06).
        # actor="deck": this is the arbiter's OWN automatic action
        # (idle-release or contention-eviction), never an operator's —
        # app.routers.control's pull-through supersession check relies on
        # that distinction to tell "the deck unloaded something idle" apart
        # from "the operator asked for this" [max-review Important-1].
        # Snapshot the prior record first, so a FAILED unload can roll it
        # back: the actuation didn't happen, and leaving 'unloaded' standing
        # against a still-loaded model derives the inert 'unexpected' status
        # until an eventual retry [max-review #9].
        prior = None
        if watcher._intent_store is not None:
            prior = watcher._intent_store.get().get(key)
            watcher._intent_store.record(
                key, state="unloaded", model=None, engine="lemonade", actor="deck")
        # Added BEFORE the engine call (mirrors the retrigger tail below) so
        # the invariant is structural: an unload wrapped in try/except in
        # the future still can't return without this key marked actuated.
        actuated.add(key)
        try:
            client.unload(model)
        except EngineError as exc:
            # Per-action isolation — _execute_restore's documented
            # invariant (see its docstring), applied to the arm that
            # lacked it: a raise here must not abort the remaining
            # actions or the reconcile/derive/provenance passes via
            # tick()'s broad catch [max-review #9].
            #
            # Logged BEFORE the rollback, not after: the diagnostic is
            # what tells an operator this happened at all, so it must
            # not be contingent on the rollback succeeding [T7 review].
            #
            # "resource" in the detail (review fix, T6 round 2): two
            # DIFFERENT lemonade-kind resources failing unload with the
            # same model name and error text would otherwise collapse
            # into ONE logged line — `_log`'s failure-dedup memo keys on
            # `(kind, sorted(detail.items()))`, so an identical detail
            # from a DIFFERENT resource was silently swallowed as if it
            # were the SAME resource's repeat failure. Folding `resource`
            # into the detail fixes both at once: the event is now
            # distinguishable AND the dedup key naturally differs per
            # resource (the key is derived from this same dict).
            watcher._log("unload-failed",
                         {"resource": resource, "model": model,
                          "error": _error_text(exc)})
            # A real compare-and-swap: the predicate and the write
            # run in ONE critical section inside the store
            # (put_back_if). An operator can record a deliberate
            # load/unload during the seconds this engine call hung,
            # and blindly putting `prior` back would silently revert
            # THEIR action — the same class of bug as the pull-through
            # supersession hole task 6 closed. An earlier version read
            # the record and then wrote in two separate critical
            # sections, which is check-then-act, not CAS [T7/T8
            # fix-round review].
            #
            # Still the speculative record this arm wrote? It is
            # deck-authored AND unloaded; an operator's write flips
            # actor to "operator" (their routes never pass
            # actor="deck"), so that pair is a sufficient witness.
            def _still_ours(current):
                return (current is not None
                        and current.get("actor") == "deck"
                        and current.get("state") == "unloaded")

            if watcher._intent_store is not None and prior is not None:
                if not watcher._intent_store.put_back_if(key, _still_ours, prior):
                    # Someone else's intent is newer. Leave it alone
                    # and say so — a silent skip here would be the
                    # same invisibility the rollback exists to end.
                    # Covers the forget()-shaped case too (current
                    # gone entirely), which the previous shape logged
                    # nothing for.
                    watcher._log("unload-rollback-skipped",
                                 {"model": model,
                                  "reason": "intent changed during the unload"})
            # prior is None (no record for this key yet) + failure
            # leaves the fresh 'unloaded' record standing: there is no
            # forget() to undo it with. Rare, and no worse than the
            # behavior this fix replaces.
        else:
            # Same re-arm as the load arm's success path: a
            # successful unload between two identical unload
            # failures must not swallow the second one.
            watcher._clear_failure_dedup()
            # Deck-initiated unload (idle release OR contention
            # eviction): arm suppression so healing can't immediately
            # revert it.
            watcher._heal_suppressor.note_deck_unload()
            watcher._log(f"unload_{resource}", {"model": model})

    def execute_load(self, watcher, resource: str, client, pending: dict,
                      actuated: set[str]) -> None:
        """Moved VERBATIM from Watcher._execute's pending-load retrigger
        tail (E1 Task 6, pre-move: ~lines 924-957) — comments included.
        See execute_unload's docstring above for what `watcher`/`actuated`
        are and why."""
        key = local_key(resource)
        # Deck-authored load: record BEFORE actuating (same rule as the
        # unload arm above). A failed load then derives 'down' and the
        # reconciler retries under the existing FAILURE_BUDGET —
        # deliberate, not a gap. actor="deck" for the same reason as the
        # unload arm above — this is automatic contention-healing, not
        # an operator's request.
        if watcher._intent_store is not None:
            watcher._intent_store.record(
                key, state="loaded", model=pending["model"], engine="lemonade",
                actor="deck")
        actuated.add(key)
        try:
            client.load(pending["model"])
        except EngineError as exc:
            # Load failed (engine unreachable, bad response, etc.) — log
            # and let the loop survive; the next tick re-evaluates.
            # "resource" in the detail: same fix, same reason as
            # execute_unload's "unload-failed" call above — two different
            # lemonade-kind resources failing load identically must not
            # collapse into one line, and the dedup key derives from this
            # same dict.
            watcher._log("load-failed",
                         {"resource": resource, "error": _error_text(exc)})
        else:
            # Re-arm the failure-dedup memo: a recovery between two
            # IDENTICAL failures must not swallow the second one —
            # fail→recover→fail is the flap the T9-fix review named as
            # this codebase's forbidden dedup class.
            watcher._clear_failure_dedup()
            # Deck-initiated load: the model is wanted resident again, so
            # clear any suppression left by a prior deliberate unload.
            watcher._heal_suppressor.clear()
            # ...and it's a genuine use of the model: record it so the
            # storage watcher's LRU order doesn't treat an auto-reloaded
            # default-route model as "never used".
            if watcher._catalog is not None:
                watcher._catalog.note_used_gguf(
                    pending["model"].removeprefix(_EXTRA_PREFIX))
            watcher._log("load-retriggered", {"model": pending["model"]})

    def restore(self, client, model: str | None) -> None:
        """The REAL restore call for a lemonade-kind resource: load by
        name (app/engines/lemonade.py:64). Moved from Watcher._restore's
        lemonade branch (E1 Task 6)."""
        client.load(model)


class _ComfyAdapter:
    """comfyui-kind: no load/unload, only a VRAM `free()`; idle-clock tracked
    by queue occupancy."""

    def observe(self, client, mem: dict, now: float, ctx: dict) -> dict:
        # Moved VERBATIM from app.state.World._snapshot_comfy (app/state.py,
        # pre-Task-3 lines 192-211). `self._comfy_last_activity_time` becomes
        # `mem["last_activity_time"]`.
        try:
            queue = client.queue_len()
        except EngineError:
            return {"state": "unknown", "queue": None, "idle_s": None}

        if mem.get("last_activity_time") is None:
            # First-ever snapshot: establish a baseline so idle_s is always
            # computable from here on, even if the queue happens to be
            # empty on this very first call.
            mem["last_activity_time"] = now

        if queue > 0:
            mem["last_activity_time"] = now
            state = "busy"
        else:
            state = "idle"

        idle_s = now - mem["last_activity_time"]
        return {"state": state, "queue": queue, "idle_s": idle_s}

    def active(self, obs: dict) -> bool:
        return obs["state"] == "busy"

    def uses_gguf(self, obs: dict) -> str | None:
        # comfyui-kind never serves a GGUF: a storage "gguf" unit can never
        # be in use by a comfy tenant, and app.notify never restarts one
        # over a gguf move (see restart_container below) — defined anyway
        # for adapter-interface completeness (every kind implements the
        # same protocol, per this module's docstring).
        return None

    def arbiter_verbs(self) -> frozenset:
        return frozenset({"free"})

    def human_verbs(self) -> frozenset:
        return frozenset({"free"})

    def demand(self) -> bool:
        return False

    def idle_action(self, obs: dict, policy: dict, gpu: dict | None, co_footprints: int) -> dict | None:
        # Moved VERBATIM from app.arbiter._decide_idle_release's pre-E1
        # rule 2 (arbiter.py:264-279) — comments included, they carry
        # incident history.
        if (
            obs["state"] == "idle"
            and obs["queue"] == 0
            and not policy["pinned"]
            and policy["idle_ttl"] > 0
            and obs["idle_s"] is not None
            and obs["idle_s"] >= policy["idle_ttl"]
            # A free that can't reclaim anything is a no-op that re-arms the
            # TTL and re-fires every idle_ttl seconds forever, flooding the
            # event ring. None (this resource's GPU unresolvable) must not
            # suppress the free: unknown usage is not proof there's nothing
            # to reclaim.
            and self.reclaimable(obs, gpu, co_footprints) != 0
        ):
            return {"type": "free"}
        return None

    def reclaimable(self, obs: dict, gpu: dict | None, co_footprints: int) -> int | None:
        """Estimated bytes a free would reclaim, or None when this resource
        isn't eligible at all (its GPU isn't in the snapshot, or it's
        busy/unknown — never free a busy or unknown-state comfy queue).
        This kind's VRAM presence isn't directly observable, so attribute to
        it whatever its GPU's usage doesn't explain — minus the known
        footprints of OTHER co-resident loaded/running tenants
        (`co_footprints`, the generalization of "minus a co-resident loaded
        lemonade's footprint" app.arbiter._eviction_candidates used to
        compute inline) and the fixed slack allowance. Moved from
        app.arbiter._comfy_reclaimable / the pre-E1 comfyui branch of
        _eviction_candidates (arbiter.py:284-305, 356-365); serves BOTH
        callers now — this adapter's own idle_action (the no-op-flood
        guard above) and app.arbiter._eviction_candidates (contention
        eligibility) — so the busy/unknown exclusion has to live HERE, not
        only in the idle_action call site that used to pre-gate it."""
        if gpu is None:
            return None
        if obs["state"] != "idle" or obs.get("queue") != 0:
            return None
        return max(0, gpu["used"] - co_footprints - _SLACK_BYTES)

    def build_client(self, connection: dict, settings):
        return ComfyClient(connection["url"])

    def restart_container(self, entry: dict) -> str | None:
        # No restart needed: ComfyUI's /api/models/{type} lists files per
        # request (app.notify's module docstring) — a moved-in file is
        # already visible with no action.
        return None

    def execute_free(self, watcher, resource: str, client) -> bool:
        """Moved VERBATIM from Watcher._execute's comfy free branch (E1
        Task 6, pre-move: ~lines 875-901) — comments included. `watcher` is
        the calling Watcher instance (see _LemonadeAdapter.execute_unload's
        docstring for why). Returns True when the free RACED or FAILED
        (VRAM not confirmed reclaimed) — the caller (Watcher._execute) uses
        this in place of the old local `eviction_raced` flag to decide
        whether the pending-load retrigger tail may run."""
        try:
            client.free()
        except GuardError:
            # Race: comfy's queue filled between decide and execute.
            # The VRAM was NOT reclaimed — log and skip the reload.
            watcher._log("free-raced", {})
            return True
        except EngineError as exc:
            # Comfy unreachable / refused. Same conservative reading
            # as the race above — the VRAM is NOT confirmed reclaimed,
            # so the pending load must not be re-triggered into a GPU
            # that may still be full — and the same per-action
            # isolation as the unload arm [max-review #9].
            watcher._log("free-failed", {"error": str(exc)})
            return True
        else:
            # Re-arm the idle TTL so the idle-release rule fires once
            # per TTL while comfy stays idle, not on every tick.
            # note_comfy_freed() -> note_freed(resource) (E1 Task 3).
            watcher._world.note_freed(resource)
            watcher._log(f"free_{resource}", {})
            return False


class _HipfireAdapter:
    """hipfire-kind: container lifecycle (park/resume), no arbiter verb —
    park stays human-only (structural omission made explicit)."""

    def observe(self, client, mem: dict, now: float, ctx: dict) -> dict:
        # Moved VERBATIM from app.state.World._snapshot_hipfire (app/state.py,
        # pre-Task-3 lines 213-232). `routes` and `resource` (for the
        # route-table model lookup, see this module's docstring) come from
        # `ctx`; hipfire has no per-resource idle-clock state, so `mem` is
        # unused here (kept in the signature for adapter-interface uniformity).
        try:
            state = client.status()
        except EngineError:
            return {"state": "unknown", "model": None, "footprint": 0, "queue_depth": None}

        # Poll /stats while running: besides surfacing queue_depth, this is
        # what feeds the HipfireClient conversation-activity tracker every
        # watcher tick (the park/apply busy guard reads that tracker). A
        # stats failure must not take down the snapshot — unknown, not fatal.
        queue_depth = None
        if state == "running":
            try:
                queue_depth = client.stats().get("queue_depth")
            except EngineError:
                queue_depth = None

        routes = ctx["routes"]
        model = None if routes is None else _strip_prefix(routes.get(ctx["resource"]), _OPENAI_PREFIX)
        footprint = HIPFIRE_FOOTPRINT if state == "running" else 0
        return {"state": state, "model": model, "footprint": footprint, "queue_depth": queue_depth}

    def active(self, obs: dict) -> bool:
        return obs["state"] == "running"

    def uses_gguf(self, obs: dict) -> str | None:
        # hipfire's own obs["model"] is a litellm-routed name, not a GGUF
        # file the storage catalog tracks — never in use by a gguf storage
        # unit. Defined anyway for adapter-interface completeness.
        return None

    def arbiter_verbs(self) -> frozenset:
        return frozenset()

    def human_verbs(self) -> frozenset:
        return frozenset({"park", "resume"})

    def demand(self) -> bool:
        return False

    def idle_action(self, obs: dict, policy: dict, gpu: dict | None, co_footprints: int) -> dict | None:
        # No arbiter verb (see arbiter_verbs above) -> no idle rule either:
        # park stays human-only. Structural omission made explicit, the
        # same posture as arbiter_verbs()'s empty frozenset — a hipfire
        # tenant is simply never a candidate for ANY automatic action.
        return None

    def reclaimable(self, obs: dict, gpu: dict | None, co_footprints: int) -> int | None:
        # Never called in practice: app.arbiter._eviction_candidates skips
        # any kind whose arbiter_verbs() is empty before it would reach a
        # reclaimable() call. Defined anyway for adapter-interface
        # completeness (every kind implements the same protocol).
        return None

    def build_client(self, connection: dict, settings):
        # DockerCtl/LiteLLMClient built fresh here rather than reusing
        # app.main._build_deck's shared instances (which still exist,
        # unchanged, for the pre-E1 actuation path) — both are stateless
        # besides their own httpx.Client, so a second instance behaves
        # identically for status()/stats() reads. See
        # app.local_clients' module docstring for the one known
        # transitional gap (HipfireClient's own conversation-activity
        # tracker), out of scope for this fix.
        container = connection["container"]
        dockerctl = DockerCtl(settings.dockerctl_url, settings.park_allowlist)
        litellm = LiteLLMClient(settings.litellm_url, settings.litellm_key)
        return HipfireClient(
            health_url=f"http://{container}:{_HIPFIRE_PORT}/health",
            dockerctl=dockerctl,
            container=container,
            litellm=litellm,
            stats_url=f"http://{container}:{_HIPFIRE_PORT}/stats",
            activity_window_s=settings.hipfire_activity_window_s,
        )

    def restart_container(self, entry: dict) -> str | None:
        # hipfire has no gguf store to register files into.
        return None

    def restore(self, client, model: str | None) -> None:
        """The REAL restore call for a hipfire-kind resource: resume its
        container (app/engines/hipfire.py:162). `model` is unused
        (hipfire's intent always carries model=None) — kept for signature
        parity with lemonade-kind's restore above. Moved from
        Watcher._restore's hipfire branch (E1 Task 6)."""
        client.resume()


# kind -> adapter instance. One instance per kind is enough — adapters carry
# no per-resource state (that lives in World's `_mem`, keyed by resource).
ENGINE_KINDS: dict[str, object] = {
    "lemonade": _LemonadeAdapter(),
    "comfyui": _ComfyAdapter(),
    "hipfire": _HipfireAdapter(),
}
