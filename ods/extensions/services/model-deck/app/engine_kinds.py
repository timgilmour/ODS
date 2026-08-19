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
"lemonade" is the only GGUF server that could exist.

Two more methods joined the protocol on the sglang-omni branch, recorded
HERE so the next reader finds the whole surface in one place:

* `unknown()` (Task 6) — REQUIRED of every kind: that kind's own "we failed
  to look" record, the exact dict its own `observe` returns on
  ``EngineError``, callable by a caller holding NO client at all (a
  DECLARED engine on a node the deck cannot reach must still appear, as
  unknown). Only this module knows a kind's record SHAPE; a caller
  synthesizing `{"state": "unknown"}` would be guessing one, and
  tests/test_engine_kinds.py proves each kind's `unknown()` equals its own
  `observe()` on an engine failure so the two can never drift.
* `build_remote_client(address, credential, resource, connection)` (Task 6,
  first implemented Task 7) — OPTIONAL: the constructor for a kind declared
  on a node-agent entry, called by `app.node_clients._adapter_remote_client`
  exactly as `build_client` is called by `app.local_clients` for local ones.
  A kind that omits it cannot run off-box at all, and
  `RemoteEngineClients.client_for` answers None ("not operable") for a
  declaration naming it — never a crash. The three E1 kinds omit it;
  sglang-omni is the first that has one (and, symmetrically, the first whose
  `build_client` REFUSES: it speaks the node-agent wire, so there is nothing
  to build for a local declaration)."""

from datetime import UTC, datetime, timedelta

from app.engines import EngineError, GuardError
from app.engines.comfyui import ComfyClient
from app.engines.docker_ctl import DockerCtl
from app.engines.hipfire import HipfireClient
from app.engines.lemonade import LemonadeClient
from app.engines.litellm import LiteLLMClient
from app.engines.sglang_omni import SglangOmniClient
from app.node_store import consented_containers
from app.observe import local_key, node_key
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

# What an sglang-omni-kind engine holds while resident, MEASURED (GF5, the
# 2026-08-16 gate run on sparky): ~62 GiB — weights plus overhead, with the
# KV ceiling at 36 G under --mem-fraction-static 0.30. Declared rather than
# inferred from GPU usage (comfyui's approach) because the GPU belongs to
# another machine: its unexplained usage is that box's arithmetic, not this
# deck's. One number for the kind, not per resource — a second sglang-omni
# declaration serving a different checkpoint would need this to become
# declaration-sourced, which is why `reclaimable` returns it rather than
# callers reading the constant.
_SGLANG_OMNI_FOOTPRINT = 62 * 1024**3

# How long after a deliberate load a `down` observation may still be called
# "booting" (see _SglangOmniAdapter.warming). GF4 measured ~3.5-4.5 min cold
# container-start to healthy; doubled, the same generosity margin
# app.engines.spark's _BOOT_WINDOW_MAX_S takes over its own worst observed
# boot — the cost of being too generous is a delayed restore, the cost of
# being too tight is restore-storming a booting engine into quarantine.
_SGLANG_OMNI_WARM_WINDOW_S = 600


def _within(ts: str | None, now: datetime | None, window_s: float) -> bool:
    """Whether ISO timestamp `ts` is less than `window_s` ago.

    Unparseable/absent reads FALSE (not recent) — see
    `_SglangOmniAdapter.warming` for why this fails the opposite way from
    app.engines.spark's `_recent`, which shares this shape.
    """
    if not isinstance(ts, str) or not ts:
        return False
    try:
        stamped = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) - stamped < timedelta(seconds=window_s)


# kind -> {"connection": {field -> required?}, "remote_capable": bool,
#          "local_capable": bool}
#
# WHERE each kind may RUN, declared in BOTH directions and enforced by
# `validate_engines` below (the deck's registry write gate reaches it
# through app/node_store.py's `_validate`/`_heal_engines`):
#
# * `remote_capable` (E1 Task 5) — may it be declared on a node-agent
#   entry's engines[]? The E1 triple all run in-process with the deck
#   (lemonade/comfyui/hipfire are sibling containers on THIS box), so none
#   of them may. "sglang-omni" (Task 7) is the first that may.
# * `local_capable` (Task 7 fix round 1, review finding 2) — may it be
#   declared on the LOCAL entry? Every E1 kind may; sglang-omni may NOT,
#   because it has no local client to build at all (every probe and every
#   verb rides a node-agent). One flag described only the remote direction,
#   so nothing refused the mirror-image mistake, and the refusal landed
#   instead at `build_client` — i.e. as a 422 on /api/state, /api/storage,
#   /api/sets and the lifecycle routes, plus a tick-error every ~2 s, until
#   an operator found and deleted the entry. Guard at the BOUNDARY: the
#   declaration is refused where it is written.
#
# Both flags are served by GET /api/engine-kinds, so a picker can filter by
# the node it is editing rather than offering a kind the write gate refuses.
# A kind with both False could be declared nowhere; tests/test_engine_kinds.py
# pins that as a shape error rather than leaving it representable.
KNOWN_KINDS: dict[str, dict[str, object]] = {
    "lemonade": {"connection": {"url": True, "metrics_url": True, "container": True},
                 "remote_capable": False, "local_capable": True},
    "comfyui": {"connection": {"url": True},
                "remote_capable": False, "local_capable": True},
    "hipfire": {"connection": {"container": True},
                "remote_capable": False, "local_capable": True},
    # One declared field, deliberately: everything the PROBE needs
    # (health_url, the busy-count port, the compose file) lives in the
    # node's own host-owned engines.json allowlist (ruling A1) — the deck
    # cannot name a compose file to run. `url` is the operator's record of
    # where this engine serves, carried in the declaration so the board can
    # show it; the deck never dials it directly (it talks to the agent).
    "sglang-omni": {"connection": {"url": True},
                    "remote_capable": True, "local_capable": False},
}

_POLICY_FIELDS = {"priority": int, "pinned": bool, "idle_ttl": int}


def _bad(reason: str) -> ValueError:
    return ValueError(reason)


def validate_engines(engines: object, remote: bool = False) -> None:
    """Raise ValueError (one-line reason) unless `engines` is a valid
    declaration list. Refuse, never coerce ([[literal-declared-inputs]]).

    `remote` (E1 Task 5): True when this list is being declared on a
    node-agent entry, not the local one — every kind named must then also
    be `remote_capable`, else refused BY NAME (the operator needs to know
    which declared kind can't run off-box, not just that "something" is
    wrong). False checks the MIRROR of that (Task 7 fix round 1): a
    remote-only kind is refused on the local entry, equally by name, and
    the message says where it does belong. One-directional checking is what
    let a local sglang-omni declaration through to fail later at
    client-construction time — see KNOWN_KINDS' own comment."""
    if not isinstance(engines, list):
        raise _bad("engines must be a list")
    seen: set[str] = set()
    for e in engines:
        if not isinstance(e, dict):
            raise _bad("engine entry must be an object")
        extra = set(e) - {"resource", "kind", "connection", "gpu_index",
                          "policy_defaults", "container_consent"}
        if extra:
            raise _bad(f"engine entry has extra field(s): {sorted(extra)}")
        resource = e.get("resource")
        if (not isinstance(resource, str) or not resource
                or "/" in resource or resource != resource.strip()):
            raise _bad("resource must be a non-empty string without '/'")
        if resource == "slot0":
            # "slot0" is the serving slot's OWN reserved observation name
            # (app.observe.slot_key: "<node>/slot0"), never a name a
            # DECLARED engine may also claim — a swap-capable node that
            # both swaps AND declares an engine resource literally called
            # "slot0" would collide with its own slot at that one
            # observation key across merge_observations (last-wins: the
            # engine's own observation silently vanishes), the intent
            # store (one record per key: an engine load overwrites the
            # slot's swap intent), and the board (duplicate React keys).
            # Checked here, in the resource-shape block, before the
            # kind/remote_capable checks below — the reservation is
            # kind- and location-independent, so it must not be reachable
            # by declaring a kind or direction the later checks happen to
            # allow.
            #
            # This is the ONE gate: `app.node_store._heal_engines` and
            # `_validate` both route every engines[] list through THIS
            # function before it is ever persisted, so a hand-written
            # nodes.json carrying a "slot0" resource heals the same way any
            # other schema-invalid list does (to []) with no second guard
            # written there — see tests/test_node_store.py's slot0 heal
            # test. Do not add a duplicate check at another call site.
            raise _bad(
                "resource 'slot0' is reserved: it is the serving slot's "
                "observation key ('<node>/slot0'), and a declared engine "
                "using it would collide with the slot's own observation, "
                "intent record, and board placement at that same key")
        if resource in seen:
            raise _bad(f"duplicate resource {resource!r}")
        seen.add(resource)
        kind = e.get("kind")
        if kind not in KNOWN_KINDS:
            raise _bad(f"unknown kind {kind!r} (known: {sorted(KNOWN_KINDS)})")
        if remote and not KNOWN_KINDS[kind]["remote_capable"]:
            raise _bad(f"{resource}: kind {kind!r} is not remote_capable")
        if not remote and not KNOWN_KINDS[kind]["local_capable"]:
            raise _bad(f"{resource}: kind {kind!r} is remote-only — declare it "
                       f"on a node-agent entry, not on the local node")
        schema = KNOWN_KINDS[kind]["connection"]
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
        consent = e.get("container_consent")
        if not isinstance(consent, bool):
            raise _bad(f"{resource}: container_consent must be a boolean — "
                       "explicit consent for the deck to stop/start this "
                       "entry's container (docker_ctl's guard reads it live)")
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
            return self.unknown()

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

    def unknown(self) -> dict:
        """This kind's own "we failed to look" record — the exact dict
        ``observe`` returns on ``EngineError``, named (sglang-omni Task 6)
        so a caller holding NO client at all can still report it.

        A DECLARED engine on a node the deck cannot reach — its agent did
        not answer, its credential vanished, its kind has no remote
        constructor — must still appear in the world, as unknown. Only this
        module knows what a record of THIS kind looks like
        (state/model/footprint/idle_s); a caller synthesizing a bare ``{"state": "unknown"}`` would be
        guessing that shape, and a guess that drifts from the real one is
        how a downstream KeyError gets in. ``observe``'s own EngineError
        arm returns THIS, so the two can never disagree.
        """
        return {"state": "unknown", "model": None, "footprint": None, "idle_s": None}

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

    def build_client(self, connection: dict, settings, node_store):
        # Moved from app.local_clients._build_client (review fix, T3 round
        # 2) — same constructor call app.main._build_deck used to make
        # once. `settings.lemonade_key` is a deck-level shared credential,
        # not part of the per-resource declared connection. `node_store` is
        # accepted-and-ignored: only the hipfire adapter's DockerCtl needs a
        # live consent source (open-rulings #1) — this kind has no Docker
        # involvement.
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
            return self.unknown()

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

    def unknown(self) -> dict:
        """This kind's own "we failed to look" record — the exact dict
        ``observe`` returns on ``EngineError``, named (sglang-omni Task 6)
        so a caller holding NO client at all can still report it.

        A DECLARED engine on a node the deck cannot reach — its agent did
        not answer, its credential vanished, its kind has no remote
        constructor — must still appear in the world, as unknown. Only this
        module knows what a record of THIS kind looks like
        (state/queue/idle_s); a caller synthesizing a bare ``{"state": "unknown"}`` would be
        guessing that shape, and a guess that drifts from the real one is
        how a downstream KeyError gets in. ``observe``'s own EngineError
        arm returns THIS, so the two can never disagree.
        """
        return {"state": "unknown", "queue": None, "idle_s": None}

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

    def build_client(self, connection: dict, settings, node_store):
        # `node_store` accepted-and-ignored — see _LemonadeAdapter's
        # build_client docstring: only hipfire's DockerCtl needs it.
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

    # THE DECK DOES NOT CHOOSE THIS KIND'S MODEL, so nothing may record one
    # as its intent. hipfire is single-model and which model it serves comes
    # from the LiteLLM route table (`observe` above reads it out of `ctx`),
    # not from any deck decision — `None` reads as "loaded, no opinion which
    # model" in app.lifecycle.derive_status, and a NAME would report drift
    # for a perfectly healthy engine the moment somebody re-pinned it through
    # the dashboard. `_hipfire_park`/`_hipfire_resume` (app/routers/control.py)
    # and `restore` below already hold this invariant by hand; this attribute
    # is what lets a GENERIC writer — app.routers.lifecycle's adopt, whose
    # observation does carry the observed name — honour it without knowing
    # the string "hipfire". Absent on every other adapter (and on kinds with
    # no adapter row at all, e.g. "spark"), where the model IS the deck's
    # opinion; readers default it to True, the same optional-attribute shape
    # `warming` uses.
    deck_pins_model = False

    def observe(self, client, mem: dict, now: float, ctx: dict) -> dict:
        # Moved VERBATIM from app.state.World._snapshot_hipfire (app/state.py,
        # pre-Task-3 lines 213-232). `routes` and `resource` (for the
        # route-table model lookup, see this module's docstring) come from
        # `ctx`; hipfire has no per-resource idle-clock state, so `mem` is
        # unused here (kept in the signature for adapter-interface uniformity).
        try:
            state = client.status()
        except EngineError:
            return self.unknown()

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

    def unknown(self) -> dict:
        """This kind's own "we failed to look" record — the exact dict
        ``observe`` returns on ``EngineError``, named (sglang-omni Task 6)
        so a caller holding NO client at all can still report it.

        A DECLARED engine on a node the deck cannot reach — its agent did
        not answer, its credential vanished, its kind has no remote
        constructor — must still appear in the world, as unknown. Only this
        module knows what a record of THIS kind looks like
        (state/model/footprint/queue_depth — footprint 0, not None: it is
        typed a plain int in the snapshot shape); a caller synthesizing a bare ``{"state": "unknown"}`` would be
        guessing that shape, and a guess that drifts from the real one is
        how a downstream KeyError gets in. ``observe``'s own EngineError
        arm returns THIS, so the two can never disagree.
        """
        return {"state": "unknown", "model": None, "footprint": 0, "queue_depth": None}

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

    def build_client(self, connection: dict, settings, node_store):
        # DockerCtl/LiteLLMClient built fresh here rather than reusing
        # app.main._build_deck's shared instances (which still exist,
        # unchanged, for the pre-E1 actuation path) — both are stateless
        # besides their own httpx.Client, so a second instance behaves
        # identically for status()/stats() reads. See
        # app.local_clients' module docstring for the one known
        # transitional gap (HipfireClient's own conversation-activity
        # tracker), out of scope for this fix.
        #
        # `allowed` closes over `node_store` (received from
        # LocalClients.client_for, which already holds it) so the guard
        # resolves container_consent LIVE, never a snapshot captured here
        # at build time (open-rulings #1; same rationale as app.main's own
        # DockerCtl construction).
        container = connection["container"]
        dockerctl = DockerCtl(settings.dockerctl_url,
                              lambda: consented_containers(node_store))
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
        (hipfire's intent always carries model=None — the invariant
        `deck_pins_model = False` at the top of this class states once, for
        every writer) — kept for signature parity with lemonade-kind's
        restore above. Moved from
        Watcher._restore's hipfire branch (E1 Task 6)."""
        client.resume()


class _SglangOmniAdapter:
    """sglang-omni-kind: a whole ENGINE brought up/down on another box
    through its node-agent (load/unload = the container's compose lifecycle),
    idle-clock tracked by the agent's in-flight request count.

    The first kind that does not run beside the deck. Three consequences
    shape everything below:

    * **Every probe is one node-agent call.** `client` is an
      app.engines.sglang_omni.SglangOmniClient, and `status()` answers the
      agent's wire dict verbatim: ``{"reachable", "healthy",
      "busy_requests"}``. There is no second surface to ask (GF1: the engine
      serves ``/health`` and ``/v1/*`` and nothing else), so this adapter
      reads exactly those three fields.
    * **The busy indicator can be unavailable, and that reads as BUSY.**
      GF3 closed negative: MiniMax-Music3 has no metrics endpoint at this
      pin, so "in flight" is an agent-side count of established connections
      on the serving port. `None` means the agent could not take that count
      — design §4 binds the unavailable indicator to fail toward ALIVE,
      because a song render runs for minutes and holds its connection the
      whole time. Reading `None` as 0 would idle-release a running render.
      The invariant lives in `reclaimable` (below), not at the call sites.
    * **A load takes minutes, not seconds.** GF4: ~3.5-4.5 min cold. See
      `warming` for the window that keeps the reconciler from calling a boot
      in progress a death.
    """

    def observe(self, client, mem: dict, now: float, ctx: dict) -> dict:
        # `ctx` is unused for this kind (no registry footprint to look up —
        # the footprint is declared, see reclaimable — and no litellm route
        # table: that one is the LOCAL box's, and World.snapshot_remote
        # hands remote kinds None for it). Kept in the signature for adapter
        # -interface uniformity, same as hipfire's unused `mem`.
        try:
            status = client.status()
        except EngineError:
            # We failed to LOOK. That is not an idle observation, so the
            # activity clock is re-armed on the way out (see the state arms
            # below for why every non-idle answer re-arms).
            mem["last_activity_time"] = now
            return self.unknown()

        busy = status["busy_requests"]
        if not status["healthy"]:
            # Covers both "the agent reached the port but /health is not
            # 200" and "nothing is listening" (`reachable` False) — the deck
            # acts on the same fact either way: the engine is not serving.
            # Distinguishing them would buy nothing an operator can use and
            # a state the reconciler cannot act on differently.
            #
            # Re-armed rather than accrued: an engine that was DOWN for ten
            # minutes and has only now come up healthy has been idle for
            # seconds. Accruing across the down window would put it past any
            # sane idle_ttl on its FIRST healthy tick, i.e. the deck
            # unloading what it just spent ~4 minutes (GF4) booting.
            mem["last_activity_time"] = now
            return {"state": "down", "busy_requests": busy, "model": None,
                    "idle_s": None}

        if busy is None or busy > 0:
            # `None` lands here deliberately (design §4, this class's own
            # docstring): unavailable fails toward alive, clock included —
            # a run of unavailable indicators must not quietly accrue the
            # idle time that authorizes an unload.
            mem["last_activity_time"] = now
            state = "busy"
        else:
            state = "idle"
            if mem.get("last_activity_time") is None:
                # First-ever observation: establish a baseline so idle_s is
                # always computable from here on (the comfyui-kind
                # precedent), even when the very first look finds it idle.
                mem["last_activity_time"] = now

        return {
            "state": state,
            "busy_requests": busy,
            # GF2: the engine's own model id is `/model`, the container's
            # mount path rather than an identity — never trusted. The
            # declaration carries no model name either (one engine, one
            # checkpoint, chosen by the node's compose file), so identity is
            # None exactly as it is for every other single-model engine;
            # app.lifecycle.derive_status's "loaded, no opinion which model"
            # branch is what consumes that.
            "model": None,
            "idle_s": now - mem["last_activity_time"],
        }

    def unknown(self) -> dict:
        """This kind's own "we failed to look" record — the exact dict
        ``observe`` returns on ``EngineError``, named (sglang-omni Task 6)
        so a caller holding NO client at all can still report it.

        A DECLARED engine on a node the deck cannot reach — its agent did
        not answer, its credential vanished, its kind has no remote
        constructor — must still appear in the world, as unknown. Only this
        module knows what a record of THIS kind looks like (state/
        busy_requests/model/idle_s); a caller synthesizing a bare
        ``{"state": "unknown"}`` would be guessing that shape, and a guess
        that drifts from the real one is how a downstream KeyError gets in.
        ``observe``'s own EngineError arm returns THIS, so the two can never
        disagree.
        """
        return {"state": "unknown", "busy_requests": None, "model": None,
                "idle_s": None}

    def active(self, obs: dict) -> bool:
        # Idle counts as active: the engine holds ~62 GiB (GF5) between
        # renders, so "idle" describes its REQUEST queue, not its residency
        # — the same reading app.observe's module docstring already spells
        # out for comfyui. `observe` emits only busy/idle/down/unknown, so
        # these two ARE the resident states; there is no separate "loaded".
        return obs["state"] in ("busy", "idle")

    def uses_gguf(self, obs: dict) -> str | None:
        # sglang-omni serves a HF checkpoint mounted into its container on
        # another box, never a GGUF unit in this box's storage catalog.
        # Defined for adapter-interface completeness.
        return None

    def arbiter_verbs(self) -> frozenset:
        return frozenset({"unload"})

    def human_verbs(self) -> frozenset:
        # load/unload ONLY — never park/resume. livetests/test_safe_sets.py's
        # FORBIDDEN_STEPS keeps its meaning because of this line: "park"/
        # "resume" appearing in a plan still means "touches hipfire".
        return frozenset({"load", "unload"})

    def demand(self) -> bool:
        # No pending-load inference: nothing routes chat traffic at this
        # engine (music generation is not chat-completions-shaped, design
        # §8), so there is no "someone asked for a model that isn't loaded"
        # signal to demand one from.
        return False

    def idle_action(self, obs: dict, policy: dict, gpu: dict | None, co_footprints: int) -> dict | None:
        # The comfyui-kind shape, with the busy count in place of the queue
        # depth. `model: None` rides along for the same reason `observe`
        # reports none (see there).
        if (
            obs["state"] == "idle"
            # Belt AND braces with `reclaimable` below: an idle observation
            # whose busy count is UNAVAILABLE must never authorize an
            # unload — a five-minute render is exactly what would be killed.
            and obs["busy_requests"] == 0
            and not policy["pinned"]
            and policy["idle_ttl"] > 0
            and obs["idle_s"] is not None
            and obs["idle_s"] >= policy["idle_ttl"]
            # An unload that can't reclaim anything is a no-op that re-arms
            # the TTL and re-fires forever (the comfyui free-spam lesson).
            # Structurally unreachable today — this kind's footprint is a
            # declared non-zero constant — and kept because the predicate,
            # not the constant, is the contract: a future declared-footprint
            # source (per-node, per-model) could answer 0.
            and self.reclaimable(obs, gpu, co_footprints) != 0
        ):
            return {"type": "unload", "model": obs["model"]}
        return None

    def reclaimable(self, obs: dict, gpu: dict | None, co_footprints: int) -> int | None:
        """The bytes an unload would return to the node, or None when this
        resource is not an eviction/idle candidate at all.

        THE busy-signal invariant's home (design §4 / the plan's Global
        Constraints: "implement in reclaimable, not at call sites"): None
        unless the engine is observed idle AND its busy count is a real 0.
        A `None` count — the agent could not take it — is not proof of an
        idle engine, and a busy one is proof of the opposite.

        `gpu`/`co_footprints` are accepted for interface uniformity and
        ignored, unlike comfyui's: this kind's footprint is DECLARED (GF5,
        measured at the 2026-08-16 gate), not inferred from a GPU's
        unexplained usage — and the GPU in question belongs to another
        machine, whose co-residency arithmetic is that box's own.
        """
        if obs["state"] != "idle" or obs["busy_requests"] != 0:
            return None
        return _SGLANG_OMNI_FOOTPRINT

    def warming(self, intent: dict | None, obs: dict, now: datetime | None = None) -> bool:
        """Whether a `down` observation is really a LOAD STILL IN FLIGHT.

        GF4: a cold start takes ~3.5-4.5 minutes, and for all of it the
        engine observes exactly like one that died — the deck asked the
        agent to bring it up, got a 202, and now sees nothing listening.
        Read as a death, the reconciler restores it every cooldown and
        quarantines it before it ever finishes booting.

        Both conditions are required, and each drops a different false
        reading — the same shape as app.engines.spark's `boot_in_flight`,
        for the same reason:

        * intent must be `loaded` — an engine nobody asked to be up is not
          booting, it is off;
        * the record must be RECENT. This is the one that is easy to miss:
          an intent record persists forever, so "we once asked for this" +
          "it is down" would shield a model that died hours later from
          restore permanently — the 26-hour hipfire failure exactly.

        A missing or unparseable `updated_ts` reads as NOT warming here,
        the opposite of `boot_in_flight`'s "unsure means do not act": what
        is at stake is inverted. There, guessing wrong interrupts a live
        multi-minute swap; here, guessing wrong means never restoring a
        genuinely dead engine, and a redundant restore is cheap (`up()` is
        `docker compose up -d`, a no-op on a container already running).

        NOT wired into the observation path, deliberately: `observe` cannot
        see intent (app.state/app.observe are intent-blind BY DESIGN — that
        separation is why app.lifecycle exists), so the join belongs where
        intent and observation already meet. Task 9 wired it there —
        `app.lifecycle.join_warming`, called by the arbiter's reconcile pass
        and by `app.routers.build_lifecycle_view` — which is why the
        arguments below stay in THIS adapter's own vocabulary (`obs` is a
        world tenant, not a `derive_status` record): the join adapts, the
        rule does not have to be re-expressed in a shape it does not own.
        """
        if intent is None or intent.get("state") != "loaded":
            return False
        if obs["state"] != "down":
            return False
        return _within(intent.get("updated_ts"), now, _SGLANG_OMNI_WARM_WINDOW_S)

    def build_client(self, connection: dict, settings, node_store):
        """REFUSED, by name: this kind speaks the node-agent wire, so there
        is no local client to build for a declaration on the local entry.

        Defined rather than omitted so the failure is a one-line refusal
        naming the kind, instead of the bare AttributeError an absent method
        would raise from inside app.local_clients on every world snapshot.

        UNREACHABLE since fix round 1: `local_capable: False` makes
        `validate_engines` refuse a local declaration of this kind at the
        BOUNDARY — through the write gate (422 on the declaration) and
        through `_heal_engines` (a hand-edited nodes.json heals to []), so
        no local entry naming this kind can reach a client build. Kept as
        the totality floor anyway, the same posture app.arbiter's
        `_dispatch_verb` raise takes for its own can't-happen branch: a
        future path that skips the gate must fail loudly and by name, not
        silently produce a client that cannot work.
        """
        raise ValueError(
            "kind 'sglang-omni' has no local client: it runs through a "
            "node-agent — declare it on a node-agent entry, not on 'local'")

    def build_remote_client(self, address: str, credential: str, resource: str,
                            connection: dict):
        """This kind's remote constructor, called by
        app.node_clients._adapter_remote_client (the remote counterpart of
        `build_client`). `connection` is accepted and unused: the probe's
        own address is the node's engines.json entry, host-owned per ruling
        A1, and the declared `url` is the operator's record of where the
        engine serves (see KNOWN_KINDS' comment)."""
        return SglangOmniClient(address, credential, resource)

    def restart_container(self, entry: dict) -> str | None:
        # No GGUF store to register a moved-in file into, and the container
        # is on another box besides — app.notify walks the LOCAL declaration
        # only. Defined for adapter-interface completeness.
        return None

    def execute_unload(self, watcher, resource: str, client, model: str | None,
                       actuated: set[str], *, node_id: str) -> None:
        """Bring the engine DOWN on its node, recording intent first.

        `node_id` is keyword-only and REQUIRED — the one signature
        difference from the lemonade-kind actuator, and deliberate: this
        kind's intent key is ``<node>/<resource>``, and there is no safe
        default for the node half. A caller that forgets it gets a TypeError
        (loud, at the call site) rather than a silently mis-keyed record
        against the LOCAL node's resource of the same name — the exact
        cross-node collision `node_key` exists to prevent.

        `model` is accepted for signature parity with the lemonade-kind
        actuator and is always None for this kind (one engine, one
        checkpoint — see `observe`).
        """
        key = node_key(node_id, resource)
        # Whoever actuates, records — and records FIRST, so a tick landing
        # mid-unload derives 'parked', never 'down'. actor="deck": this is
        # the arbiter's OWN automatic action (idle release), never an
        # operator's. Prior record snapshotted so a FAILED unload can roll
        # it back.
        prior = None
        if watcher._intent_store is not None:
            prior = watcher._intent_store.get().get(key)
            watcher._intent_store.record(
                key, state="unloaded", model=None, engine="sglang-omni",
                actor="deck")
        # Added BEFORE the engine call so the invariant stays structural.
        actuated.add(key)
        try:
            client.down()
        except EngineError as exc:
            # Per-action isolation: a raise here must not abort the
            # remaining actions or the reconcile/derive passes. Logged
            # BEFORE the rollback — the diagnostic must not be contingent on
            # the rollback succeeding. `node` rides in the detail alongside
            # `resource` for the same reason `resource` does on the
            # lemonade-kind arm: `_log`'s failure-dedup memo keys on the
            # detail, so two nodes' same-named resources failing identically
            # must not collapse into one line.
            watcher._log("unload-failed",
                         {"node": node_id, "resource": resource,
                          "error": _error_text(exc)})
            # A real compare-and-swap (predicate + write in ONE critical
            # section): an operator can record a deliberate load during the
            # seconds this call hung, and blindly putting `prior` back would
            # silently revert THEIR action. Still the speculative record
            # this arm wrote? It is deck-authored AND unloaded; an
            # operator's write flips actor to "operator".
            def _still_ours(current):
                return (current is not None
                        and current.get("actor") == "deck"
                        and current.get("state") == "unloaded")

            if watcher._intent_store is not None and prior is not None:
                if not watcher._intent_store.put_back_if(key, _still_ours, prior):
                    watcher._log("unload-rollback-skipped",
                                 {"node": node_id, "resource": resource,
                                  "reason": "intent changed during the unload"})
        else:
            # A successful unload between two identical failures must not
            # swallow the second one.
            watcher._clear_failure_dedup()
            # RE-ARM THE IDLE CLOCK AND DROP THE CACHED OBSERVATION (Task 9),
            # the comfyui-kind free's own pair of obligations, sharpened by
            # this kind being REMOTE. The observation is TTL-cached
            # (app.node_clients.RemoteObserver, 10 s) and `down()` is a 202 —
            # the container is still going down for seconds afterwards — so
            # the very same idle-past-ttl record is what every tick in that
            # window reads, and the idle rule would re-fire on it every ~2 s
            # (the comfyui free-spam lesson, with a whole engine at the other
            # end). Invalidating alone is not enough (the fresh observation
            # can still be idle-past-ttl while the container stops) and
            # re-arming alone is not enough (a cached half never re-reads the
            # clock at all): both, or the storm survives.
            #
            # The deck-wide `invalidate()` is what exists (no per-node form —
            # T8 review M-2); its cost is clearing every node's probe backoff,
            # paid at most once per idle_ttl per resource, which is nothing
            # beside a duplicate whole-engine unload.
            watcher._world.note_freed(resource, node_id=node_id)
            if watcher._remote_observer is not None:
                watcher._remote_observer.invalidate()
            watcher._log(f"unload_{resource}", {"node": node_id})
            # NOT armed: watcher._heal_suppressor. That window exists to stop
            # the LOCAL contention-healing path from re-loading a deliberately
            # unloaded lemonade model (HealSuppressor's own docstring names
            # the kind); arming it from a remote node's unload would suppress
            # this box's healing for another box's action.

    def execute_load(self, watcher, resource: str, client, pending: dict,
                     actuated: set[str], *, node_id: str) -> None:
        """Bring the engine UP on its node, recording intent first. See
        `execute_unload` for what `node_id` is and why it is required.

        `pending` is accepted for signature parity with the lemonade-kind
        actuator and its `model` is deliberately NOT read: this kind serves
        one checkpoint, chosen by the node's compose file, so the recorded
        model is always None (see `observe`). Reading a caller's model here
        would write an identity the engine can never be compared against,
        and `derive_status` would then report permanent drift.

        The arbiter never reaches this today (`demand()` is False, so there
        is no pending-load retrigger for this kind); it is the load half of
        the same record-first discipline, for the reconcile/heal paths Task
        9 wires.
        """
        key = node_key(node_id, resource)
        if watcher._intent_store is not None:
            watcher._intent_store.record(
                key, state="loaded", model=None, engine="sglang-omni",
                actor="deck")
        actuated.add(key)
        try:
            client.up()
        except EngineError as exc:
            # No rollback here, the same deliberate asymmetry the
            # lemonade-kind load arm documents: a failed load derives 'down'
            # and the reconciler retries under the existing failure budget.
            # Forgetting the intent would make the deck forget it ever
            # wanted the engine up.
            watcher._log("load-failed",
                         {"node": node_id, "resource": resource,
                          "error": _error_text(exc)})
        else:
            watcher._clear_failure_dedup()
            watcher._log("load-retriggered", {"node": node_id,
                                              "resource": resource})

    def restore(self, client, model: str | None) -> None:
        """The REAL restore call for an sglang-omni-kind resource: ask the
        node to bring the engine up. `model` is unused (this kind's intent
        always carries model=None) — kept for signature parity with the
        other kinds' restore. 202-async by contract: the request is queued
        for the node's swap-helper and this call does not wait for it (see
        app/engines/sglang_omni.py's module docstring — the deck's own
        reconciler paces retries, a client that retried here would fight
        it)."""
        client.up()


# kind -> adapter instance. One instance per kind is enough — adapters carry
# no per-resource state (that lives in World's `_mem`, keyed by resource).
ENGINE_KINDS: dict[str, object] = {
    "lemonade": _LemonadeAdapter(),
    "comfyui": _ComfyAdapter(),
    "hipfire": _HipfireAdapter(),
    "sglang-omni": _SglangOmniAdapter(),
}
