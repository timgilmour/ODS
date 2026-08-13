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
signature whether or not that particular kind needs the resource name."""

from app.engines import EngineError
from app.engines.comfyui import ComfyClient
from app.engines.docker_ctl import DockerCtl
from app.engines.hipfire import HipfireClient
from app.engines.lemonade import LemonadeClient
from app.engines.litellm import LiteLLMClient
from app.registry import HIPFIRE_FOOTPRINT

_OPENAI_PREFIX = "openai/"
_EXTRA_PREFIX = "extra."

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

    def arbiter_verbs(self) -> frozenset:
        return frozenset({"unload"})

    def human_verbs(self) -> frozenset:
        return frozenset({"load", "unload"})

    def demand(self) -> bool:
        return True

    def build_client(self, connection: dict, settings):
        # Moved from app.local_clients._build_client (review fix, T3 round
        # 2) — same constructor call app.main._build_deck used to make
        # once. `settings.lemonade_key` is a deck-level shared credential,
        # not part of the per-resource declared connection.
        return LemonadeClient(connection["url"], settings.lemonade_key,
                              metrics_url=connection["metrics_url"])


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

    def arbiter_verbs(self) -> frozenset:
        return frozenset({"free"})

    def human_verbs(self) -> frozenset:
        return frozenset({"free"})

    def demand(self) -> bool:
        return False

    def build_client(self, connection: dict, settings):
        return ComfyClient(connection["url"])


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

    def arbiter_verbs(self) -> frozenset:
        return frozenset()

    def human_verbs(self) -> frozenset:
        return frozenset({"park", "resume"})

    def demand(self) -> bool:
        return False

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


# kind -> adapter instance. One instance per kind is enough — adapters carry
# no per-resource state (that lives in World's `_mem`, keyed by resource).
ENGINE_KINDS: dict[str, object] = {
    "lemonade": _LemonadeAdapter(),
    "comfyui": _ComfyAdapter(),
    "hipfire": _HipfireAdapter(),
}
