"""
Model Deck world-state snapshot.

``World.snapshot()`` assembles one point-in-time view of every DECLARED
local engine (see ``app.engine_kinds``) plus GPU VRAM totals and a
best-effort "externals" list, from already-fetched engine/GPU data. This
module does no I/O of its own — the caller owns sysfs reads
(``app.gpu.read_gpus``) and every engine HTTP call (via ``clients``,
app.local_clients.LocalClients), and passes the results/clients in. That
keeps the aggregation logic here pure and cheap to test, following the
repo's functional-core/imperative-shell convention.

E1 generalization (Task 3): a ``World`` no longer knows lemonade/comfyui/
hipfire by name. It iterates whatever ``engines`` (the declaration list)
holds, asks ``app.engine_kinds.ENGINE_KINDS[kind]`` to observe each one,
and keys everything by RESOURCE, not kind — two lemonade-kind resources
are two independent tenants with independent idle clocks. Absence is
representable: a resource nobody declared has no tenant, no mem entry, no
"unknown" placeholder (spec §1).

A ``World`` instance holds one piece of in-memory state across calls,
``self._mem``: a dict keyed by RESOURCE of each adapter's own idle-clock
bookkeeping (lemonade's last-seen activity-counter value/timestamp/loaded
model, comfyui's last-busy timestamp — see app.engine_kinds' adapters for
what each kind actually stores there). Driven by an injectable ``clock``
(defaults to ``time.monotonic``) so tests can move time forward
deterministically without real sleeps — a fresh ``World()`` per process is
expected to be re-created at most once (the arbiter/watcher owns exactly
one instance for its whole lifetime); idle clocks reset to "just started"
if the process restarts. A resource that disappears from the declaration
has its mem entry dropped (re-declaring it later starts a fresh clock,
same as a process restart would).

Per-engine failures: ``EngineError`` raised by an engine call during a
snapshot degrades only that tenant to ``state="unknown"`` (fields that
have no sensible zero value go to ``None``; hipfire's ``footprint`` is
typed as a plain ``int`` in the snapshot shape, not ``int | None``, so it
goes to ``0`` instead) rather than failing the whole snapshot. This is the
one exception type this module catches. Any other exception (e.g. a bare
``KeyError`` from a malformed response body) is a real bug and is allowed
to propagate per the house "let it crash" policy — the watcher loop that
calls ``snapshot()`` on a timer is expected to catch per-tick, not this
module.

hipfire's ``model`` name and ``default_route`` both come from
``litellm.route_table()``. A litellm failure there does NOT downgrade
hipfire's ``state`` to "unknown" — hipfire's operational state came back
fine from ``HipfireClient.status()`` independently, so only the ``model``
name (which litellm alone can supply) and ``default_route`` go to
``None``. Only a failure of hipfire's *own* status call marks the hipfire
tenant "unknown".

Externals (minimal heuristic, display-only — see task brief for the full
version this deliberately simplifies): a KFD pid using more than 1 GiB
counts as "external" only when NO tenant anywhere reports itself "active"
(per that tenant's adapter's ``active(obs)`` — lemonade "loaded", comfyui
"busy", hipfire "running"). This heuristic doesn't scope to "the GPU that
tenant is actually on" (that would need attributing ``app.gpu.read_gpus``
pids to specific tenants, which is explicitly out of scope here) even
though ``placement`` now carries the resource->GPU index mapping.
Practical effect: as soon as any tenant is loaded/running anywhere in the
box, all fat pids on all GPUs are treated as accounted-for, even ones on a
GPU with no tenant activity at all — this under-reports externals whenever
a tenant is loaded on one GPU while unrelated heavy VRAM use happens on
another. Acceptable for a display-only UI list; do not use this for
eviction decisions.

``snapshot_remote()`` (sglang-omni Task 6) is the same assembly for
engines DECLARED on a registry entry OTHER than the local one, kept as its
OWN half of the world (``remote_tenants``, keyed ``<node>/<resource>``,
with a per-node GPU pool the caller supplies) rather than merged into
``tenants``/``gpus``: a remote engine's ``gpu_index`` addresses ITS OWN
box's GPU list, so merging would let the arbiter's co-residency arithmetic
compare two machines' GPU 0 — and a hand-edited registry can still hold two
nodes' same-named resources (the declaration boundary refuses that
deck-wide since Task 9's ruling R10, and ``NodeStore._load`` heals a
hand-edit, but this assembly must not depend on either), which merging
would collapse into one tenant. It keeps its own ``_remote_mem`` for the
same shared-instance reason spelled out at that attribute.

No Settings import here — pure inputs only.
"""

import time

from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError
from app.observe import node_key

# Externals heuristic floor: below this, transient/small allocations are
# noise, not worth surfacing as a tenant-shaped list entry.
_EXTERNAL_FLOOR_BYTES = 1 * 1024**3  # 1 GiB

_OPENAI_PREFIX = "openai/"

# Sentinel key inside each resource's `self._mem[resource]` dict recording
# which KIND last wrote it — lets `snapshot` detect an in-place kind change
# (same resource, re-declared under a different kind) and reset the entry
# instead of handing the new kind's adapter another kind's stale
# bookkeeping (review fix, T3 round 2). Leading underscore only as a
# naming convention (adapters never inspect the full mem dict, only their
# own named keys via .get()/assignment, so this coexists harmlessly).
_KIND_MEM_KEY = "_kind"


def _strip_prefix(name: str | None, prefix: str) -> str | None:
    if name is None:
        return None
    return name.removeprefix(prefix)


class World:
    """In-memory idle-clock state across repeated ``snapshot()`` calls."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        # resource -> that resource's adapter-owned idle-clock bookkeeping.
        # Keyed by resource (not kind): two lemonade-kind resources get
        # independent entries, independent clocks.
        self._mem: dict[str, dict] = {}
        # The same thing for engines declared on OTHER nodes, keyed
        # "<node>/<resource>" (sglang-omni Task 6). A SEPARATE dict, not a
        # second namespace inside `_mem`, for one load-bearing reason: each
        # half is pruned against the declaration ITS OWN snapshot call was
        # given, and ONE World instance is shared by the arbiter tick and
        # every HTTP path (app.main._build_watcher passes deck["world"]).
        # A caller that snapshots only the local half must not wipe the
        # remote half's idle clocks — sharing one dict would reset every
        # remote resource's clock on every tick that skipped the remote
        # pass, i.e. an idle TTL that never elapses.
        self._remote_mem: dict[str, dict] = {}

    def snapshot(self, gpus, engines, clients, litellm, registry) -> dict:
        now = self._clock()

        gpu_list = [
            {
                "index": gpu["index"],
                "total": gpu["vram_total"],
                "used": gpu["vram_used"],
                "free": gpu["vram_total"] - gpu["vram_used"],
            }
            for gpu in gpus
        ]

        try:
            routes = litellm.route_table()
        except EngineError:
            routes = None

        declared = {e["resource"]: e for e in engines}
        # A resource that disappeared from the declaration since the last
        # snapshot loses its idle-clock memory — re-declaring it later is
        # indistinguishable from a fresh process (matches the docstring's
        # "process restart" framing, generalized to per-resource).
        for gone in set(self._mem) - set(declared):
            del self._mem[gone]

        tenants: dict[str, dict] = {}
        for resource, entry in declared.items():
            kind = entry["kind"]
            adapter = ENGINE_KINDS[kind]
            client = clients.client_for(resource)
            mem = self._mem.setdefault(resource, {})
            # A resource re-declared under a DIFFERENT kind in place (same
            # resource name, new `kind`) must not inherit the old kind's
            # idle-clock bookkeeping — different adapters read/write
            # overlapping mem key NAMES (e.g. both lemonade's and comfy's
            # observe() use "last_activity_time" for unrelated clocks), so
            # a stale value under the new kind reads as a real, wrong idle
            # baseline instead of a fresh one (review fix, T3 round 2).
            # `mem.get(_KIND_MEM_KEY, kind)` defaults to `kind` itself when
            # absent — a brand-new mem entry (or one seeded by note_freed,
            # which sets no _kind marker) is never treated as a mismatch.
            if mem.get(_KIND_MEM_KEY, kind) != kind:
                mem.clear()
            mem[_KIND_MEM_KEY] = kind
            # A fresh per-resource ctx (not one shared dict mutated in
            # place): see app.engine_kinds' module docstring for why
            # `resource` rides in ctx rather than a fifth `observe` param.
            ctx = {"registry": registry, "routes": routes, "resource": resource}
            obs = adapter.observe(client, mem, now, ctx)
            obs["engine"] = kind
            obs["gpu_index"] = entry["gpu_index"]
            tenants[resource] = obs

        default_route = None if routes is None else _strip_prefix(routes.get("default"), _OPENAI_PREFIX)

        externals = self._externals(gpus, engines, tenants)

        return {
            "gpus": gpu_list,
            "tenants": tenants,
            "externals": externals,
            "default_route": default_route,
            # Disambiguates default_route=None: "litellm says there is no
            # default route is configured" (True) vs "we could not reach litellm to ask"
            # (False). The storage guards fail CLOSED on the latter — see
            # app.storage.plan_move / storage_decide.
            "routes_known": routes is not None,
            "placement": {resource: entry["gpu_index"] for resource, entry in declared.items()},
        }

    def snapshot_remote(self, engines, clients, gpu_pools, registry) -> dict[str, dict]:
        """Observe every DECLARED engine on a registry entry OTHER than the
        local one, keyed ``<node>/<resource>`` (app.observe.node_key).
        sglang-omni Task 6.

        ``engines`` is the remote declaration list, each entry stamped with
        its owning ``node_id`` (app.node_clients.remote_engine_declarations).
        ``clients`` is a ``RemoteEngineClients``: ``client_for(node_id,
        resource)`` answers None for a pair that is not operable and NEVER
        raises. ``gpu_pools`` is ``{node_id: [gpu, ...] | None}`` — None
        meaning the node's own agent could not be read.

        Deliberately its own half of the world, NOT merged into
        ``snapshot()``'s ``tenants``/``gpus``: a remote engine's
        ``gpu_index`` addresses ITS OWN box's GPU list, so folding remote
        tenants into the local resource-keyed map would let
        ``app.arbiter``'s co-residency and eviction arithmetic (which
        matches ``tenant["gpu_index"]`` against the LOCAL gpu list) compare
        two different machines' GPU 0 — and a hand-edited registry can hold
        two nodes' same-named resources (refused at the declaration boundary
        and healed at load since Task 9, but not something this assembly may
        lean on), so the map keys could collide outright.

        ``gpu_pools[node] is None`` (or no entry at all) ⇒ that node's
        engines are reported unknown WITHOUT probing them one by one. The
        agent's GPU read is the node's liveness probe — app/node_observer.py
        already binds a node's whole status to that one call — so behind an
        agent that just failed to answer, N per-engine probes would only buy
        N more transport timeouts inside a 2 s arbiter tick.

        A pair with no operable client is unknown for the same reason: it is
        still DECLARED, so it must appear, and "we failed to look" is not
        "nothing is loaded". Both unknown records come from the KIND's own
        ``unknown()`` (app.engine_kinds) — never synthesized here, which
        would mean this module guessing a per-kind shape only that one knows.
        """
        now = self._clock()

        declared = {node_key(e["node_id"], e["resource"]): e for e in engines}
        # Same rule as the local half: a resource that left the declaration
        # loses its idle-clock memory; re-declaring it later starts fresh.
        for gone in set(self._remote_mem) - set(declared):
            del self._remote_mem[gone]

        tenants: dict[str, dict] = {}
        for key, entry in declared.items():
            kind = entry["kind"]
            adapter = ENGINE_KINDS[kind]
            node_id = entry["node_id"]
            resource = entry["resource"]
            mem = self._remote_mem.setdefault(key, {})
            # Same in-place kind-change reset the local half does — see
            # snapshot()'s own comment for why stale bookkeeping under a new
            # kind reads as a real, wrong idle baseline.
            if mem.get(_KIND_MEM_KEY, kind) != kind:
                mem.clear()
            mem[_KIND_MEM_KEY] = kind

            client = (clients.client_for(node_id, resource)
                      if gpu_pools.get(node_id) is not None else None)
            if client is None:
                # Drop this tenant's idle-clock bookkeeping (fix round 1,
                # review finding 1). The adapter's own observe() — where
                # every kind's "a non-idle answer re-arms the clock" rule
                # lives — does NOT run on this path, and this path is
                # exactly the one a powered-off box takes, for as long as it
                # stays off (app.node_clients.RemoteObserver's backoff
                # stretches it to minutes). Left standing, the bookkeeping
                # from the last time we COULD see this engine survives the
                # whole dark window, so the FIRST observation after the box
                # returns reads that window as observed idle time and the
                # idle rule fires against an engine that may have finished
                # booting seconds ago (~4 min for sglang-omni, GF4).
                #
                # Cleared rather than re-stamped, and kind-agnostically:
                # only the adapter knows what its own mem holds, and every
                # one of them treats an EMPTY mem as "first-ever
                # observation" and establishes a fresh baseline from it
                # (app.engine_kinds — comfyui/sglang-omni's
                # `last_activity_time is None` arm, lemonade's `loaded !=
                # mem.get("last_loaded")` transition check). Per TENANT, not
                # a blanket wipe: an engine on a node that is still
                # answering keeps its clock.
                #
                # `_KIND_MEM_KEY` is re-stamped because it is this module's
                # own marker, not the adapter's — dropping it would make the
                # next snapshot see a kind CHANGE that never happened.
                mem.clear()
                mem[_KIND_MEM_KEY] = kind
                obs = adapter.unknown()
            else:
                # `routes: None` — the litellm route table is the LOCAL
                # box's; a remote engine is not in it, and asking would
                # answer about the wrong machine. Kinds that read it degrade
                # that one field to None exactly as they do when litellm
                # itself is unreachable.
                obs = adapter.observe(
                    client, mem, now,
                    {"registry": registry, "routes": None, "resource": resource})
            obs["engine"] = kind
            obs["gpu_index"] = entry["gpu_index"]
            # The two fields every downstream key is built from
            # (app.observe.observe_remote). Carried ON the record rather
            # than parsed back out of the map key, so the two cannot drift.
            obs["node_id"] = node_id
            obs["resource"] = resource
            tenants[key] = obs
        return tenants

    def note_freed(self, resource: str, node_id: str | None = None) -> None:
        """A successful deck actuation re-arms that resource's idle TTL.
        Without this, idle_s only grows once the resource is idle (acting on
        it changes none of the idle-release rule's inputs), so the watcher
        re-emits its action on every tick — flooding the event ring and the
        engine's endpoint. A no-op (mem entry harmlessly created, then pruned
        next snapshot) if `resource` isn't currently declared.

        `node_id` (sglang-omni Task 9) selects WHICH half's clock: None — a
        successful VRAM free, the comfyui-kind original — re-arms the local
        `_mem`; a node id re-arms `_remote_mem` under that half's own
        `<node>/<resource>` key. The remote case is the one this matters most
        for: a remote engine's observation is TTL-cached and its container
        takes seconds to stop, so the very same idle-past-ttl record is what
        the next several ticks would otherwise read and re-act on.
        """
        mem = self._mem if node_id is None else self._remote_mem
        key = resource if node_id is None else node_key(node_id, resource)
        mem.setdefault(key, {})["last_activity_time"] = self._clock()

    def _externals(self, gpus, engines, tenants: dict[str, dict]) -> list[dict]:
        any_tenant_active = any(
            ENGINE_KINDS[entry["kind"]].active(tenants[entry["resource"]])
            for entry in engines
        )
        if any_tenant_active:
            return []

        externals = []
        for gpu in gpus:
            for pid, nbytes in gpu["pids"].items():
                if nbytes > _EXTERNAL_FLOOR_BYTES:
                    externals.append({"pid": pid, "gpu": gpu["index"], "bytes": nbytes})
        return externals
