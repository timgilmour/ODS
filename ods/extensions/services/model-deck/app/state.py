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

No Settings import here — pure inputs only.
"""

import time

from app.engine_kinds import ENGINE_KINDS
from app.engines import EngineError

# Externals heuristic floor: below this, transient/small allocations are
# noise, not worth surfacing as a tenant-shaped list entry.
_EXTERNAL_FLOOR_BYTES = 1 * 1024**3  # 1 GiB

_OPENAI_PREFIX = "openai/"


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
            adapter = ENGINE_KINDS[entry["kind"]]
            client = clients.client_for(resource)
            mem = self._mem.setdefault(resource, {})
            # A fresh per-resource ctx (not one shared dict mutated in
            # place): see app.engine_kinds' module docstring for why
            # `resource` rides in ctx rather than a fifth `observe` param.
            ctx = {"registry": registry, "routes": routes, "resource": resource}
            obs = adapter.observe(client, mem, now, ctx)
            obs["engine"] = entry["kind"]
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

    def note_freed(self, resource: str) -> None:
        """A successful VRAM free (any kind whose arbiter verb is "free" —
        comfyui-kind today) re-arms that resource's idle TTL. Without this,
        idle_s only grows once the resource is idle (freeing changes none of
        the idle-release rule's inputs), so the watcher re-emits its free
        action on every tick — flooding the event ring and the engine's
        free endpoint. A no-op (mem entry harmlessly created, then pruned
        next snapshot) if `resource` isn't currently declared."""
        self._mem.setdefault(resource, {})["last_activity_time"] = self._clock()

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
