"""Tests for app.arbiter — the priority arbiter (pure ``decide``) + watcher.

``decide(world, policy, pending_load) -> list[dict]`` is a pure function: no
I/O, deterministic, and every guard below is tested for the ABSENCE of the
dangerous action, not merely the presence of the safe one — this module
decides real GPU unloads on a live box.

Action dicts (E1: every action carries the RESOURCE it targets):
    {"type": "unload", "resource": <str>, "model": <str>}
    {"type": "free", "resource": <str>}
    {"type": "noop", "reason": <str>}

The Watcher tests drive one ``tick()`` at a time against stub deps (no real
thread needed for behavior coverage); a single thread test proves
start()/stop() runs at least one tick and joins promptly.
"""

import json
import threading
import time

import pytest

from app.arbiter import HealSuppressor, Watcher, decide
from app.characteristics import CharacteristicsStore
from app.engines import EngineError, GuardError
from app.engines.node_agent import NodeAgentUnreachable
from app.engines.spark import boot_in_flight
from app.events import tail_events
# app.harvest's real sentinel, for the C2 remote-pair end-to-end tests below
# (task 8): a hand-seeded option_catalog shape would bypass
# parse_probe_output entirely and could never catch a real parsing bug --
# exactly the masking bug C1's final review caught (see this section's C2
# subsection comment).
from app.harvest import _SENTINEL
# The node id the settings API reads catalogs under (app/observe.py:28) —
# harvest-scoped tests inject engine names via `configurable_engines` and
# must key their assertions off THIS constant, never a hand-typed "local"
# literal or `settings.node_label`, so a future vocabulary regression here
# would show up as an import/assertion mismatch, not a silently-passing
# coincidence (see Watcher._configurable_engines's docstring, app/arbiter.py).
from app.observe import _LOCAL_NODE, SparkObserver, local_key

# The reconcile pass is the first watcher code to call the hipfire client, so
# it needs a real fake rather than the bare object() the arbitration tests
# pass. Reuse test_api's rather than growing a second, subtly different one.
# FakeNodeClients (task 8): the node_clients= constructor param's fake -- a
# plain dict lookup, so a test wires exactly the client(s) it means to.
from tests.test_api import FakeHipfire, FakeNodeClients

# Catalog-harvest tests (task 8) reuse test_harvest's captured probe output
# rather than growing a second, subtly different fixture.
from tests.test_harvest import PROBE_OUTPUT

GIB = 1024**3


# --- world / policy builders -------------------------------------------------


def _lem(state="unloaded", model=None, footprint=None, idle_s=None):
    return {"state": state, "model": model, "footprint": footprint, "idle_s": idle_s}


def _comfy(state="idle", queue=0, idle_s=0):
    return {"state": state, "queue": queue, "idle_s": idle_s}


def _hip(state="parked", model=None, footprint=0):
    return {"state": state, "model": model, "footprint": footprint}


def _gpu(index=1, total=34 * GIB, used=0, free=None):
    if free is None:
        free = total - used
    return {"index": index, "total": total, "used": used, "free": free}


def _world(gpus=None, lemonade=None, comfyui=None, hipfire=None, default_route=None, placement=None):
    placement = (
        placement if placement is not None else {"hipfire": 0, "lemonade": 1, "comfyui": 1}
    )
    tenants = {
        "lemonade": lemonade if lemonade is not None else _lem(),
        "comfyui": comfyui if comfyui is not None else _comfy(),
        "hipfire": hipfire if hipfire is not None else _hip(),
    }
    # E1 backfill (obligation 1, T5 review): Task 3's real World.snapshot
    # stamps `engine`/`gpu_index` onto every tenant obs (app/state.py:
    # `obs["engine"] = kind; obs["gpu_index"] = entry["gpu_index"]`) —
    # decide() now reads BOTH unconditionally (app.engine_kinds dispatch,
    # per-GPU eviction scoping). Every hand-built world dict in this file
    # routes through this one function, so backfilling here — using the
    # tenant KEY as the kind name and this same `placement` mapping as the
    # gpu_index source — covers every one of them at a single point.
    # Resource == kind name for this fixed triple (the coexistence-era
    # shape every test here still builds), so the backfill is exact, not a
    # guess. `setdefault` leaves any fixture that already sets these
    # explicitly untouched.
    for resource, tenant in tenants.items():
        tenant.setdefault("engine", resource)
        tenant.setdefault("gpu_index", placement.get(resource))
    return {
        "gpus": gpus if gpus is not None else [_gpu()],
        "tenants": tenants,
        "externals": [],
        "default_route": default_route,
        "routes_known": True,
        "placement": placement,
    }


def _policy(lem_pinned=False, lem_idle=900, comfy_pinned=False, comfy_idle=300):
    return {
        "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
        "lemonade": {"priority": 50, "pinned": lem_pinned, "idle_ttl": lem_idle},
        "comfyui": {"priority": 40, "pinned": comfy_pinned, "idle_ttl": comfy_idle},
    }


_VALID_ACTION_TYPES = {"unload", "free", "noop"}


def _assert_only_valid_actions(actions, world):
    """No action type may ever target hipfire or an external — the only way
    to prove that guard is to assert by omission across every result.

    T5 review fix: once every kind shares the same generic action vocabulary
    ("unload"/"free"/"noop"), a type-only check no longer proves the TARGET
    wasn't hipfire — only that no exotic type slipped through. Derive the
    answer honestly from `world`'s own tenants (their `engine` field) rather
    than a fixture-naming assumption (a resource need not be named
    "hipfire" to BE hipfire-kind), so this helper keeps meaning what its
    docstring promises regardless of which fixture style calls it."""
    tenants = world["tenants"]
    for a in actions:
        assert a["type"] in _VALID_ACTION_TYPES, a
        resource = a.get("resource")
        if resource is not None:
            assert tenants[resource]["engine"] != "hipfire", a


def _types(actions):
    return [a["type"] for a in actions]


# ===========================================================================
# RULE 1 — IDLE RELEASE (pending_load is None)
# ===========================================================================


def test_idle_release_unloads_default_route_lemonade_when_idle_past_ttl():
    """Idle release IS allowed on the default-route model — that's the
    idle-GPU-burn fix (reload is ~4 s). This is the deliberate exception to
    the 'never touch the default-route model' rule and applies ONLY to idle
    release, never to contention (rule 2)."""
    model = "extra.Qwen3.5-27B-Q4_K_M.gguf"
    world = _world(
        lemonade=_lem(state="loaded", model=model, footprint=19 * GIB, idle_s=901),
        default_route=model,  # loaded model IS the default route
    )

    result = decide(world, _policy(lem_idle=900), None)

    assert result == [{"type": "unload", "resource": "lemonade", "model": model}]
    _assert_only_valid_actions(result, world)


def test_idle_release_no_unload_when_lemonade_pinned():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=5000),
    )

    result = decide(world, _policy(lem_pinned=True), None)

    assert "unload" not in _types(result)


def test_idle_release_no_unload_when_idle_ttl_zero():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=99999),
    )

    result = decide(world, _policy(lem_idle=0), None)

    assert "unload" not in _types(result)


def test_idle_release_no_unload_when_idle_s_none():
    """idle_s None means activity metrics are unavailable, which is NOT the
    same as idle — never release on missing telemetry."""
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=None),
    )

    result = decide(world, _policy(lem_idle=900), None)

    assert "unload" not in _types(result)


def test_idle_release_no_unload_when_below_ttl():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=100),
    )

    result = decide(world, _policy(lem_idle=900), None)

    assert result == []


def test_idle_release_frees_comfyui_when_idle_past_ttl():
    # comfy's GPU shows real usage (nothing else accounts for it), so the
    # free actually reclaims VRAM.
    world = _world(
        gpus=[_gpu(index=1, used=6 * GIB)],
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert result == [{"type": "free", "resource": "comfyui"}]
    _assert_only_valid_actions(result, world)


def test_idle_release_no_free_when_comfy_holds_no_vram():
    """TTL expiry alone must not free: with comfy's GPU essentially empty
    (used below the slack allowance), the free is a guaranteed no-op that
    would just re-arm the TTL and spam the event ring forever."""
    world = _world(
        gpus=[_gpu(index=1, used=60 * 1024**2)],
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free" not in _types(result)


def test_idle_release_no_free_when_gpu_usage_is_lemonades():
    """Usage fully accounted for by a loaded lemonade on the same GPU is not
    comfy's to reclaim — freeing comfy would be a no-op."""
    world = _world(
        gpus=[_gpu(index=1, used=10 * GIB)],
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free" not in _types(result)


def test_idle_release_frees_comfyui_when_gpu_unresolvable():
    """Placement pointing at a GPU we have no telemetry for must NOT suppress
    the free — unknown usage is not proof there's nothing to reclaim."""
    world = _world(
        gpus=[_gpu(index=0, used=0)],
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
        placement={"hipfire": 0, "lemonade": 1, "comfyui": 1},
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert result == [{"type": "free", "resource": "comfyui"}]


def test_idle_release_no_free_when_comfy_pinned():
    world = _world(comfyui=_comfy(state="idle", queue=0, idle_s=5000))

    result = decide(world, _policy(comfy_pinned=True), None)

    assert "free" not in _types(result)


def test_idle_release_no_free_when_comfy_busy():
    """A busy comfy (queue>0) is never freed even long 'past' a TTL."""
    world = _world(comfyui=_comfy(state="busy", queue=3, idle_s=5000))

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free" not in _types(result)


def test_idle_release_no_free_when_comfy_idle_ttl_zero():
    world = _world(comfyui=_comfy(state="idle", queue=0, idle_s=99999))

    result = decide(world, _policy(comfy_idle=0), None)

    assert "free" not in _types(result)


def test_idle_release_no_free_when_comfy_unknown():
    """state 'unknown' (queue/idle_s None) must never trigger a free."""
    world = _world(comfyui={"state": "unknown", "queue": None, "idle_s": None})

    result = decide(world, _policy(comfy_idle=300), None)

    assert result == []


def test_idle_release_empty_when_nothing_idle():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
    )

    result = decide(world, _policy(), None)

    assert result == []


# ===========================================================================
# RULE 2 — CONTENTION HEALING (pending_load set)
# ===========================================================================


def _pending(model="extra.model.gguf", footprint=19 * GIB, gpu_index=1, resource="lemonade"):
    # "resource" mirrors Watcher._infer_pending's real output shape (E1
    # Task 6 added it there); decide() itself never reads it (pending_load
    # only ever needs footprint/gpu_index), so every decide()-only caller
    # of this helper is unaffected — only test_same_tick_evict_and_reload_
    # pins_single_load_and_final_intent, which feeds this straight into
    # watcher._execute(), actually needs it.
    return {"model": model, "footprint": footprint, "gpu_index": gpu_index, "resource": resource}


def test_contention_todays_incident_frees_comfyui():
    """The exact incident from the brief: gpu1 free 12 GiB, comfy idle 400 s
    holding ~21 GiB, pending 27B footprint 19 GiB -> [free_comfyui]."""
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "free", "resource": "comfyui"}]
    _assert_only_valid_actions(result, world)


def test_contention_noop_fits_when_free_already_sufficient():
    world = _world(gpus=[_gpu(index=1, total=34 * GIB, used=14 * GIB)])  # free = 20 GiB

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "fits"}]


def test_contention_wont_fit_when_nothing_evictable():
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="busy", queue=2, idle_s=0),  # busy -> not evictable
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "free" not in _types(result)
    assert "unload" not in _types(result)


def test_contention_never_evicts_default_route_lemonade():
    """ABSOLUTE guard: even though unloading the default-route model here
    WOULD free enough VRAM to fit, it must NOT be evicted for contention.
    Comfy is busy so lemonade is the only thing that could be freed."""
    model = "extra.default.gguf"
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model=model, footprint=25 * GIB, idle_s=10),
        comfyui=_comfy(state="busy", queue=1, idle_s=0),
        default_route=model,  # loaded model IS the default route
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    # Would have fit (5 + 25 >= 19) if the guard weren't absolute.
    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "unload" not in _types(result)


def test_contention_evicts_lemonade_when_not_default_route():
    """Complement of the guard: a loaded, non-default, unpinned lemonade IS
    a valid contention-eviction candidate."""
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=25 * GIB, idle_s=10),
        comfyui=_comfy(state="busy", queue=1, idle_s=0),  # not evictable
        default_route="extra.SOMETHING-ELSE.gguf",
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "unload", "resource": "lemonade", "model": "extra.other.gguf"}]


def test_contention_no_evict_when_lemonade_footprint_unknown():
    """A loaded lemonade with an unknown (None) footprint can't be reasoned
    about, so it is not a usable contention candidate -> wont-fit, no unload."""
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=None, idle_s=10),
        comfyui=_comfy(state="busy", queue=1, idle_s=0),
        default_route="extra.diff.gguf",
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "unload" not in _types(result)


def test_contention_evicts_comfy_before_lemonade_ascending_priority():
    """Comfy (priority 40) is evicted before lemonade (priority 50); when
    comfy alone suffices, lemonade is left untouched (minimal set)."""
    world = _world(
        gpus=[_gpu(index=1, total=60 * GIB, used=55 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=10 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
        default_route="extra.diff.gguf",
    )
    # comfy reclaimable = used(55) - lem(10) - 1 slack = 44 GiB; 5 + 44 >= 19.

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "free", "resource": "comfyui"}]
    assert "unload" not in _types(result)


def test_contention_evicts_both_when_comfy_insufficient():
    """Comfy first, then lemonade, when comfy alone can't cover the deficit."""
    world = _world(
        gpus=[_gpu(index=1, total=36 * GIB, used=31 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=20 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
        default_route="extra.diff.gguf",
    )
    # comfy reclaimable = 31 - 20 - 1 = 10 GiB; comfy alone: 5 + 10 = 15 < 30.
    # + lemonade 20 -> 35 >= 30.

    result = decide(world, _policy(), _pending(footprint=30 * GIB, gpu_index=1))

    assert result == [
        {"type": "free", "resource": "comfyui"},
        {"type": "unload", "resource": "lemonade", "model": "extra.other.gguf"},
    ]


def test_contention_wont_fit_emits_no_partial_eviction():
    """Feasibility-first: if the FULL set of eligible evictions still can't
    fit, emit noop 'wont-fit' and evict NOTHING — never kill a tenant for a
    load that won't fit anyway."""
    world = _world(
        gpus=[_gpu(index=1, total=36 * GIB, used=31 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.default.gguf", footprint=20 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
        default_route="extra.default.gguf",  # lemonade protected
    )
    # Only comfy is eligible: reclaimable = 31 - 20 - 1 = 10 GiB; 5 + 10 = 15 < 30.

    result = decide(world, _policy(), _pending(footprint=30 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "free" not in _types(result)  # comfy NOT killed pointlessly


def test_contention_pinned_comfy_not_evicted():
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=5000),
    )

    result = decide(world, _policy(comfy_pinned=True), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "free" not in _types(result)


def test_contention_busy_comfy_never_freed():
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="busy", queue=5, idle_s=0),
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert "free" not in _types(result)


def test_contention_never_touches_hipfire():
    """hipfire running on its own GPU while a load contends on gpu1: the only
    'reclaimable' VRAM belongs to hipfire, but no action may target it."""
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="busy", queue=1, idle_s=0),  # busy, off-limits
        hipfire=_hip(state="running", model="q", footprint=33 * GIB),
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    # _assert_only_valid_actions now checks BOTH the type vocabulary and (T5
    # review) that no action's resource is a hipfire-kind tenant in `world`
    # — the per-site version of that second check used to live here
    # explicitly; deleting it and relying on the shared helper is what
    # proves the helper actually covers this invariant (guard-at-the-
    # boundary).
    _assert_only_valid_actions(result, world)


def test_contention_gpu_index_not_found_is_wont_fit():
    world = _world(gpus=[_gpu(index=1, total=34 * GIB, used=0)])

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=0))

    assert result == [{"type": "noop", "reason": "wont-fit"}]


def test_contention_honors_policy_priority_low_evicted_first():
    """Policy priority drives eviction order: with comfyui set to a HIGHER
    priority (90) than lemonade (50) — the inverse of the defaults — the
    lower-priority tenant (lemonade) is evicted first. Since lemonade alone
    suffices, comfyui must be left untouched (minimal set, honoring the new
    order)."""
    policy = {
        "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
        "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
        "comfyui": {"priority": 90, "pinned": False, "idle_ttl": 300},
    }
    world = _world(
        gpus=[_gpu(index=1, total=60 * GIB, used=55 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=10 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
        default_route="extra.diff.gguf",
    )
    # lemonade reclaimable = 10 GiB; free(5) + 10 = 15 >= 14 -> lemonade alone
    # suffices. comfy reclaimable would be 55 - 10 - 1 = 44 GiB (also
    # feasible alone), but lemonade's lower priority (50 < 90) sorts it first.

    result = decide(world, policy, _pending(footprint=14 * GIB, gpu_index=1))

    assert result == [{"type": "unload", "resource": "lemonade", "model": "extra.other.gguf"}]


# ===========================================================================
# E1 GENERALIZATION — decide() over a genuinely declaration-shaped world:
# more than one resource per kind, GPUs that don't match live topology, an
# empty declaration. Fixture rule (same convention as test_engine_kinds.py's
# adapter section): resources gguf-a/gguf-b/img/agent, GPUs 2 and 3 — none of
# this may coincidentally match lemonade/comfyui/hipfire on GPUs 0/1, so a
# bug that silently falls back to a kind-name-as-resource-name assumption
# cannot hide behind a fixture that happens to agree with it.
# ===========================================================================


def _dworld(tenants: dict, gpus: list[dict]) -> dict:
    return {
        "gpus": gpus,
        "tenants": tenants,
        "externals": [],
        "default_route": "default.gguf",
        "routes_known": True,
        "placement": {r: t["gpu_index"] for r, t in tenants.items()},
    }


def _dgpu(index, total, used):
    return {"index": index, "total": total, "used": used, "free": total - used}


def test_idle_release_iterates_every_declared_gguf():
    """E1: rule 1 runs independently for EVERY declared lemonade-kind
    resource, not a single hardcoded 'lemonade' tenant."""
    world = _dworld({
        "gguf-a": {"engine": "lemonade", "gpu_index": 2, "state": "loaded",
                   "model": "a.gguf", "footprint": 10, "idle_s": 100.0},
        "gguf-b": {"engine": "lemonade", "gpu_index": 3, "state": "loaded",
                   "model": "b.gguf", "footprint": 10, "idle_s": 5.0},
    }, [_dgpu(2, 100, 50), _dgpu(3, 100, 50)])
    policy = {"gguf-a": {"priority": 1, "pinned": False, "idle_ttl": 60},
              "gguf-b": {"priority": 1, "pinned": False, "idle_ttl": 60}}

    actions = decide(world, policy, None)

    assert actions == [{"type": "unload", "resource": "gguf-a", "model": "a.gguf"}]


def test_contention_is_scoped_to_the_pending_gpu():
    """Two contended GPUs at once (impossible pre-E1): only residents of
    the pending load's GPU are candidates."""
    world = _dworld({
        "gguf-a": {"engine": "lemonade", "gpu_index": 2, "state": "loaded",
                   "model": "a.gguf", "footprint": 40, "idle_s": 0.0},
        "img":    {"engine": "comfyui", "gpu_index": 3, "state": "idle",
                   "queue": 0, "idle_s": 500.0},
    }, [_dgpu(2, 100, 95), _dgpu(3, 100, 95)])
    policy = {"gguf-a": {"priority": 1, "pinned": False, "idle_ttl": 0},
              "img": {"priority": 2, "pinned": False, "idle_ttl": 300}}
    pending = {"resource": "gguf-b", "model": "b.gguf", "footprint": 30,
               "gpu_index": 2}

    actions = decide(world, policy, pending)

    assert actions == [{"type": "unload", "resource": "gguf-a",
                        "model": "a.gguf"}]      # img is on GPU 3: untouchable


def test_agent_kind_is_never_a_candidate():
    """hipfire-kind has no arbiter verbs — even unpinned, even alone."""
    world = _dworld({
        "agent": {"engine": "hipfire", "gpu_index": 2, "state": "running",
                  "model": "x", "footprint": 90, "queue_depth": 0},
    }, [_dgpu(2, 100, 95)])
    policy = {"agent": {"priority": 1, "pinned": False, "idle_ttl": 0}}
    pending = {"resource": "gguf-b", "model": "b.gguf", "footprint": 30,
               "gpu_index": 2}

    assert decide(world, policy, pending) == [{"type": "noop", "reason": "wont-fit"}]


def test_default_route_model_still_never_evicted():
    world = _dworld({
        "gguf-a": {"engine": "lemonade", "gpu_index": 2, "state": "loaded",
                   "model": "default.gguf", "footprint": 40, "idle_s": 0.0},
    }, [_dgpu(2, 100, 95)])
    policy = {"gguf-a": {"priority": 1, "pinned": False, "idle_ttl": 0}}
    pending = {"resource": "gguf-b", "model": "b.gguf", "footprint": 30,
               "gpu_index": 2}

    assert decide(world, policy, pending) == [{"type": "noop", "reason": "wont-fit"}]


def test_decide_empty_declaration_idle_path_is_a_true_noop():
    """Obligation 2 (T5 review): decide() must tolerate a tenants/policy map
    missing any legacy name entirely — an empty declaration is not a crash,
    it's nothing to do."""
    world = _dworld({}, [])

    assert decide(world, {}, None) == []


def test_decide_empty_declaration_pending_load_noops_nothing_evictable():
    """Same tolerance for the contention path: a pending load against a GPU
    with zero declared tenants can't be healed by eviction (nothing to
    evict), so it must resolve to a plain 'wont-fit' noop, never a KeyError
    from indexing a tenants/policy map that has no entries at all."""
    world = _dworld({}, [_dgpu(2, 100, 90)])  # free = 10
    pending = {"resource": "gguf-a", "model": "a.gguf", "footprint": 30,
               "gpu_index": 2}

    assert decide(world, {}, pending) == [{"type": "noop", "reason": "wont-fit"}]


def test_idle_release_comfy_suppressed_by_coresident_hipfire_footprint():
    """T5 review: `_co_resident_footprints` must generalize past "minus
    lemonade's footprint" to ANY co-resident loaded/running tenant — proven
    here with a co-resident HIPFIRE, a kind that never appeared in this role
    pre-E1. A running hipfire-kind resource on the same GPU as an idle
    comfy-kind resource explains almost all of that GPU's usage
    (used 26 GiB, hipfire footprint 25 GiB, 1 GiB slack -> reclaimable ==
    exactly 0), so the free must be SUPPRESSED — not fire as a guaranteed
    no-op that would just re-arm the TTL forever."""
    world = _dworld({
        "img": {"engine": "comfyui", "gpu_index": 2, "state": "idle",
                "queue": 0, "idle_s": 500.0},
        "agent": {"engine": "hipfire", "gpu_index": 2, "state": "running",
                  "model": "x", "footprint": 25 * GIB, "queue_depth": 0},
    }, [_dgpu(2, 34 * GIB, 26 * GIB)])
    policy = {"img": {"priority": 1, "pinned": False, "idle_ttl": 300},
              "agent": {"priority": 1, "pinned": False, "idle_ttl": 0}}

    actions = decide(world, policy, None)

    assert actions == []


# ===========================================================================
# WATCHER — thread + execution shell (stub deps)
# ===========================================================================


class FakeWorld:
    """Returns a preset snapshot dict; counts snapshot() calls. Optionally
    raises on snapshot to exercise the per-tick error swallow."""

    def __init__(self, snapshot, raises=None):
        self._snapshot = snapshot
        self._raises = raises
        self.calls = 0
        self.comfy_freed_notes = 0

    def snapshot(self, gpus, engines, clients, litellm, registry):
        # E1 Task 3: World.snapshot's signature moved from the fixed
        # (lemonade, comfy, hipfire) triple to (engines, clients) — an
        # ARITY-only rename here (this fake never inspected the individual
        # per-kind params, only counted calls / returned the canned
        # snapshot), so no test behavior changes.
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._snapshot

    def note_freed(self, resource):
        # E1 Task 3: note_comfy_freed() -> note_freed(resource); every
        # existing caller in this file only ever freed comfy, so the
        # counter's meaning is unchanged.
        self.comfy_freed_notes += 1


class FakeRegistry:
    def __init__(self, footprints=None, missing=()):
        self._footprints = footprints or {}
        self._missing = set(missing)

    def footprint(self, key):
        if key in self._missing:
            raise FileNotFoundError(key)
        return self._footprints[key]


class FakePolicyStore:
    def __init__(self, policy, auto=True):
        self._policy = policy
        self._auto = auto

    def get(self):
        return self._policy

    def auto_enabled(self):
        return self._auto

    def set_auto(self, enabled):
        self._auto = bool(enabled)

    def forget(self, resource):
        """app.policy.PolicyStore.forget's contract: pop the stored row for
        `resource`, and a resource with no row is a no-op rather than a
        KeyError. Added for the §6.2 remote coexistence proof (sglang-omni
        Task 11), which drives the REAL forget route — and that route's third
        write is this call."""
        self._policy.pop(resource, None)


class FakeLemonade:
    def __init__(self, raise_on_load=None, raise_on_unload=None, in_flight=False):
        self.unloaded = []
        self.loaded = []
        self._raise_on_load = raise_on_load
        self._raise_on_unload = raise_on_unload
        self._in_flight = in_flight

    def load_in_flight(self):
        return self._in_flight

    def unload(self, model):
        # Records the ATTEMPT before its guard, so a test can tell "the
        # unload was never dispatched" apart from "it was dispatched and
        # raised" (the FakeSpark lesson from task 1).
        self.unloaded.append(model)
        if self._raise_on_unload is not None:
            raise self._raise_on_unload

    def load(self, model):
        if self._raise_on_load is not None:
            raise self._raise_on_load
        self.loaded.append(model)


class FakeComfy:
    def __init__(self, raise_guard=False, raise_on_free=None):
        self.freed = 0
        self._raise_guard = raise_guard
        self._raise_on_free = raise_on_free

    def free(self):
        if self._raise_guard:
            raise GuardError("queue raced non-empty")
        if self._raise_on_free is not None:
            raise self._raise_on_free
        self.freed += 1


class RecordingReadGpus:
    def __init__(self, result=None):
        self.result = result if result is not None else []
        self.calls = []

    def __call__(self, drm_root, kfd_root):
        self.calls.append((drm_root, kfd_root))
        return self.result


def _settings(**overrides):
    from app.settings import Settings

    kwargs = dict(watch_interval=0.01, lemonade_gpu_index=1)
    kwargs.update(overrides)
    return Settings(**kwargs)


def _make_watcher(
    tmp_path, world, registry, policy, lemonade=None, comfy=None, read_gpus=None,
    heal_suppressor=None, hostagent=None, catalog=None, hipfire=None,
    intent_store=None, hold_store=None, node_clients=None, node_observers=None, auto=True,
    characteristics_store=None,
    gguf_dir=None, clock=None, on_derive=None, engine_exec=None,
    configurable_engines=None, provenance_store=None, dockerctl=None,
    node_store=None, local_clients=None, litellm=None,
    remote_engine_clients=None, node_agent_client_factory=None,
    remote_observer=None, **sett,
):
    # T13 review fix (Critical 1): `litellm` used to be hardcoded to a bare
    # `object()` below with no override — silently correct for every
    # FakeWorld-driven test in this file (which never calls litellm at all),
    # but two_gguf_watcher drives a REAL World.snapshot, which calls
    # `litellm.route_table()` unconditionally (app/state.py). A bare
    # `object()` there raises AttributeError, which Watcher.tick()'s broad
    # supervisor catch (app/arbiter.py) swallows into a `tick-error` event —
    # the tick never reaches arbitration/reconciliation, so an assertion
    # like "nothing was actuated" passes VACUOUSLY, proof-of-absence
    # satisfied by a dead tick. `litellm=None` (every caller before this
    # fix) preserves the exact old default; two_gguf_watcher now passes its
    # own working `_FakeLiteLLMRoutes` fake.
    events_path = tmp_path / "events.jsonl"
    watcher = Watcher(
        settings=_settings(**sett),
        world=world,
        litellm=litellm if litellm is not None else object(),
        registry=registry,
        # Ruling #4b: Watcher no longer builds a legacy fallback client map
        # itself (_LegacyClients deleted) — callers that don't wire their own
        # declaration-shaped local_clients= get an equivalent
        # LocalClients-shaped fake built HERE, from the same
        # lemonade=/comfy=/hipfire= keyword defaults the deleted class
        # received verbatim, mapped onto the same legacy resource names.
        local_clients=(
            local_clients if local_clients is not None
            else _FakeLocalClients({
                "lemonade": lemonade if lemonade is not None else FakeLemonade(),
                "comfyui": comfy if comfy is not None else FakeComfy(),
                "hipfire": hipfire if hipfire is not None else object(),
            })
        ),
        policy_store=FakePolicyStore(policy, auto=auto),
        events_path=events_path,
        read_gpus=read_gpus if read_gpus is not None else RecordingReadGpus(),
        heal_suppressor=heal_suppressor,
        hostagent=hostagent,
        catalog=catalog,
        intent_store=intent_store,
        hold_store=hold_store,
        node_clients=node_clients,
        node_observers=node_observers,
        characteristics_store=characteristics_store,
        gguf_dir=gguf_dir,
        clock=clock if clock is not None else time.monotonic,
        on_derive=on_derive,
        engine_exec=engine_exec,
        configurable_engines=configurable_engines,
        provenance_store=provenance_store,
        dockerctl=dockerctl,
        # E1 Task 6: declaration-shaped watcher tests (two_gguf_watcher
        # below) wire a real node_store + a LocalClients-shaped fake here;
        # every other caller in this file leaves both None, which routes
        # actuation through the _FakeLocalClients fallback built above
        # instead (ruling #4b: Watcher itself no longer builds one).
        node_store=node_store,
        # sglang-omni Task 6: the remote half of the tick's world. All
        # three None (every caller before that task) means "no remote
        # engines" — the remote half is skipped entirely, exactly as it is
        # for a deck with no node-agent entry declaring anything.
        # `remote_observer` (Task 7) is what paces those probes; production
        # always supplies one (app.main), so a test driving the remote half
        # supplies one too or the half is skipped.
        remote_engine_clients=remote_engine_clients,
        node_agent_client_factory=node_agent_client_factory,
        remote_observer=remote_observer,
    )
    return watcher, events_path


def _watcher(tmp_path, world=None, registry=None, policy=None, **kwargs):
    """Thin wrapper around ``_make_watcher`` for tests (the characteristics
    derive-pass tests below) that don't care about arbitration/reconciliation
    setup and just want a watcher with sane defaults, returning the watcher
    alone rather than the ``(watcher, events_path)`` pair."""
    watcher, _events_path = _make_watcher(
        tmp_path,
        world if world is not None else FakeWorld(_world()),
        registry if registry is not None else FakeRegistry(),
        policy if policy is not None else _policy(),
        **kwargs,
    )
    return watcher


# ===========================================================================
# two_gguf_watcher — E1 Task 6: a watcher over TWO independent lemonade-kind
# resources (declaration shape), extending _make_watcher/_watcher above with
# the node_store=/local_clients= wiring those gained for a resource whose
# name isn't one of the legacy lemonade/comfyui/hipfire triple. T13 extends
# this SAME fixture further (load_natively/forget helpers — see the plan's
# pre-flight conflict scan); don't fork it there either.
# ===========================================================================


class _FakeLocalClients:
    """LocalClients-shaped fake (``client_for`` only — the same shape as
    tests/test_state.py's own ``_FakeLocalClients``), plus ``.fake(resource)``:
    a test-only accessor this file's declaration-shaped tests use to reach a
    specific resource's fake client directly (mirrors
    ``deck["local_clients"].fake`` in tests/test_api.py, per the plan's
    pre-flight conflict scan)."""

    def __init__(self, clients: dict) -> None:
        self._clients = clients

    def client_for(self, resource: str):
        return self._clients.get(resource)

    def retire_absent(self, keep_resources) -> None:
        pass  # nothing built lazily here to retire

    def fake(self, resource: str):
        return self._clients[resource]


class _FakeLiteLLMRoutes:
    """Minimal ``route_table()`` fake for two_gguf_watcher's REAL
    ``World.snapshot`` calls: ``default`` starts unset (``route_table()``
    then carries no "default" key at all, matching a box with no default
    route configured — same shape ``world["default_route"]`` reads as
    falsy for), and is driven by ``_PendingCapableLemonade.set_pending``
    below."""

    def __init__(self) -> None:
        self.default = None

    def route_table(self) -> dict:
        return {} if self.default is None else {"default": self.default}


class _PendingCapableLemonade:
    """A lemonade-kind client wired for BOTH observation (status/activity/
    load_in_flight — the contract app.engine_kinds._LemonadeAdapter.observe
    calls, same shape as tests/test_state.py's StubLemonade) and actuation
    (load/unload — the same call-log contract as FakeLemonade above): the
    two_gguf_watcher fixture's tests exercise a REAL World.snapshot (unlike
    every other watcher test in this file, which hands the watcher a canned
    FakeWorld dict), so its resources need both halves at once.

    ``set_pending`` is test-only sugar for
    test_pending_from_second_gguf_carries_its_own_gpu:
    Watcher._infer_pending's actual MECHANISM is UNCHANGED by E1 Task 6 —
    still ``world["default_route"]`` + ``self._registry.footprint()``, both
    box-wide, not per-resource (see that method's own docstring); only the
    per-resource iteration generalizes. This feeds those SAME
    fixture-level fake litellm/registry — a resource-scoped handle is test
    ergonomics, and it also guarantees the call is scoped to a resource
    that observes 'unloaded' (this class's own ``loaded_model``), the
    OTHER precondition _infer_pending checks.

    ``calls``/``load_natively`` (T13, spec §6 coexistence proofs): ``calls``
    is a unified DECK-ACTUATION log — every ``load()``/``unload()`` this
    class's own caller (the watcher, control routes, set apply) drives
    through, alongside the pre-existing kind-specific ``loaded``/
    ``unloaded`` lists those calls already append to (kept so every T6
    assertion against them stays exact). ``load_natively`` sets
    ``loaded_model`` DIRECTLY, bypassing ``load()`` entirely — it never
    touches ``calls``/``loaded`` — because it simulates ODS itself loading a
    model outside the deck's own actuation path (the coexistence posture
    spec §6 pins: the deck must observe this and do nothing, never treat it
    as if the deck had done it).
    """

    def __init__(self, litellm: "_FakeLiteLLMRoutes", registry: "FakeRegistry",
                 loaded: str | None = None, raise_on_unload=None, raise_on_load=None) -> None:
        self._litellm = litellm
        self._registry = registry
        self.loaded_model = loaded
        self.unloaded = []
        self.loaded = []
        self.calls = []  # T13: unified log, see class docstring
        # Mirror FakeLemonade's own raise-on-* knobs (review fix, T6 round
        # 2: the dedup-collision regression test needs two DIFFERENT
        # resources to fail identically, which the earlier version of this
        # fixture had no way to drive).
        self._raise_on_unload = raise_on_unload
        self._raise_on_load = raise_on_load

    def status(self) -> dict:
        return {"loaded": self.loaded_model}

    def activity(self):
        return None

    def load_in_flight(self) -> bool:
        return False

    def unload(self, model: str) -> None:
        self.unloaded.append(model)
        self.calls.append(("unload", model))
        if self._raise_on_unload is not None:
            raise self._raise_on_unload
        self.loaded_model = None

    def load(self, model: str) -> None:
        if self._raise_on_load is not None:
            raise self._raise_on_load
        self.loaded.append(model)
        self.calls.append(("load", model))
        self.loaded_model = model

    def load_natively(self, model: str) -> None:
        """T13: an ODS-native load happening OUTSIDE the deck (e.g. a direct
        llama-server request) — sets residency directly, never through
        ``load()``, so it leaves no actuation trace (``calls``/``loaded``
        stay empty) for the coexistence tests to (correctly) find nothing
        in."""
        self.loaded_model = model

    def set_pending(self, model: str, *, footprint: int) -> None:
        self._litellm.default = model
        self._registry._footprints[model.removeprefix("extra.")] = footprint


# Declaration for two_gguf_watcher: resource names deliberately NOT
# "lemonade" ([[defaults-that-hide-bugs]] — a resource==kind-name
# coincidence would prove nothing about the per-resource generalization),
# GPUs 2/3 (away from the legacy triple's 0/1 placement).
_TWO_GGUF_ENGINES = [
    {"resource": "gguf-a", "kind": "lemonade",
     "connection": {"url": "http://gguf-a:8080",
                    "metrics_url": "http://gguf-a:8081/metrics",
                    "container": "gguf-a"},
     "gpu_index": 2,
     "container_consent": True,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
    {"resource": "gguf-b", "kind": "lemonade",
     "connection": {"url": "http://gguf-b:8080",
                    "metrics_url": "http://gguf-b:8081/metrics",
                    "container": "gguf-b"},
     "gpu_index": 3,
     "container_consent": True,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
]


class _TwoGgufWatcher:
    """two_gguf_watcher's return value: the watcher plus everything needed
    to build a REAL World snapshot against its two declared resources."""

    def __init__(self, watcher, clients, intent_store, gpus, litellm, registry, world,
                 events_path, node_store) -> None:
        self.watcher = watcher
        self.clients = clients
        self.intent_store = intent_store
        self.events_path = events_path
        self._gpus = gpus
        self._litellm = litellm
        self._registry = registry
        self._world = world
        self._node_store = node_store  # T13: forget()'s own declaration write

    def world_snapshot(self) -> dict:
        return self._world.snapshot(
            self._gpus, _TWO_GGUF_ENGINES, self.clients, self._litellm, self._registry)

    def forget(self, resource: str) -> None:
        """T13: store-level mirror of app.routers.nodes.forget_engine
        (app/routers/nodes.py:274-319) — drop the declaration entry AND the
        intent record (that route's own "declaration first, then intent"
        ordering, same file, for the same crash-safety reason). No
        policy_store is wired into this fixture (two_gguf_watcher's watcher
        gets a bare FakePolicyStore the fixture never returns), so there is
        no third write to mirror here — bookkeeping-only either way: this
        never calls the engine, same as the real route."""
        engines = self._node_store.get("local")["engines"]
        self._node_store.update(
            "local", {"engines": [e for e in engines if e["resource"] != resource]})
        self.intent_store.forget(local_key(resource))


@pytest.fixture
def two_gguf_watcher(tmp_path):
    """A watcher over two INDEPENDENT lemonade-kind resources (declaration
    shape) — gguf-a on GPU 2 (loaded with "a.gguf"), gguf-b on GPU 3
    (unloaded) — built through _make_watcher's node_store=/local_clients=
    params (this task's own new additions) rather than a second, forked
    watcher builder."""
    from app.intent import IntentStore
    from app.node_store import NodeStore
    from app.state import World

    node_store = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    node_store.add({"id": "local", "label": "local", "agent_kind": "local"})
    node_store.update("local", {"engines": _TWO_GGUF_ENGINES})

    litellm = _FakeLiteLLMRoutes()
    # gguf-a's own footprint is needed even though this fixture's tests
    # never make gguf-a pending: World.snapshot's lemonade-kind observe()
    # always resolves a LOADED resource's footprint from the registry.
    # "outside.gguf" (T13 review fix, Critical 1): the ODS-native-load
    # coexistence test loads THIS name onto gguf-a via load_natively(), and
    # a loaded resource's observe() unconditionally looks its footprint up
    # (app/engine_kinds.py's _LemonadeAdapter.observe) — FakeRegistry.footprint
    # raises a bare KeyError for any key not seeded here (only `missing=`
    # entries raise the FileNotFoundError observe() actually catches), which
    # used to escape uncaught into Watcher.tick()'s broad supervisor catch
    # and get swallowed as a `tick-error` — the second, independent vacuity
    # the review found.
    registry = FakeRegistry(footprints={"a.gguf": 5 * GIB, "outside.gguf": 3 * GIB})
    # World.snapshot's OWN gpu shape (app.gpu.read_gpus' raw output:
    # vram_total/vram_used/pids) — NOT this file's own `_gpu()` helper,
    # which builds the ALREADY-PROCESSED total/used/free shape decide()
    # consumes; two_gguf_watcher drives a REAL World.snapshot, so its
    # inputs must match what that function actually expects (see
    # tests/test_state.py's own `_gpu()` for the same raw shape). Small
    # raw byte counts, not GiB-scaled — sized to be comparable against
    # test_pending_from_second_gguf_carries_its_own_gpu's own small
    # footprint=30: GPU 2 has plenty of free space (gguf-a is loaded
    # anyway, never a candidate), GPU 3's free (10) is BELOW that
    # footprint, which is what makes gguf-b pending.
    gpus = [
        {"index": 2, "vram_total": 100, "vram_used": 10, "pids": {}},
        {"index": 3, "vram_total": 100, "vram_used": 90, "pids": {}},
    ]
    clients = _FakeLocalClients({
        "gguf-a": _PendingCapableLemonade(litellm, registry, loaded="a.gguf"),
        "gguf-b": _PendingCapableLemonade(litellm, registry, loaded=None),
    })
    intent_store = IntentStore(tmp_path / "intent.json")
    world = World()

    watcher, events_path = _make_watcher(
        tmp_path, world, registry, _policy(),
        node_store=node_store, local_clients=clients, intent_store=intent_store,
        # T13 review fix (Critical 1): this fixture drives a REAL
        # World.snapshot (unlike every FakeWorld-driven test in this file),
        # which calls litellm.route_table() unconditionally — _make_watcher's
        # default bare object() has no such method. Pass the fixture's own
        # working fake so a real tick can actually reach arbitration.
        litellm=litellm,
    )

    return _TwoGgufWatcher(watcher, clients, intent_store, gpus, litellm, registry, world,
                            events_path, node_store)


# ===========================================================================
# T13 — the two binding coexistence tests (spec §6). Both are proofs of
# ABSENCE (no client call, no intent write), driven through a REAL
# watcher.tick() rather than the pure decide()/_execute() seams the rest of
# this section drives directly — coexistence is a property of the WHOLE
# tick (arbitration finding nothing to do AND reconciliation refusing to
# adopt an unmanaged observation into a restore), not of either pass alone.
# ===========================================================================


def test_natively_loaded_model_is_left_alone(two_gguf_watcher):
    """Spec §6.1: observation present, NO intent -> reconciler no-ops.
    ODS-native loading must keep working with the deck watching."""
    w = two_gguf_watcher
    w.clients.fake("gguf-a").load_natively("outside.gguf")
    assert w.intent_store.get() == {}
    w.watcher.tick()
    assert w.clients.fake("gguf-a").calls == []       # nothing actuated
    assert w.intent_store.get() == {}                  # nothing recorded
    # T13 review fix (Critical 1): a swallowed tick-error (litellm.route_table()
    # AttributeError, or a FakeRegistry KeyError on an unseeded footprint) would
    # make BOTH assertions above pass VACUOUSLY — the tick dies before it ever
    # reaches arbitration/reconciliation, "nothing actuated" because nothing
    # ran at all. This guards against that proof-of-absence-by-dead-tick class.
    assert not any(e["kind"] == "tick-error" for e in tail_events(w.events_path))


def test_forget_then_tick_never_calls_the_engine(two_gguf_watcher):
    w = two_gguf_watcher
    w.forget("gguf-a")            # helper: DELETE route or store-level
    w.watcher.tick()
    assert w.clients.fake("gguf-a").calls == []
    # T13 review fix (Critical 1): same vacuity guard as the sibling test
    # above — see its comment.
    assert not any(e["kind"] == "tick-error" for e in tail_events(w.events_path))


def test_pending_from_second_gguf_carries_its_own_gpu(two_gguf_watcher):
    """A second lemonade-kind resource is an independent demand source
    with its own declared GPU (kills settings.lemonade_gpu_index)."""
    w = two_gguf_watcher            # fixture: gguf-a on GPU 2, gguf-b on 3
    w.clients.fake("gguf-b").set_pending("b.gguf", footprint=30)
    pending = w.watcher._infer_pending(w.world_snapshot())
    assert pending["resource"] == "gguf-b" and pending["gpu_index"] == 3


def test_deck_unload_records_intent_for_the_right_resource(two_gguf_watcher):
    w = two_gguf_watcher
    w.watcher._execute([{"type": "unload", "resource": "gguf-b",
                         "model": "b.gguf"}], None)
    rec = w.intent_store.get()["local/gguf-b"]
    assert rec["state"] == "unloaded" and rec["actor"] == "deck"
    assert "local/gguf-a" not in w.intent_store.get()


def test_execute_load_and_restore_target_the_right_non_legacy_resource(two_gguf_watcher):
    """execute_load's key generalization (`local_key(resource)`, not the
    dead `LOCAL_LEMONADE_KEY`) proven away from resource==kind name — the
    unload sibling above only pinned this for the unload arm. Folds in a
    `_restore` assertion too: both resolve gguf-b's OWN client, never
    gguf-a's (review fix, T6 round 2)."""
    w = two_gguf_watcher
    pending = {"resource": "gguf-b", "model": "b.gguf", "footprint": 30, "gpu_index": 3}

    w.watcher._execute([], pending)

    rec = w.intent_store.get()["local/gguf-b"]
    assert rec["state"] == "loaded" and rec["actor"] == "deck"
    assert "local/gguf-a" not in w.intent_store.get()
    assert w.clients.fake("gguf-b").loaded == ["b.gguf"]
    assert w.clients.fake("gguf-a").loaded == []

    # _restore dispatches through the SAME per-resource resolution —
    # resolves gguf-b's client off the intent key, not gguf-a's.
    w.watcher._restore({"key": "local/gguf-b", "engine": "lemonade", "model": "b.gguf"})
    assert w.clients.fake("gguf-b").loaded == ["b.gguf", "b.gguf"]
    assert w.clients.fake("gguf-a").loaded == []


def test_unload_failures_from_different_resources_are_not_deduped(two_gguf_watcher):
    """Two DIFFERENT lemonade-kind resources failing unload with the SAME
    model name and error text must produce TWO distinguishable events, not
    collapse into one via `_log`'s failure-dedup memo (review fix, T6
    round 2: the memo's key derives from `detail`, and the pre-fix
    "unload-failed"/"load-failed" details carried no `resource`, so an
    identical failure from a DIFFERENT resource was silently swallowed as
    if it were the SAME resource's repeat)."""
    w = two_gguf_watcher
    err = EngineError("connection refused")
    w.clients.fake("gguf-a")._raise_on_unload = err
    w.clients.fake("gguf-b")._raise_on_unload = err

    w.watcher._execute([{"type": "unload", "resource": "gguf-a", "model": "x.gguf"}], None)
    w.watcher._execute([{"type": "unload", "resource": "gguf-b", "model": "x.gguf"}], None)

    failed = [e for e in tail_events(w.events_path) if e["kind"] == "unload-failed"]
    assert len(failed) == 2  # neither swallowed the other as a "repeat"
    assert {e["detail"]["resource"] for e in failed} == {"gguf-a", "gguf-b"}


def test_kind_for_uses_the_tick_cache_when_set(tmp_path):
    """review fix, T6 round 2: _kind_for must resolve from
    ``self._tick_engines`` (tick()'s own actuation-phase stash of the ONE
    declaration read that tick already took) rather than re-reading
    self._node_store on every call — proven directly against the seam
    (a counting fake node_store), not through a full contention tick:
    two_gguf_watcher's ``_policy()`` doesn't cover gguf-a/gguf-b, so a real
    tick there would noop both regardless of how many times the
    declaration got read."""

    class _CountingNodeStore:
        def __init__(self, engines):
            self._engines = engines
            self.calls = 0

        def get(self, node_id):
            self.calls += 1
            return {"engines": self._engines}

    node_store = _CountingNodeStore(
        [{"resource": "gguf-a", "kind": "lemonade"},
         {"resource": "gguf-b", "kind": "lemonade"}]
    )
    watcher = _watcher(tmp_path, node_store=node_store)

    # No cache (outside a tick, self._tick_engines still None): falls back
    # to a fresh read every call, same as before this fix.
    assert watcher._kind_for("gguf-a") == "lemonade"
    assert watcher._kind_for("gguf-b") == "lemonade"
    assert node_store.calls == 2

    # Cache set, as tick() does for the duration of its actuation phase:
    # both resolve from it, zero further reads.
    watcher._tick_engines = [{"resource": "gguf-a", "kind": "lemonade"},
                             {"resource": "gguf-b", "kind": "lemonade"}]
    assert watcher._kind_for("gguf-a") == "lemonade"
    assert watcher._kind_for("gguf-b") == "lemonade"
    assert node_store.calls == 2  # unchanged

    # Cleared (as tick()'s finally does): back to a fresh read.
    watcher._tick_engines = None
    assert watcher._kind_for("gguf-a") == "lemonade"
    assert node_store.calls == 3


class _FakeClock:
    """A monotonic()-alike callable clock, advanceable without sleeping —
    distinct from FakeClock above (HealSuppressor's clock) so the derive-pass
    throttle tests don't entangle with heal-suppression tests."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _checkpoint_tree(tmp_path):
    """One checkpoint directory with a minimal config.json, for
    Watcher._derive_pass's gguf_dir scan."""
    gguf_dir = tmp_path / "checkpoints"
    model_dir = gguf_dir / "some-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["SomeArch"]}))
    return gguf_dir


def _raises(exc):
    """A callable that raises `exc` when invoked, any arguments accepted —
    for injecting a derive-pass failure without a real unreadable checkpoint
    or engine. Accepts args/kwargs so the same helper covers both
    `on_derive` (called with none) and `engine_exec` (called with
    node/engine/interpreter/source)."""

    def _fn(*args, **kwargs):
        raise exc

    return _fn


def test_watcher_tick_heals_contention_then_reloads(tmp_path):
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    read_gpus = RecordingReadGpus(result=[{"probe": True}])
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
        read_gpus=read_gpus,
    )

    watcher.tick()

    assert comfy.freed == 1
    assert lemonade.loaded == ["extra.model.gguf"]  # re-triggered with full name
    events = tail_events(events_path)
    kinds = [e["kind"] for e in events]
    assert "free_comfyui" in kinds
    assert read_gpus.calls  # read_gpus was actually invoked with the sysfs roots
    assert "load-retriggered" in kinds
    retrig = next(e for e in events if e["kind"] == "load-retriggered")
    assert retrig["detail"] == {"model": "extra.model.gguf"}


def test_watcher_tick_heals_contention_load_failure_logged(tmp_path):
    """If the re-triggered lemonade.load() raises EngineError (engine
    unreachable, bad response, ...), the watcher logs 'load-failed' with the
    error string, does NOT crash, and the loop survives for the next tick."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade = FakeLemonade(raise_on_load=EngineError("connection refused"))
    comfy = FakeComfy()
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()  # must not raise

    assert comfy.freed == 1
    assert lemonade.loaded == []  # load raised -> nothing recorded as loaded
    events = tail_events(events_path)
    kinds = [e["kind"] for e in events]
    assert "load-failed" in kinds
    assert "load-retriggered" not in kinds
    failed = next(e for e in events if e["kind"] == "load-failed")
    assert "connection refused" in failed["detail"]["error"]


def test_watcher_tick_idle_release_unloads_and_logs(tmp_path):
    model = "extra.idle.gguf"
    snapshot = _world(
        lemonade=_lem(state="loaded", model=model, footprint=10 * GIB, idle_s=1000),
        default_route=None,  # no pending inference
    )
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        lemonade=lemonade,
    )

    watcher.tick()

    assert lemonade.unloaded == [model]
    assert lemonade.loaded == []  # no pending -> no reload
    events = tail_events(events_path)
    assert events[-1]["kind"] == "unload_lemonade"
    assert events[-1]["detail"]["model"] == model


def test_watcher_guarderror_race_logged_and_no_reload(tmp_path):
    """If comfy's queue fills between decide and execute, free() raises
    GuardError; the watcher logs 'free-raced', does NOT crash, and does NOT
    re-trigger the load (the VRAM was never actually reclaimed)."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy(raise_guard=True)
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()  # must not raise

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "free-raced" in kinds
    assert lemonade.loaded == []  # eviction raced -> no premature reload


def test_failed_unload_does_not_abort_the_tick_or_strand_intent(tmp_path):
    """Idle-TTL fires an unload while lemonade is briefly unresponsive
    (EngineError). Per-action isolation — _execute_restore's documented
    invariant (arbiter.py's _execute_restore docstring), applied to the arm
    that lacked it
    [max-review #9]. The tick must:

    1. keep going — the reconcile pass still runs, so a due hipfire restore
       fires (before this fix the raise escaped into tick()'s broad catch,
       skipping reconcile/derive/provenance for the whole tick);
    2. roll the just-recorded 'unloaded' intent back to its prior value —
       the actuation did NOT happen, and 'unloaded' standing against a
       still-loaded model derives the inert 'unexpected' status;
    3. log 'unload-failed'.
    """
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    # Seeded AWAY from record()'s model=None default: a rollback that
    # "restores" nothing would still pass a None-vs-None assertion.
    intent.record("local/lemonade", state="loaded", model="extra.foo.gguf",
                  engine="lemonade")
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire")

    lemonade = FakeLemonade(raise_on_unload=EngineError("lemonade unreachable"))
    hipfire = FakeHipfire(state="parked")  # intent says loaded -> restore is due
    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.foo.gguf", idle_s=1000),
        hipfire=_hip(state="parked"),
        default_route=None,  # no pending inference to re-trigger
    )
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot), FakeRegistry(), _policy(lem_idle=900),
        lemonade=lemonade, hipfire=hipfire, intent_store=intent,
    )

    watcher.tick()  # must not raise

    assert lemonade.unloaded == ["extra.foo.gguf"]  # it WAS dispatched
    assert "resume" in hipfire.calls  # (1) the tick carried on into reconcile
    record = intent.get()["local/lemonade"]  # (2) rolled back, not stranded
    assert record["state"] == "loaded"
    assert record["model"] == "extra.foo.gguf"
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "unload-failed" in kinds  # (3)
    assert "unload_lemonade" not in kinds  # never claim an unload that failed


def test_failed_unload_rollback_preserves_the_prior_actor_and_timestamp(tmp_path):
    """The rollback RESTORES the prior record — it does not author a fresh
    deck-stamped one. Two independent consumers depend on that, and neither
    is exercised by the sibling test above:

    * ``actor`` — app.routers.control's pull-through supersession check
      honors ONLY operator-authored records; relabeling an operator's record
      "deck" would hide it there, silently dropping the operator's override.
    * ``updated_ts`` — it is the settings-drift baseline
      (app/routers/__init__.py:131-148), documented to move only at a
      DELIBERATE load/unload, i.e. when a process relaunches and re-consumes
      its settings. A failed unload relaunched nothing, so a fresh stamp
      would silently clear a legitimate "settings changed since launch" flag.
    * ``quarantined``/``failures`` — record() resets both, so only a verbatim
      put-back returns a quarantined key to its quarantine; re-recording
      would put a crash-looping resource back into the restore rotation.

    Every fixture field is set away from what a naive rollback would write:
    operator-authored (not "deck"), an OLD timestamp (not now), and
    quarantined with a non-zero failure count (not the 0/False reset).
    """
    from app.intent import IntentStore

    old_ts = "2026-08-09T12:00:00+00:00"
    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/lemonade", state="loaded", model="extra.foo.gguf",
                  engine="lemonade", actor="operator", now=old_ts)
    intent.note_failure("local/lemonade")
    intent.note_failure("local/lemonade")  # -> FAILURE_BUDGET -> quarantined
    prior = intent.get()["local/lemonade"]
    assert prior["quarantined"] is True  # fixture precondition

    lemonade = FakeLemonade(raise_on_unload=EngineError("lemonade unreachable"))
    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.foo.gguf", idle_s=1000),
        default_route=None,
    )
    watcher, _events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot), FakeRegistry(), _policy(lem_idle=900),
        lemonade=lemonade, intent_store=intent,
    )

    watcher.tick()

    record = intent.get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["actor"] == "operator"   # not relabeled to "deck"
    assert record["updated_ts"] == old_ts  # not advanced by a failed unload


def test_failed_free_comfyui_skips_retrigger_and_logs(tmp_path):
    """An EngineError from free() (vs the already-caught GuardError race):
    logged as 'free-failed', and the VRAM is treated as NOT reclaimed — so
    the pending default-route load is not re-triggered into a GPU that may
    still be full."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade = FakeLemonade()
    comfy = FakeComfy(raise_on_free=EngineError("comfy unreachable"))
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()  # must not raise

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "free-failed" in kinds
    assert "free_comfyui" not in kinds  # never claim a free that failed
    assert lemonade.loaded == []  # VRAM not confirmed reclaimed -> no reload


def test_watcher_successful_free_rearms_comfy_idle_clock(tmp_path):
    """A successful free_comfyui must re-arm the world's comfy idle clock —
    otherwise idle_s keeps growing and the idle-release rule re-emits the
    free on every subsequent tick for as long as comfy stays idle."""
    snapshot = _world(
        gpus=[_gpu(index=1, used=6 * GIB)],
        comfyui=_comfy(state="idle", queue=0, idle_s=5000),
    )
    world, comfy = FakeWorld(snapshot), FakeComfy()
    watcher, _ = _make_watcher(
        tmp_path, world, FakeRegistry(), _policy(comfy_idle=300), comfy=comfy
    )

    watcher.tick()

    assert comfy.freed == 1
    assert world.comfy_freed_notes == 1


def test_watcher_raced_free_does_not_rearm_comfy_idle_clock(tmp_path):
    """A GuardError'd free reclaimed nothing — the idle clock must NOT be
    re-armed, so the free is retried once comfy actually drains."""
    snapshot = _world(
        gpus=[_gpu(index=1, used=6 * GIB)],
        comfyui=_comfy(state="idle", queue=0, idle_s=5000),
    )
    world, comfy = FakeWorld(snapshot), FakeComfy(raise_guard=True)
    watcher, _ = _make_watcher(
        tmp_path, world, FakeRegistry(), _policy(comfy_idle=300), comfy=comfy
    )

    watcher.tick()

    assert world.comfy_freed_notes == 0


def test_watcher_tick_error_swallowed_and_loop_survives(tmp_path):
    world = FakeWorld(snapshot=None, raises=KeyError("boom"))
    watcher, events_path = _make_watcher(tmp_path, world, FakeRegistry(), _policy())

    watcher.tick()  # first tick errors...
    watcher.tick()  # ...and the second still runs (loop survives)

    # Loop survived both ticks: snapshot() was attempted twice.
    assert world.calls == 2
    # But consecutive IDENTICAL tick-errors collapse to a single log line (I6).
    events = tail_events(events_path)
    tick_errors = [e for e in events if e["kind"] == "tick-error"]
    assert len(tick_errors) == 1
    assert "boom" in tick_errors[0]["detail"]["error"]


def test_watcher_tick_error_dedup_resets_on_different_detail(tmp_path):
    """Repeated identical tick-errors log once; a different error detail in
    between resets the suppression so the next one logs again (I6)."""
    world = FakeWorld(snapshot=None, raises=KeyError("boom"))
    watcher, events_path = _make_watcher(tmp_path, world, FakeRegistry(), _policy())

    watcher.tick()  # 'boom' -> logged
    watcher.tick()  # 'boom' again -> deduped
    world._raises = KeyError("different")
    watcher.tick()  # 'different' -> resets + logged
    world._raises = KeyError("boom")
    watcher.tick()  # 'boom' again, but preceded by 'different' -> logged

    tick_errors = [e for e in tail_events(events_path) if e["kind"] == "tick-error"]
    assert len(tick_errors) == 3
    assert "boom" in tick_errors[0]["detail"]["error"]
    assert "different" in tick_errors[1]["detail"]["error"]
    assert "boom" in tick_errors[2]["detail"]["error"]


def test_watcher_suppresses_consecutive_duplicate_noop(tmp_path):
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="busy", queue=2, idle_s=0),  # nothing evictable
        default_route="extra.model.gguf",
    )
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
    )

    watcher.tick()
    watcher.tick()
    watcher.tick()

    noops = [e for e in tail_events(events_path) if e["kind"] == "noop"]
    assert len(noops) == 1
    assert noops[0]["detail"]["reason"] == "wont-fit"


def test_watcher_no_pending_when_free_already_fits(tmp_path):
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=4 * GIB)],  # free = 30 GiB
        lemonade=_lem(state="unloaded"),
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()

    assert lemonade.loaded == []  # free VRAM already fits -> no pending inferred
    assert comfy.freed == 0


def test_watcher_no_pending_when_model_not_in_registry(tmp_path):
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),  # below TTL: no idle release
        default_route="extra.notgguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(missing={"notgguf"}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()

    assert lemonade.loaded == []
    assert comfy.freed == 0  # not a loadable GGUF -> no contention healing


def test_watcher_thread_runs_a_tick_and_stops_promptly(tmp_path):
    """start() runs at least one tick against stub deps; stop() joins well
    within 2 s."""
    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=1),
        default_route=None,
    )
    world = FakeWorld(snapshot)
    watcher, _ = _make_watcher(tmp_path, world, FakeRegistry(), _policy())

    watcher.start()
    deadline = time.time() + 1.5
    while world.calls < 1 and time.time() < deadline:
        time.sleep(0.005)
    assert world.calls >= 1  # at least one tick happened

    t0 = time.time()
    watcher.stop()
    elapsed = time.time() - t0

    assert elapsed < 2.0
    assert "model-deck-watcher" not in {t.name for t in threading.enumerate()}


# ===========================================================================
# C2 — HealSuppressor (unit) + watcher suppression behavior
# ===========================================================================


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_heal_suppressor_notes_and_expires():
    clock = FakeClock(100.0)
    s = HealSuppressor(window_s=600, clock=clock)

    assert s.suppressed() is False  # fresh -> not suppressed
    s.note_deck_unload()
    assert s.suppressed() is True  # armed

    clock.t = 100.0 + 599  # still inside the window
    assert s.suppressed() is True
    clock.t = 100.0 + 600  # window elapsed (>= boundary)
    assert s.suppressed() is False  # expired
    assert s.suppressed() is False  # stays disarmed


def test_heal_suppressor_clear_disarms():
    s = HealSuppressor(window_s=600, clock=FakeClock(0.0))
    s.note_deck_unload()
    assert s.suppressed() is True
    s.clear()  # a deck-initiated load clears it
    assert s.suppressed() is False


def test_watcher_suppressed_skips_pending_inference(tmp_path):
    """While suppressed, the watcher does NOT infer a pending load, so a
    contention it would normally heal is left alone (idle rules still run, but
    here nothing is idle past its TTL)."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),  # below TTL: no idle free
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    suppressor = HealSuppressor(window_s=600, clock=FakeClock(0.0))
    suppressor.note_deck_unload()  # armed -> suppressed
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
        heal_suppressor=suppressor,
    )

    watcher.tick()

    # No contention healing happened: comfy not freed, nothing reloaded.
    assert comfy.freed == 0
    assert lemonade.loaded == []
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "free_comfyui" not in kinds
    assert "load-retriggered" not in kinds


def test_watcher_not_suppressed_heals_same_scenario(tmp_path):
    """Control for the test above: with suppression OFF, the very same
    snapshot DOES heal the contention (proving suppression is what blocks it)."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=10),
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
    )

    watcher.tick()

    assert comfy.freed == 1
    assert lemonade.loaded == ["extra.model.gguf"]


def test_watcher_unload_engages_suppressor(tmp_path):
    """The watcher's own deck-initiated unload (here an idle release) arms the
    suppressor so a later tick won't heal it back."""
    model = "extra.idle.gguf"
    snapshot = _world(
        lemonade=_lem(state="loaded", model=model, footprint=10 * GIB, idle_s=1000),
        default_route=None,
    )
    suppressor = HealSuppressor(window_s=600, clock=FakeClock(0.0))
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        lemonade=FakeLemonade(),
        heal_suppressor=suppressor,
    )

    assert suppressor.suppressed() is False
    watcher.tick()  # idle release unloads lemonade
    assert suppressor.suppressed() is True


def test_watcher_heal_load_clears_suppressor(tmp_path):
    """A heal re-trigger LOAD clears a previously-armed suppression."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    # Start suppressed but with an OLD unload; a fresh tick that heals + loads
    # must clear it. (suppressed() is checked each tick; inject a clock so the
    # window hasn't expired, proving the LOAD — not expiry — cleared it.)
    suppressor = HealSuppressor(window_s=600, clock=FakeClock(0.0))
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=FakeLemonade(),
        comfy=FakeComfy(),
        heal_suppressor=suppressor,
    )

    watcher.tick()  # not suppressed -> heals contention -> loads -> clears
    assert suppressor.suppressed() is False


# ===========================================================================
# I1 — watcher yields to an in-flight actuator holding app.actuation.LOCK
# (max-review #7: the tick's yield used to be a start-of-tick PEEK against
# app.sets.apply_in_progress(); a tick already past that peek kept actuating
# even though an apply had since started, interleaving restores with the
# apply's own evictions/loads. The peek is gone — tick() now takes its
# (cheap, read-only) snapshot unconditionally, then TRY-ACQUIRES
# app.actuation.LOCK around the actuation+reconcile phase specifically. The
# three tests below prove: (1) lock-held-for-the-whole-tick still yields
# actuation cleanly, same observable effect as the old peek; (2) a lock that
# materializes only AFTER the snapshot — the exact race the peek could not
# see — is still caught, because the try-acquire is checked at the moment of
# actuation, not at tick start; (3) the direction that matters most on a live
# box: an apply blocks out a tick's in-flight actuation, never interleaves
# with it.)
# ===========================================================================


def test_tick_actuation_yields_while_lock_held(tmp_path):
    """Same-thread try-acquire probe: this test's own thread holds
    actuation.LOCK for the whole tick (a plain threading.Lock is non-
    reentrant, so tick()'s own acquire(blocking=False) fails exactly as it
    would if a DIFFERENT thread held it) — a tick with a due idle-release
    unload must perform ZERO engine calls. The snapshot itself still runs
    (unconditionally, cheap, read-only) — only the actuation+reconcile
    phase is gated. [c42]. The cross-thread cases (a real other thread
    holding the lock, on both sides of the race) are covered by this
    test's siblings below,
    ``test_tick_yields_when_lock_acquired_after_the_snapshot`` and
    ``test_apply_waits_for_an_in_flight_tick_actuation``."""
    from app import actuation

    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=20 * GIB)],
        lemonade=_lem(state="loaded", model="m.gguf", idle_s=1000),
    )
    world = FakeWorld(snapshot)
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path, world, FakeRegistry(), _policy(lem_idle=300), lemonade=lemonade
    )

    with actuation.LOCK:
        watcher.tick()

    assert world.calls == 1  # the snapshot DOES run now — no start-of-tick peek
    assert lemonade.unloaded == []  # but nothing was actuated
    assert tail_events(events_path) == []  # and nothing was logged


def test_tick_yields_when_lock_acquired_after_the_snapshot(tmp_path):
    """The race [c42] flags and the old start-of-tick peek could not catch:
    the lock is free when the tick's snapshot is taken, and only acquired by
    another actor (simulating a set apply's HTTP thread) WHILE the snapshot
    call is in flight — i.e. strictly after where the old peek ran. The
    try-acquire at the actuation boundary must still yield cleanly; a peek
    checked only at tick start could not have seen this."""
    from app import actuation

    class _LockGrabbedDuringSnapshot:
        """Like FakeWorld, but the moment snapshot() is called it hands the
        lock to another thread and waits for confirmation THAT THREAD holds
        it before returning — reproducing "someone else acquired the lock
        after this tick's peek would have already passed"."""

        def __init__(self, snapshot):
            self._snapshot = snapshot
            self.calls = 0
            self._release = threading.Event()
            self._locker = None

        def snapshot(self, gpus, engines, clients, litellm, registry):
            # E1 Task 3 arity rename (see the file-level FakeWorld.snapshot
            # for the same change) — this local class never inspected these
            # params either.
            self.calls += 1
            acquired = threading.Event()

            def _hold():
                actuation.LOCK.acquire()
                acquired.set()
                self._release.wait(timeout=5)
                actuation.LOCK.release()

            self._locker = threading.Thread(target=_hold, daemon=True)
            self._locker.start()
            assert acquired.wait(timeout=5), "other actor never acquired the lock"
            return self._snapshot

        def release_other_lock_holder(self):
            self._release.set()
            self._locker.join(timeout=5)
            assert not self._locker.is_alive()

    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=20 * GIB)],
        lemonade=_lem(state="loaded", model="m.gguf", idle_s=1000),
    )
    world = _LockGrabbedDuringSnapshot(snapshot)
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path, world, FakeRegistry(), _policy(lem_idle=300), lemonade=lemonade
    )

    try:
        watcher.tick()

        assert world.calls == 1  # the snapshot ran BEFORE the lock was ever held
        assert lemonade.unloaded == []  # yet the actuation phase still yielded
        assert tail_events(events_path) == []
    finally:
        # try/finally, not a bare call after the asserts: a failed assertion
        # above must not leak the OTHER thread's held actuation.LOCK past
        # this test — every later test in the process shares this same
        # module-level lock, and a still-locked LOCK would falsely make the
        # NEXT test's tick look like it's yielding to something.
        world.release_other_lock_holder()
    assert actuation.in_progress() is False  # sanity: fully cleaned up


def test_apply_waits_for_an_in_flight_tick_actuation(tmp_path):
    """The other direction: a tick's actuation (fake unload) blocks on an
    event; a concurrent apply() must not reach its own first engine call
    until the tick releases the lock. Bounded — a deadlock fails the test,
    not the suite."""
    from tests.test_sets import (
        RecComfy,
        RecHipfire,
        RecHostAgent,
        RecLemonade,
        RecLocalClients,
        RecPolicyStore,
    )
    from tests.test_sets import _eph as sets_eph
    from tests.test_sets import make_world as sets_make_world

    from app.sets import ConfigSet, SetStore, apply

    entered = threading.Event()
    release = threading.Event()

    class BlockingLemonade(FakeLemonade):
        """unload() signals entry then blocks — holds actuation.LOCK open
        from inside a tick so a concurrent apply() can be proven to wait."""

        def unload(self, model):
            entered.set()
            release.wait(timeout=5)
            super().unload(model)

    tick_snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=20 * GIB)],
        lemonade=_lem(state="loaded", model="m.gguf", idle_s=1000),
    )
    tick_world = FakeWorld(tick_snapshot)
    tick_lemonade = BlockingLemonade()
    watcher, _ = _make_watcher(
        tmp_path, tick_world, FakeRegistry(), _policy(lem_idle=300), lemonade=tick_lemonade
    )

    tick_thread = threading.Thread(target=watcher.tick)
    tick_thread.start()
    assert entered.wait(timeout=5), "tick never reached its blocking unload"

    apply_world = sets_make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg = ConfigSet(name="chat", ephemeral=sets_eph(lemonade={"state": "loaded"}))
    apply_lemonade = RecLemonade()
    apply_clients = {
        "local_clients": RecLocalClients(apply_lemonade, RecComfy(), RecHipfire()),
        "hostagent": RecHostAgent(),
        "policy_store": RecPolicyStore(),
        "store": SetStore(tmp_path / "sets"),
        "events_path": tmp_path / "apply-events.jsonl",
    }

    results = {}
    apply_thread = threading.Thread(
        target=lambda: results.__setitem__(
            "apply", apply(cfg, world=apply_world, **apply_clients)
        )
    )
    apply_thread.start()

    try:
        # While the tick holds the lock (blocked mid-unload), the apply must
        # not get anywhere near its own engine call.
        apply_thread.join(timeout=0.5)
        assert apply_thread.is_alive(), "apply proceeded despite the tick's held lock"
        assert apply_lemonade.calls == []
    finally:
        # try/finally: a failed assertion above must not leave the tick
        # thread's blocking unload() waiting out its own 5 s internal
        # timeout while later tests run — release it immediately either way.
        release.set()
    tick_thread.join(timeout=5)
    apply_thread.join(timeout=5)
    assert not tick_thread.is_alive()
    assert not apply_thread.is_alive()

    assert tick_lemonade.unloaded == ["m.gguf"]
    assert apply_lemonade.calls == [("load", "extra.d.gguf")]
    assert results["apply"]["failed"] is None


# ===========================================================================
# I2 — watcher yields to an in-flight host-agent lifecycle operation
# ===========================================================================


class _BusyHostAgent:
    def lifecycle(self):
        return {"active": True, "operation": "model_activation", "target": "qwen3-30b"}


class _IdleHostAgent:
    def lifecycle(self):
        return {"active": False, "operation": None, "target": None}


class _ExplodingHostAgent:
    """Raises if probed — proves the watcher only checks lifecycle() when
    there is real work to skip, never on a tick with nothing to do."""

    def lifecycle(self):
        raise AssertionError("hostagent must not be probed when there is no real work")


def test_tick_skips_actions_while_host_agent_busy(tmp_path):
    """Same scenario as test_watcher_tick_idle_release_unloads_and_logs, plus
    a busy hostagent: the idle-release unload must be suppressed and a
    'host-agent-busy' event logged instead of the unload."""
    model = "extra.idle.gguf"
    snapshot = _world(
        lemonade=_lem(state="loaded", model=model, footprint=10 * GIB, idle_s=1000),
        default_route=None,
    )
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        lemonade=lemonade,
        hostagent=_BusyHostAgent(),
    )

    watcher.tick()

    assert lemonade.unloaded == []  # action suppressed
    events = tail_events(events_path)
    assert events[-1]["kind"] == "host-agent-busy"
    assert events[-1]["detail"]["operation"] == "model_activation"
    assert events[-1]["detail"]["target"] == "qwen3-30b"


def test_tick_executes_when_host_agent_idle(tmp_path):
    """Same scenario, but an idle hostagent must not block the unload —
    behavior unchanged from the no-hostagent case."""
    model = "extra.idle.gguf"
    snapshot = _world(
        lemonade=_lem(state="loaded", model=model, footprint=10 * GIB, idle_s=1000),
        default_route=None,
    )
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        lemonade=lemonade,
        hostagent=_IdleHostAgent(),
    )

    watcher.tick()

    assert lemonade.unloaded == [model]  # unchanged behavior
    events = tail_events(events_path)
    assert events[-1]["kind"] == "unload_lemonade"


def test_host_agent_busy_events_are_deduped(tmp_path):
    model = "extra.idle.gguf"
    snapshot = _world(
        lemonade=_lem(state="loaded", model=model, footprint=10 * GIB, idle_s=1000),
        default_route=None,
    )
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        hostagent=_BusyHostAgent(),
    )

    watcher.tick()
    watcher.tick()

    events = [e for e in tail_events(events_path) if e["kind"] == "host-agent-busy"]
    assert len(events) == 1


def test_idle_tick_does_not_probe_host_agent_when_nothing_to_do(tmp_path):
    """No idle-past-TTL, no pending load -> decide() returns []. There is no
    real work to skip, so the watcher must not even call lifecycle()."""
    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=10),
        default_route=None,
    )
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(),
        _policy(lem_idle=900),
        hostagent=_ExplodingHostAgent(),
    )

    watcher.tick()  # must not raise (an exploding lifecycle() would blow up)

    assert tail_events(events_path) == []


# ===========================================================================
# last_used observation on the heal re-trigger load (I5)
# ===========================================================================


class _RecCatalog:
    def __init__(self):
        self.noted = []

    def note_used_gguf(self, filename):
        self.noted.append(filename)


def test_watcher_heal_load_notes_last_used(tmp_path):
    """The watcher's heal re-trigger is a real load of a real model. Without
    noting it, an auto-reloaded default-route model looks "never used" to the
    storage watcher's LRU ordering."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    catalog = _RecCatalog()
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        catalog=catalog,
    )

    watcher.tick()

    assert catalog.noted == ["model.gguf"]      # bare name, "extra." stripped


def test_watcher_failed_heal_load_does_not_note_last_used(tmp_path):
    """Only a load that actually succeeded counts as an observation."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    catalog = _RecCatalog()
    watcher, _ = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=FakeLemonade(raise_on_load=EngineError("connection refused")),
        catalog=catalog,
    )

    watcher.tick()

    assert catalog.noted == []


# ===========================================================================
# WATCHER — LIFECYCLE RECONCILE PASS (task 8)
# ===========================================================================
#
# The reconcile pass answers a different question from arbitration: not
# "who must give up VRAM right now" but "is what we asked for still true".
# It runs after arbitration on the same snapshot — see Watcher._reconcile_pass
# for why that order is load-bearing.


class FakeSpark:
    """Mirrors SparkClient's real surface: status() returns the NODE payload
    shape ({"profiles", "swap_status", "serving"}) — not the observation
    shape — and swap() takes a profile name. models() returns the /v1/models
    body consumed by the characteristics derive pass. The watcher does the
    translation, so faking the node payload is what actually exercises it."""

    def __init__(self, profile="heretic", serving_model=None, endpoint_ok=False,
                 swap_state="error", swap_ts=None, raises=None,
                 models_body=None, models_raises=None):
        self.calls = []
        self.raises = raises
        self.swap_fail = None
        self.models_raises = models_raises
        self.models_calls = 0
        self._models_body = models_body if models_body is not None else {"data": []}
        self._payload = {
            "profiles": [{"name": profile, "engine": "vllm"}],
            "swap_status": {"state": swap_state, "profile": profile,
                            "ts": swap_ts},
            "serving": {"model": serving_model, "endpoint_ok": endpoint_ok,
                        "container_status": None},
        }

    def status(self):
        if self.raises is not None:
            raise self.raises
        return self._payload

    def swap(self, profile, force=False):
        # Record the ATTEMPT first, guard second. Watcher._execute_restore
        # swallows every exception _restore raises into a
        # lifecycle-restore-failed event, so a reconciler regression that
        # dispatches a swap during an in-flight boot (or while spark is
        # unreachable) must still show up in self.calls — appending only
        # after the guard clears would let that regression hide behind the
        # swallowed GuardError/EngineError, leaving `spark.calls == []`
        # true for the wrong reason instead of proving no attempt was made.
        self.calls.append(("swap", profile))
        # The REAL client's boot-window guard, via the real shared judgement
        # (app.engines.spark.boot_in_flight) — a fake that omits it hides
        # guard regressions from every arbiter test.
        status = self.status()
        serving = status.get("serving") or {}
        if not serving.get("endpoint_ok") and not force and boot_in_flight(
                {"swap_status": status.get("swap_status"), "serving": serving}):
            raise GuardError("previous swap is still booting")
        if self.swap_fail is not None:
            raise self.swap_fail
        # A successful swap stamps a FRESH swap_status.ts, exactly as the
        # real node-agent does (swapctl.request_swap writes "swapping" at
        # request time). Tests that tick more than once depend on this: it
        # is what makes an invalidated observer see the swap's own effect
        # instead of the stale pre-swap snapshot (2026-08-06 review, C1).
        from datetime import UTC, datetime
        self._payload = {
            **self._payload,
            "swap_status": {"state": "done", "profile": profile,
                            "ts": datetime.now(UTC).isoformat()},
        }
        return {"id": "u1", "profile": profile}

    def models(self):
        self.models_calls += 1
        if self.models_raises is not None:
            raise self.models_raises
        return self._models_body


class FakeObservers:
    """Minimal NodeObservers stand-in: fixed observer map (task 8). Unlike
    the real NodeObservers, this never re-reads a registry -- a test wires
    exactly the node(s) it means to exercise."""

    def __init__(self, observers):
        self._observers = dict(observers)

    def snapshot(self):
        return dict(self._observers)

    def observer_for(self, node_id):
        return self._observers.get(node_id)

    def invalidate(self, node_id):
        # DELEGATES, exactly as the real NodeObservers does
        # (app/node_clients.py). It used to be a no-op `pass`, which made
        # this fake structurally unable to observe the 2026-08-06 C1 bug:
        # a reconciler that never invalidated after a successful swap read
        # identically here and in production, and only production
        # quarantined a model that had just come back.
        observer = self._observers.get(node_id)
        if observer is not None:
            observer.invalidate()


class FakeSlotObserver:
    """Minimal single-node observer stand-in: status() returns a FIXED
    value in app.observe.SparkObserver's STATUS shape (what its own
    status() returns, e.g. {"profile", "serving", "reachable",
    "swap_in_progress"} -- observe_spark's input, not its output) so a test
    can drive a node's derived state without a real node payload or the
    real translation/caching. No error path: a test that needs a parked
    error wraps a real SparkObserver around a raising client instead (see
    test_unreachable_event_names_the_node), which exercises the real
    park/clear mechanism rather than a second, hand-rolled copy of it."""

    def __init__(self, status):
        self._status = status

    def status(self):
        return self._status

    def take_error(self):
        return None


def _sparky_wiring(spark):
    """node_clients/node_observers kwargs wiring a real SparkObserver
    around `spark` under the "sparky" node id -- the single-node shape
    every converted spark test below needs (task 8: node_clients/
    node_observers replace the old spark=/spark_observer= params). A real
    SparkObserver, not FakeSlotObserver, so the existing translation/
    caching/backoff/boot-window coverage keeps exercising the real class."""
    return {
        "node_clients": FakeNodeClients({"sparky": spark}),
        "node_observers": FakeObservers({"sparky": SparkObserver(lambda: spark)}),
    }


def _intent(tmp_path, key="local/hipfire", state="loaded", model=None, engine="hipfire"):
    from app.intent import IntentStore

    store = IntentStore(tmp_path / "intent.json")
    store.record(key, state=state, model=model, engine=engine)
    return store


def _reconcile_watcher(tmp_path, intent_store, hipfire=None, snapshot=None, **kw):
    """A watcher with nothing for arbitration to do, so only the reconcile
    pass can produce an action."""
    return _make_watcher(
        tmp_path,
        FakeWorld(snapshot if snapshot is not None else _world()),
        FakeRegistry(),
        _policy(),
        hipfire=hipfire if hipfire is not None else FakeHipfire(state="parked"),
        intent_store=intent_store,
        **kw,
    )


def test_tick_restores_a_down_resource(tmp_path):
    """The reboot case: intent says loaded, nothing is loaded, so restore."""
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    watcher, events_path = _reconcile_watcher(tmp_path, store, hipfire=hipfire)

    watcher.tick()

    assert "resume" in hipfire.calls
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-restore" in kinds


def test_tick_does_not_restore_a_parked_resource(tmp_path):
    """THE regression. A deliberate park must survive every tick forever."""
    store = _intent(tmp_path, state="unloaded", model=None)
    hipfire = FakeHipfire(state="parked")
    watcher, _ = _reconcile_watcher(tmp_path, store, hipfire=hipfire)

    watcher.tick()
    watcher.tick()
    watcher.tick()

    assert "resume" not in hipfire.calls


def test_tick_does_not_restore_an_unmanaged_resource(tmp_path):
    """No intent recorded at all is not a licence to act: the deck did not
    author this state, so it reports and leaves it alone."""
    from app.intent import IntentStore

    store = IntentStore(tmp_path / "intent.json")   # nothing recorded
    hipfire = FakeHipfire(state="parked")
    watcher, _ = _reconcile_watcher(tmp_path, store, hipfire=hipfire)

    watcher.tick()

    assert hipfire.calls == []


def test_tick_stamps_last_healthy_when_serving(tmp_path):
    store = _intent(tmp_path)
    watcher, _ = _reconcile_watcher(
        tmp_path, store,
        snapshot=_world(hipfire=_hip(state="running", model="gpt-oss")),
        hipfire=FakeHipfire(state="running"),
    )

    watcher.tick()

    assert store.get()["local/hipfire"]["last_healthy_ts"] is not None


def test_two_failed_restores_quarantine_and_stop_retrying(tmp_path):
    """The crash-loop guard: after FAILURE_BUDGET attempts, stop.

    Clock advances past _RESTORE_COOLDOWN_S between ticks: this test's
    invariant is the failure budget, not the Task 5 per-key cooldown, so the
    cooldown must not itself suppress the second (budget-exhausting) attempt.
    Ticks 3-4 stay at the same time on purpose — once quarantined, status is
    'quarantined' not 'down' (app.lifecycle.derive_status), so plan_reconcile
    emits nothing for them regardless of cooldown state."""
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    hipfire.fail = EngineError("boom")
    clock = _FakeClock()
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, hipfire=hipfire, clock=clock,
    )

    watcher.tick()
    clock.advance(31)
    watcher.tick()
    watcher.tick()
    watcher.tick()

    assert hipfire.calls.count("resume") == 2
    assert store.get()["local/hipfire"]["quarantined"] is True
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-quarantined" in kinds


def test_restore_failure_is_logged_as_an_event(tmp_path):
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    hipfire.fail = EngineError("boom")
    watcher, events_path = _reconcile_watcher(tmp_path, store, hipfire=hipfire)

    watcher.tick()

    events = tail_events(events_path)
    failed = [e for e in events if e["kind"] == "lifecycle-restore-failed"]
    assert failed
    assert failed[0]["detail"]["key"] == "local/hipfire"
    assert "boom" in failed[0]["detail"]["error"]


def test_auto_disabled_stops_restores(tmp_path):
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    watcher, _ = _reconcile_watcher(tmp_path, store, hipfire=hipfire, auto=False)

    watcher.tick()

    assert "resume" not in hipfire.calls


def test_reconcile_error_does_not_kill_the_tick(tmp_path):
    """Supervisor-loop rule: a reconcile bug must not stop arbitration, and an
    unroutable engine must be isolated to its own key rather than escaping
    into the tick's broad catch."""
    store = _intent(tmp_path, model="x", engine="nonexistent-engine")
    watcher, events_path = _reconcile_watcher(tmp_path, store)

    watcher.tick()   # must not raise

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-restore-failed" in kinds
    assert "tick-error" not in kinds


def test_reconcile_pass_is_a_noop_without_an_intent_store(tmp_path):
    """Every watcher constructed before this task (and every caller that
    doesn't want lifecycle) keeps working untouched."""
    hipfire = FakeHipfire(state="parked")
    watcher, _ = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(), hipfire=hipfire,
    )

    watcher.tick()

    assert hipfire.calls == []


def test_a_held_key_is_not_restored(tmp_path):
    """An announced absence is not a death.

    The mirror of test_tick_restores_a_down_resource above: identical setup,
    one hold, opposite outcome.
    """
    from app.holds import HoldStore

    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    holds = HoldStore()
    holds.hold("local/hipfire", 60.0)
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, hipfire=hipfire, hold_store=holds)

    watcher.tick()

    assert "resume" not in hipfire.calls
    assert "lifecycle-restore" not in [e["kind"] for e in tail_events(events_path)]


def test_a_held_key_charges_no_failure_and_leaves_no_stamp(tmp_path):
    """A skipped restore is a non-action — matches the cooldown skip.

    Seed `_restore_unverified` so the failure-budget charge path (arbiter.py
    :1317-1321) is live for this key: without that seed, a first tick can
    never reach the charge (it's populated only by a previous
    `_execute_restore`), and the `failures`/`quarantined` assertions below
    would pass identically whether the hold guard exists or not. Review
    finding (Task 2 fix round): the RED proof lives in task-2-report.md.
    """
    from app.holds import HoldStore

    store = _intent(tmp_path)
    holds = HoldStore()
    holds.hold("local/hipfire", 60.0)
    watcher, _events_path = _reconcile_watcher(
        tmp_path, store, hold_store=holds)
    watcher._restore_unverified.add("local/hipfire")

    watcher.tick()

    assert "local/hipfire" not in watcher._restore_last_attempt_at
    assert store.get()["local/hipfire"].get("failures", 0) == 0
    assert store.get()["local/hipfire"].get("quarantined") is not True


def test_a_hold_protects_only_its_own_key(tmp_path):
    """Not a blanket amnesty: another key's hold does not cover hipfire."""
    from app.holds import HoldStore

    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    holds = HoldStore()
    holds.hold("local/lemonade", 60.0)
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, hipfire=hipfire, hold_store=holds)

    watcher.tick()

    assert "resume" in hipfire.calls
    assert "lifecycle-restore" in [e["kind"] for e in tail_events(events_path)]


def test_an_expired_hold_resumes_reconciliation(tmp_path):
    """Fail open: a crashed actuator cannot silence the deck forever."""
    from app.holds import HoldStore

    now = [1000.0]
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    holds = HoldStore(clock=lambda: now[0])
    holds.hold("local/hipfire", 60.0)
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, hipfire=hipfire, hold_store=holds)

    now[0] += 61.0  # nobody ever released it
    watcher.tick()

    assert "resume" in hipfire.calls
    assert "lifecycle-restore" in [e["kind"] for e in tail_events(events_path)]


def test_restore_runs_after_arbitration(tmp_path):
    """Ordering guard: arbitration's actions are executed before any restore,
    so reconciliation can never load something arbitration is about to evict."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],   # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
        hipfire=_hip(state="parked"),
    )
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="parked")
    watcher, events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        hipfire=hipfire,
        intent_store=store,
    )

    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.index("load-retriggered") < kinds.index("lifecycle-restore")


def test_restore_dispatches_a_lemonade_load(tmp_path):
    store = _intent(tmp_path, key="local/lemonade", model="extra.model.gguf",
                    engine="lemonade")
    lemonade = FakeLemonade()
    watcher, _ = _reconcile_watcher(tmp_path, store, lemonade=lemonade)

    watcher.tick()

    assert lemonade.loaded == ["extra.model.gguf"]


# --- spark slot -------------------------------------------------------------


def test_restore_dispatches_a_spark_swap(tmp_path):
    """A swap that errored never started a boot, so a down endpoint is a
    genuine 'down' and the profile is swapped back in."""
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", swap_state="error")
    watcher, _ = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()

    assert spark.calls == [("swap", "heretic")]


def test_restore_dispatches_a_spark_swap_after_the_boot_window_expires(tmp_path):
    """swap_status stays 'done' forever after a successful swap — a model that
    died hours later must still be restored, not read as 'still booting'."""
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", swap_state="done", swap_ts=old)
    watcher, _ = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()

    assert spark.calls == [("swap", "heretic")]


def test_spark_boot_in_flight_derives_warming_and_is_not_restored(tmp_path):
    """While a swap is booting, 'not loaded yet' and 'died' are the same
    observation and guessing wrong costs a multi-minute swap.

    The suppression is per-resource: observe_spark marks the slot
    ``transitioning``, which derives ``warming``, which plan_reconcile never
    acts on. (It used to be a GLOBAL boot-window flag that also froze local
    hipfire and lemonade — see tests/test_reconcile.py.)"""
    from datetime import UTC, datetime

    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", swap_state="done",
                      swap_ts=datetime.now(UTC).isoformat())
    watcher, _ = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()

    assert spark.calls == []


def test_unreachable_spark_is_not_a_dead_model(tmp_path):
    """Failing to look is not the same as looking and seeing nothing."""
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(raises=EngineError("connection refused"))
    watcher, events_path = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()

    assert spark.calls == []
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "tick-error" not in kinds


def test_spark_serving_the_intended_profile_is_left_alone(tmp_path):
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", serving_model="heretic", endpoint_ok=True)
    watcher, _ = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()

    assert spark.calls == []
    assert store.get()["sparky/slot0"]["last_healthy_ts"] is not None


def test_restore_dispatches_once_for_the_reboot_restore_incident_shape(tmp_path):
    """THE headline positive at the arbiter level: the 2026-08-05 incident's
    exact status.json shape — profile echoed as the served model,
    endpoint_ok False, swap_status hours-old (well past the 20-minute boot
    window) — must dispatch exactly ONE restore.

    Deliberately TWO ticks, not one (2026-08-06 review, C1): a one-tick
    version of this test would miss the bug entirely. Tick 1 restores
    correctly; it is tick 2+ that, without invalidating the node observer's
    cache after a successful swap, keeps re-deriving 'down' from the SAME
    stale pre-swap snapshot (its 10 s TTL outlives several 2 s tick
    intervals) and re-dispatches a SECOND swap for a restore that already
    worked — which in production hits the live guard's boot-window check,
    is refused as "still booting", and quarantines the resource within ~4 s
    of a restore that succeeded. FakeSpark.swap() stamps a fresh
    swap_status.ts on success (mirroring the real node-agent), so a
    correctly invalidated cache sees the swap's own effect on tick 2
    (transitioning, shielded) instead of stale pre-swap data."""
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC) - timedelta(hours=19)).isoformat()
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", serving_model="heretic", endpoint_ok=False,
                      swap_state="done", swap_ts=old)
    watcher, _ = _reconcile_watcher(tmp_path, store, **_sparky_wiring(spark))

    watcher.tick()
    watcher.tick()

    assert spark.calls == [("swap", "heretic")]


def test_no_spark_configured_emits_no_phantom_key(tmp_path):
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    watcher, events_path = _reconcile_watcher(tmp_path, store)

    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-restore" not in kinds
    assert "lifecycle-restore-failed" not in kinds


# --- multi-node swap (task 8: node_clients/node_observers replace spark) ----


def test_restore_swaps_the_right_nodes_client(tmp_path):
    """intent for boxb/slot0 down -> boxb's client (not boxa's) gets
    swap(profile) -- proves _restore resolves the node id out of the KEY
    itself, not "whichever client happens to be wired"."""
    store = _intent(tmp_path, key="boxb/slot0", model="heretic", engine="spark")
    box_a = FakeSpark(profile="aeon", serving_model="aeon", endpoint_ok=True)
    box_b = FakeSpark(profile="heretic", swap_state="error")
    node_clients = FakeNodeClients({"boxa": box_a, "boxb": box_b})
    node_observers = FakeObservers({
        "boxb": FakeSlotObserver(
            # reachable (the node responded) but nothing loaded -- observe_
            # spark's "down" shape, distinct from a probe FAILURE
            # ({"reachable": False, ...}), which derives "unreachable"
            # instead and is never actionable (app.reconcile.plan_reconcile).
            {"profile": None, "serving": None, "reachable": True,
             "swap_in_progress": False}),
    })
    watcher, _ = _reconcile_watcher(
        tmp_path, store, node_clients=node_clients, node_observers=node_observers)

    watcher.tick()

    assert box_a.calls == []
    assert box_b.calls == [("swap", "heretic")]


def test_restore_without_operable_client_charges_failure(tmp_path):
    """client_for returns None (node not operable) -> _restore raises
    EngineError -> _execute_restore's catch turns that into a
    lifecycle-restore-failed event, with the failure charged against
    boxb/slot0's own failure budget."""
    store = _intent(tmp_path, key="boxb/slot0", model="heretic", engine="spark")
    node_clients = FakeNodeClients({})  # boxb wired nowhere -> not operable
    node_observers = FakeObservers({
        "boxb": FakeSlotObserver(
            # reachable (the node responded) but nothing loaded -- observe_
            # spark's "down" shape, distinct from a probe FAILURE
            # ({"reachable": False, ...}), which derives "unreachable"
            # instead and is never actionable (app.reconcile.plan_reconcile).
            {"profile": None, "serving": None, "reachable": True,
             "swap_in_progress": False}),
    })
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_clients=node_clients, node_observers=node_observers)

    watcher.tick()

    events = tail_events(events_path)
    failed = [e for e in events if e["kind"] == "lifecycle-restore-failed"]
    assert failed
    assert failed[0]["detail"]["key"] == "boxb/slot0"
    assert store.get()["boxb/slot0"]["failures"] == 1


def test_reconcile_merges_every_nodes_observation(tmp_path):
    """Two FakeSlotObservers, one reachable-loaded, one down: statuses
    derive per node from the MERGED observation (app.observe.
    merge_observations over every node_observations() entry), and only the
    down one gets a restore action."""
    from app.intent import IntentStore

    store = IntentStore(tmp_path / "intent.json")
    store.record("boxa/slot0", state="loaded", model="aeon", engine="spark")
    store.record("boxb/slot0", state="loaded", model="heretic", engine="spark")
    box_b = FakeSpark(profile="heretic", swap_state="error")
    node_clients = FakeNodeClients({"boxb": box_b})
    node_observers = FakeObservers({
        "boxa": FakeSlotObserver(
            {"profile": "aeon", "serving": {"model": "aeon", "endpoint_ok": True},
             "reachable": True, "swap_in_progress": False}),
        "boxb": FakeSlotObserver(
            # reachable (the node responded) but nothing loaded -- observe_
            # spark's "down" shape, distinct from a probe FAILURE
            # ({"reachable": False, ...}), which derives "unreachable"
            # instead and is never actionable (app.reconcile.plan_reconcile).
            {"profile": None, "serving": None, "reachable": True,
             "swap_in_progress": False}),
    })
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_clients=node_clients, node_observers=node_observers)

    watcher.tick()

    assert box_b.calls == [("swap", "heretic")]
    assert store.get()["boxa/slot0"]["last_healthy_ts"] is not None
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-restore" in kinds


def test_unreachable_event_names_the_node(tmp_path):
    """An observer with a parked error -> lifecycle-spark-unreachable event
    whose detail includes {"node": "boxb"} -- existing event kind, tagged
    with whichever node actually failed (design §9), not a new kind."""
    store = _intent(tmp_path)  # local/hipfire only; not exercised here
    spark = FakeSpark(raises=EngineError("connection refused"))
    node_observers = FakeObservers({"boxb": SparkObserver(lambda: spark)})
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=node_observers)

    watcher.tick()

    events = tail_events(events_path)
    unreachable = [e for e in events if e["kind"] == "lifecycle-spark-unreachable"]
    assert unreachable
    assert unreachable[0]["detail"]["node"] == "boxb"
    assert "connection refused" in unreachable[0]["detail"]["error"]


# ---------------------------------------------------------------------------
# NODE-MISCONFIGURED surfacing (2026-08-13 incident follow-up, Part B deck
# half). The node-agent's serving payload may carry a "warning" field (its
# serving.py PROBE_URL_WARNING — vllm profiles configured, probe URL unset);
# agent logs are where signals go to die, so the watcher surfaces it in the
# Events tab as lifecycle-node-misconfigured, once per (node, warning-text)
# incident, re-armed when the warning disappears.
# ---------------------------------------------------------------------------


_MISCONFIG_WARNING = ("vllm profiles configured but NODE_SERVING_PROBE_URL "
                      "is unset — serving detection is blind")


def _node_status(warning=None):
    """A reachable node status in SparkObserver's STATUS shape; `warning`
    lands inside the serving payload exactly where the agent puts it."""
    serving = {"model": "laguna", "endpoint_ok": True, "container_status": None}
    if warning is not None:
        serving["warning"] = warning
    return {"profile": "laguna", "reachable": True, "swap_in_progress": False,
            "serving": serving}


def test_node_warning_logs_a_misconfigured_event(tmp_path):
    store = _intent(tmp_path)  # local/hipfire only; not exercised here
    obs = FakeSlotObserver(_node_status(warning=_MISCONFIG_WARNING))
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=FakeObservers({"sparky": obs}))

    watcher.tick()

    events = [e for e in tail_events(events_path)
              if e["kind"] == "lifecycle-node-misconfigured"]
    assert len(events) == 1
    assert events[0]["detail"] == {"node": "sparky",
                                   "warning": _MISCONFIG_WARNING}


def test_node_warning_is_logged_once_per_incident(tmp_path):
    """The warning holds tick after tick for as long as the config is wrong;
    one event per incident, not one per tick."""
    store = _intent(tmp_path)
    obs = FakeSlotObserver(_node_status(warning=_MISCONFIG_WARNING))
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=FakeObservers({"sparky": obs}))

    watcher.tick()
    watcher.tick()
    watcher.tick()

    events = [e for e in tail_events(events_path)
              if e["kind"] == "lifecycle-node-misconfigured"]
    assert len(events) == 1


def test_node_warning_rearms_after_it_clears(tmp_path):
    """Fixed config then broken again = a NEW incident, logged again — the
    dedup-key-never-cleared class this codebase forbids."""
    store = _intent(tmp_path)
    obs = FakeSlotObserver(_node_status(warning=_MISCONFIG_WARNING))
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=FakeObservers({"sparky": obs}))

    watcher.tick()                                     # incident 1
    obs._status = _node_status()                       # config fixed
    watcher.tick()
    obs._status = _node_status(warning=_MISCONFIG_WARNING)  # broken again
    watcher.tick()                                     # incident 2

    events = [e for e in tail_events(events_path)
              if e["kind"] == "lifecycle-node-misconfigured"]
    assert len(events) == 2


def test_unreachable_gap_does_not_relog_the_warning(tmp_path):
    """An unreachable blip is 'we failed to look', not 'we looked and the
    warning was gone' (the SparkObserver distinction): the cached
    unreachable status carries serving: None, which must NOT re-arm the
    memo — else a regularly powered-off node with a standing
    misconfiguration logs a fresh amber event every reachability cycle."""
    store = _intent(tmp_path)
    obs = FakeSlotObserver(_node_status(warning=_MISCONFIG_WARNING))
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=FakeObservers({"sparky": obs}))

    watcher.tick()                                     # incident logged
    obs._status = {"profile": None, "serving": None,   # observer's cached
                   "reachable": False,                 # _UNREACHABLE_SPARK
                   "swap_in_progress": False}
    watcher.tick()                                     # blip: failed to look
    obs._status = _node_status(warning=_MISCONFIG_WARNING)
    watcher.tick()                                     # same incident holds

    events = [e for e in tail_events(events_path)
              if e["kind"] == "lifecycle-node-misconfigured"]
    assert len(events) == 1


def test_no_warning_key_logs_nothing(tmp_path):
    """Pre-N1 agents don't send the field at all — additive, backward
    compatible: no key means nothing to surface, not an error."""
    store = _intent(tmp_path)
    obs = FakeSlotObserver(_node_status())
    watcher, events_path = _reconcile_watcher(
        tmp_path, store, node_observers=FakeObservers({"sparky": obs}))

    watcher.tick()

    assert not [e for e in tail_events(events_path)
                if e["kind"] == "lifecycle-node-misconfigured"]


# ===========================================================================
# CHARACTERISTICS DERIVE PASS (task 6) — throttled by settings.derive_interval_s
# ===========================================================================


def test_derive_pass_runs_on_first_tick(tmp_path, monkeypatch):
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       gguf_dir=_checkpoint_tree(tmp_path))

    watcher.tick()

    assert store.get(), "expected at least one derived entry"


def test_derive_pass_skips_dot_directories(tmp_path, monkeypatch):
    """A dot-directory in the gguf store (hf-transfer's `.cache` scratch,
    live-found 08-06) is tooling residue, not a checkpoint — but
    derive_checkpoint unconditionally emits an `identity` fact for any
    directory, so without a scan-side filter the store grows a phantom
    `model/.cache` key. The real checkpoint beside it must still derive."""
    gguf_dir = _checkpoint_tree(tmp_path)
    (gguf_dir / ".cache").mkdir()  # bare, like the live one — no config.json

    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       gguf_dir=gguf_dir)

    watcher.tick()

    keys = set(store.get())
    assert "model/some-model" in keys
    assert "model/.cache" not in keys


def test_derive_pass_is_throttled(tmp_path, monkeypatch):
    """The watcher ticks every 2 s. Scanning every checkpoint that often is
    pointless I/O, and the reason someone would turn this off."""
    calls = []
    watcher = _watcher(tmp_path=tmp_path, derive_interval_s=300.0,
                       on_derive=lambda: calls.append(1))

    watcher.tick()
    watcher.tick()
    watcher.tick()

    assert len(calls) == 1


def test_derive_pass_reruns_after_the_interval(tmp_path, monkeypatch):
    clock = _FakeClock()
    calls = []
    watcher = _watcher(tmp_path=tmp_path, derive_interval_s=300.0,
                       on_derive=lambda: calls.append(1), clock=clock)

    watcher.tick()
    clock.advance(301)
    watcher.tick()

    assert len(calls) == 2


def test_derive_failure_does_not_break_the_tick(tmp_path, monkeypatch):
    """Supervisor rule again: a bad checkpoint must not stop arbitration or
    reconciliation."""
    watcher = _watcher(tmp_path=tmp_path, on_derive=_raises(OSError("disk")))

    watcher.tick()   # must not raise


def test_restore_clears_derive_throttle_so_the_next_tick_derives(tmp_path):
    """A resource that just came back up has fresh live facts worth
    capturing — that must not wait up to derive_interval_s later. Tick 1
    matches intent (nothing to restore), so derive fires and arms the
    throttle. Tick 2 finds the resource down and restores it, well within
    the interval; the restore must clear the throttle so THIS tick's
    (already-scheduled) derive pass runs again instead of being skipped."""
    store = _intent(tmp_path)  # local/hipfire wants state=loaded
    hipfire = FakeHipfire(state="running")
    world = FakeWorld(_world(hipfire=_hip(state="running", model="gpt-oss")))
    calls = []
    watcher = _watcher(
        tmp_path, world=world, hipfire=hipfire, intent_store=store,
        on_derive=lambda: calls.append(1),
    )

    watcher.tick()  # matches intent -> no restore; first-ever derive fires
    assert len(calls) == 1

    # hipfire goes down -- well within derive_interval_s (default 300 s) of
    # the first derive, so an un-reset throttle would skip this tick's pass.
    world._snapshot = _world(hipfire=_hip(state="parked"))

    watcher.tick()  # restore -> must clear the throttle -> derives again
    assert "resume" in hipfire.calls
    assert len(calls) == 2

    # Control: nothing left to restore -> the throttle is back in force.
    world._snapshot = _world(hipfire=_hip(state="running", model="gpt-oss"))
    watcher.tick()
    assert len(calls) == 2


def test_restore_floor_limits_repeated_per_tick_derives(tmp_path):
    """A resource that keeps crash-looping can have _restore() succeed at the
    API level every ~2 s tick (e.g. resume() returns fine) without ever
    raising, so the failure-budget/quarantine machinery — which only trips
    on a raise — never engages. Before the Task 5 per-key restore cooldown,
    reconcile re-dispatched a restore every tick for as long as the observed
    world stayed down; now the cooldown itself bounds that to one real
    restore attempt per _RESTORE_COOLDOWN_S (both constants are 30 s), so a
    5-tick rapid-succession crash loop produces exactly ONE restore, not
    five. The first (only) restore of the incident still derives immediately
    (Tim's ruling) and, because no further restore fires within the window,
    the derive-restore floor never gets a second attempt to suppress either —
    it and the cooldown converge on the same one-extra-derive outcome here."""
    store = _intent(tmp_path)  # local/hipfire wants state=loaded
    hipfire = FakeHipfire(state="running")
    world = FakeWorld(_world(hipfire=_hip(state="running", model="gpt-oss")))
    calls = []
    watcher = _watcher(
        tmp_path, world=world, hipfire=hipfire, intent_store=store,
        on_derive=lambda: calls.append(1),
    )

    watcher.tick()  # healthy -> no restore; first-ever derive fires
    assert len(calls) == 1

    # Incident starts: hipfire goes down and NEVER comes back up in this
    # world's observation (the crash loop) for the next 5 ticks.
    world._snapshot = _world(hipfire=_hip(state="parked"))

    for _ in range(5):
        watcher.tick()

    assert hipfire.calls.count("resume") == 1   # cooldown: one attempt, not five
    assert len(calls) == 2                       # and still only ONE extra derive


def test_restore_triggered_derive_resumes_after_the_floor_elapses(tmp_path):
    """The floor rate-limits, it does not latch permanently: once it
    elapses, a restore (a later flap of the same resource, or a fresh
    incident) triggers an immediate derive again."""
    clock = _FakeClock()
    store = _intent(tmp_path)
    hipfire = FakeHipfire(state="running")
    world = FakeWorld(_world(hipfire=_hip(state="parked")))
    calls = []
    watcher = _watcher(
        tmp_path, world=world, hipfire=hipfire, intent_store=store,
        on_derive=lambda: calls.append(1), clock=clock,
    )

    watcher.tick()  # first restore of the incident -> derives
    assert len(calls) == 1

    watcher.tick()  # immediately again -> within the floor -> no derive
    assert len(calls) == 1

    clock.advance(31)  # past the 30 s floor
    watcher.tick()  # floor elapsed -> derives again
    assert len(calls) == 2


def test_derive_pass_survives_missing_gguf_dir(tmp_path):
    """A missing /gguf-store mount (host misconfigured, or a container
    started before the volume attached) must degrade to 'no local facts',
    not raise FileNotFoundError into tick()'s supervisor catch every
    settings.derive_interval_s. The gguf scan being skippable must not
    also skip the rest of the pass — live derivation still runs."""
    store = CharacteristicsStore(tmp_path / "c.json")
    spark = FakeSpark()  # models() -> {"data": []}, but proves it was called
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       gguf_dir=tmp_path / "does-not-exist", **_sparky_wiring(spark))

    watcher.tick()  # must not raise

    assert store.get() == {}  # nothing readable: no gguf mount, no live models
    assert spark.models_calls == 1  # live derivation still ran past the guard


def test_derive_pass_retains_last_known_facts_when_spark_unreachable(tmp_path):
    """Watcher-level counterpart to test_models_raises_engineerror_on_non_2xx
    (app.engines.spark): when spark.models() raises EngineError mid-tick, the
    derive pass must retain whatever facts are already on disk rather than
    losing them or crashing the tick."""
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("model/heretic", {
        "served": {"value": True, "source": "/v1/models", "derived_ts": "t0"}})
    spark = FakeSpark(models_raises=EngineError("connection refused"))
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, **_sparky_wiring(spark))

    watcher.tick()  # must not raise

    assert store.get()["model/heretic"]["served"]["value"] is True


def test_derive_pass_skips_spark_probe_while_observer_says_unreachable(tmp_path):
    """Mirrors the provenance-pass guard below: the observer's cached
    backoff exists precisely so a down node costs nothing between probes —
    the live-facts loop calling spark.models() directly regardless would
    defeat it every settings.derive_interval_s."""
    store = CharacteristicsStore(tmp_path / "c.json")
    spark = FakeSpark(raises=EngineError("connection refused"))
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, **_sparky_wiring(spark))

    watcher._node_observers.observer_for("sparky").status()  # seed the cache: unreachable
    watcher._derive_pass()

    assert spark.models_calls == 0


# ===========================================================================
# WATCHER RECORDS THE INTENT IT AUTHORS — 2026-08-06 reconciler-intent fix
# ===========================================================================


def test_idle_release_records_unloaded_intent_and_reconciler_stays_quiet(tmp_path):
    """Whoever actuates, records: an idle-release must record state=unloaded
    BEFORE unloading, so the very next reconcile pass derives 'parked', not
    'down' — the reconciler must not reload what the watcher just released
    (live defect 2026-08-06, events 21:37:38 -> 21:37:41)."""
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/lemonade", state="loaded", model="extra.m.gguf", engine="lemonade")

    lemonade = FakeLemonade()
    # lemonade loaded and idle past a 10 s TTL -> decide() emits unload_lemonade
    snapshot = _world(lemonade=_lem(state="loaded", model="extra.m.gguf", idle_s=1000))
    policy = _policy(lem_idle=10)
    watcher, _events = _make_watcher(
        tmp_path, FakeWorld(snapshot), FakeRegistry(), policy,
        lemonade=lemonade, intent_store=intent,
    )

    watcher.tick()

    assert lemonade.unloaded == ["extra.m.gguf"]
    assert intent.get()["local/lemonade"]["state"] == "unloaded"
    # The record must be stamped actor="deck": this is the arbiter's OWN
    # idle-release, and app.routers.control's pull-through supersession check
    # honors only OPERATOR-authored records — a deck unload mislabeled
    # "operator" silently drops an operator's in-flight pull-through load
    # [max-review Important-1, task 6]. Asserted here at the PRODUCER because
    # the supersession tests seed their records via IntentStore.record()
    # directly, so deleting actor="deck" from this call site left the whole
    # suite green. The seed above is operator-authored (record() defaults),
    # so this assertion sits away from the fixture's own value.
    assert intent.get()["local/lemonade"]["actor"] == "deck"

    # Next tick observes the unloaded engine; intent now says unloaded too,
    # so plan_reconcile derives 'parked' and must NOT restore.
    watcher._world = FakeWorld(_world(lemonade=_lem(state="unloaded")))
    watcher.tick()
    assert lemonade.loaded == []


def test_load_retriggered_records_loaded_intent(tmp_path):
    """The contention-heal reload is a deck-authored load and must record
    state=loaded (before this fix it recorded nothing, leaving intent stale).
    It must also load EXACTLY once: the same-tick reconcile pass runs on the
    pre-actuation snapshot, which still reads lemonade as unloaded — without
    dropping this tick's actuated key from that pass, 'down' derives and the
    reconciler restores (i.e. loads) it a second time, fighting its own
    watcher within one tick."""
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    # Mirrors test_watcher_tick_heals_contention_then_reloads's arrangement:
    # a pending default-route load healed by freeing comfy, then re-triggered.
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade, comfy = FakeLemonade(), FakeComfy()
    read_gpus = RecordingReadGpus(result=[{"probe": True}])
    watcher, _events_path = _make_watcher(
        tmp_path,
        FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(),
        lemonade=lemonade,
        comfy=comfy,
        read_gpus=read_gpus,
        intent_store=intent,
        auto=True,
    )

    watcher.tick()

    assert lemonade.loaded == ["extra.model.gguf"]  # exactly once, not twice
    record = intent.get()["local/lemonade"]
    assert record == {**record, "state": "loaded", "model": "extra.model.gguf"}
    # Producer-side pin, same reason as the idle-release arm's: this
    # contention-heal reload is deck-authored, and nothing else in the suite
    # catches this call site losing its actor="deck" [task 6 open finding].
    assert record["actor"] == "deck"


def test_same_tick_evict_and_reload_pins_single_load_and_final_intent(tmp_path):
    """The invariant class this whole wave guards: a SINGLE _execute() call
    that both evicts a currently-loaded non-default-route lemonade model
    (unload_lemonade, records state=unloaded) AND re-triggers the pending
    default-route load (records state=loaded) for the SAME intent key. The
    per-tick actuated-keys set must be idempotent (one key, added twice by
    the two arms), record() must be last-write-wins (final state='loaded',
    not the first arm's 'unloaded'), the key must be excluded from this same
    tick's reconcile pass, and only ONE lemonade.load() call may ever happen
    for it.

    NOTE on the exclusion specifically: unlike the pure load-retrigger case
    (test_load_retriggered_records_loaded_intent), reverting the
    observed.pop(actuated key) in _reconcile_pass does NOT make this test
    fail. The pre-action snapshot here already shows lemonade LOADED (with
    the model that is about to be evicted), so even without the pop,
    derive_status compares intent='loaded'/wanted=the new model against
    observed='loaded'/actual=the evicted model and derives 'drifted', not
    'down' — and only 'down' is ever actionable (app.reconcile.
    ACTIONABLE_STATUS). 'drifted' does not exist for the pure load-retrigger
    case (there, the pre-action snapshot shows NOTHING loaded, which derives
    the actionable 'down'). Verified by hand: temporarily commenting out the
    pop still passes this test. The pop remains correct and is still
    directly asserted below (the key must be entirely absent from this
    tick's derived statuses) — it just isn't the ONLY thing standing between
    this particular scenario and a double-load, whereas it is the only thing
    for the pure-reload case.

    Driven via _execute()/_reconcile_pass() directly, NOT watcher.tick():
    Watcher._infer_pending() only ever infers a pending load while lemonade
    is OBSERVED 'unloaded' (`if world["tenants"]["lemonade"]["state"] !=
    "unloaded": return None`) — by design, a single-slot engine can't be
    "pending" a different model while one is already resident. So the real
    tick() pipeline can never hand decide() a pending load while lemonade
    is loaded, and the lemonade-eviction branch of _decide_contention (see
    test_contention_evicts_lemonade_when_not_default_route) is unreachable
    through tick() alone. The pure decide()/_decide_contention() function
    has no such restriction — pending_load is just a parameter — so this
    test supplies one directly and calls _execute()/_reconcile_pass() in
    the exact sequence tick() uses, to exercise the real combined-actuation
    code path for this key."""
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    # Mirrors test_contention_evicts_lemonade_when_not_default_route (the
    # eviction) crossed with test_watcher_tick_heals_contention_then_reloads
    # (the reload): lemonade holds a non-default model, comfy is busy (not
    # evictable) so lemonade alone must cover the pending footprint.
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="loaded", model="extra.other.gguf", footprint=25 * GIB, idle_s=10),
        comfyui=_comfy(state="busy", queue=1, idle_s=0),  # not evictable
        default_route="extra.model.gguf",
    )
    pending = _pending()  # model="extra.model.gguf", footprint=19*GIB, gpu_index=1

    lemonade = FakeLemonade()
    watcher, _events = _make_watcher(
        tmp_path, FakeWorld(world), FakeRegistry(), _policy(),
        lemonade=lemonade, intent_store=intent, auto=True,
    )

    actions = decide(world, _policy(), pending)
    # Sanity: both arms are about to fire for the SAME key — the eviction
    # alone (25 GiB freed) covers the 19 GiB pending footprint given 5 GiB
    # already free, so comfy (busy anyway) is never a candidate.
    assert actions == [{"type": "unload", "resource": "lemonade", "model": "extra.other.gguf"}]

    actuated_keys = watcher._execute(actions, pending)
    assert actuated_keys == {"local/lemonade"}  # idempotent: one key, added by both arms
    watcher._reconcile_pass(world, actuated_keys)

    assert lemonade.unloaded == ["extra.other.gguf"]
    assert lemonade.loaded == ["extra.model.gguf"]  # exactly once — no reconcile double-load
    record = intent.get()["local/lemonade"]
    assert record["state"] == "loaded"  # last write wins, not the eviction arm's 'unloaded'
    assert record["model"] == "extra.model.gguf"



def test_reconciler_restores_a_key_at_most_once_per_cooldown(tmp_path):
    """An externally-dead engine with intent=loaded derives 'down' every
    tick; without a cooldown that is one restore attempt per 2 s tick.
    Bound it: one attempt per key per _RESTORE_COOLDOWN_S."""
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/lemonade", state="loaded", model="extra.m.gguf", engine="lemonade")

    clock = _FakeClock()
    lemonade = FakeLemonade()
    # world permanently shows nothing loaded (the fake load has no effect on
    # the snapshot) -> 'down' derives on every tick.
    watcher, _events = _make_watcher(
        tmp_path, FakeWorld(_world(lemonade=_lem(state="unloaded"))),
        FakeRegistry(), _policy(), lemonade=lemonade, intent_store=intent,
        clock=clock,
    )

    watcher.tick()
    watcher.tick()
    assert lemonade.loaded == ["extra.m.gguf"]  # second tick suppressed

    clock.advance(31)
    watcher.tick()
    assert lemonade.loaded == ["extra.m.gguf", "extra.m.gguf"]  # cooldown expired


def test_reconciler_skips_a_lemonade_restore_while_a_deck_load_is_in_flight(tmp_path):
    """The race this guards (whole-branch review, finding 1): a ROUTER-
    authored load records intent=loaded and starts the engine load while a
    watcher tick is already in flight. That tick's world snapshot predates
    the load (lemonade observed unloaded at snapshot time), so its fresh
    intent read still derives 'down' and would restore — i.e. load the SAME
    model a second time from the watcher thread, racing the deck's own
    in-flight load. FakeLemonade.load_in_flight()=True stands in for that
    live in-flight load; the world is arranged exactly like
    test_reconciler_restores_a_key_at_most_once_per_cooldown (permanently
    unloaded, so 'down' derives every tick) to isolate this guard from the
    cooldown."""
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/lemonade", state="loaded", model="extra.m.gguf", engine="lemonade")

    lemonade = FakeLemonade(in_flight=True)
    watcher, _events = _make_watcher(
        tmp_path, FakeWorld(_world(lemonade=_lem(state="unloaded"))),
        FakeRegistry(), _policy(), lemonade=lemonade, intent_store=intent,
    )

    watcher.tick()

    assert lemonade.loaded == []  # no second, watcher-thread load fired
    # A skipped restore is a non-action (no event, no failure charged) — it
    # must not consume the per-key cooldown slot either, so the key
    # reconciles cleanly the moment the in-flight load clears.
    assert "local/lemonade" not in watcher._restore_last_attempt_at

    # Control: once the deck-authored load is no longer in flight, the exact
    # same 'down' derivation DOES restore — the guard is per-tick (reads
    # load_in_flight() fresh), not a sticky suppression.
    lemonade._in_flight = False
    watcher.tick()

    assert lemonade.loaded == ["extra.m.gguf"]


# ---------------------------------------------------------------------------
# RESTORE-WITHOUT-HEALTH charges the failure budget (2026-08-13 incident
# follow-up, Part A). A restore that dispatches fine but never produces a
# `serving` tick used to charge nothing — the budget only tripped when
# _restore() RAISED — so an unhealthy placement restored forever (live-proven
# 08-12: 7 restores at 20-min boot-window intervals, no quarantine). The
# evidence that restore N failed is the reconciler being about to dispatch
# restore N+1 for the same key with no `serving` tick in between.
# ---------------------------------------------------------------------------


def _unverified_setup(tmp_path):
    """A lemonade key whose restores dispatch fine (no raise) while the world
    permanently shows it unloaded -> 'down' derives every tick, `serving`
    never arrives. The exact incident shape Part A exists to bound."""
    intent = _intent(tmp_path, key="local/lemonade", model="extra.m.gguf",
                     engine="lemonade")
    clock = _FakeClock()
    lemonade = FakeLemonade()
    world = FakeWorld(_world(lemonade=_lem(state="unloaded")))
    watcher, events_path = _make_watcher(
        tmp_path, world, FakeRegistry(), _policy(),
        lemonade=lemonade, intent_store=intent, clock=clock,
    )
    return watcher, events_path, intent, lemonade, clock, world


def test_unverified_restore_charges_the_budget_on_redispatch(tmp_path):
    """First unverified redispatch: exactly ONE failure charged (count
    asserted, not just a flag), the event reuses the existing
    lifecycle-restore-failed kind, and the second restore still dispatches
    (budget not yet exhausted)."""
    watcher, events_path, intent, lemonade, clock, _ = _unverified_setup(tmp_path)

    watcher.tick()                        # restore #1 dispatches, no raise
    assert lemonade.loaded == ["extra.m.gguf"]
    assert intent.get()["local/lemonade"]["failures"] == 0

    clock.advance(31)                     # past _RESTORE_COOLDOWN_S
    watcher.tick()                        # still down -> charge, restore #2

    assert intent.get()["local/lemonade"]["failures"] == 1
    assert lemonade.loaded == ["extra.m.gguf", "extra.m.gguf"]
    failed = [e for e in tail_events(events_path)
              if e["kind"] == "lifecycle-restore-failed"]
    assert failed
    assert failed[-1]["detail"] == {
        "key": "local/lemonade",
        "error": "previous restore never became healthy",
        "failures": 1,
    }


def test_second_unverified_restore_quarantines_and_skips_dispatch(tmp_path):
    """Budget exhausted: the third would-be dispatch charges failure #2,
    quarantines, logs, and does NOT dispatch (the engine fake records no
    third call). Incident replay: quarantine after 2 restores, not 7+."""
    watcher, events_path, intent, lemonade, clock, _ = _unverified_setup(tmp_path)

    watcher.tick()                        # restore #1, marker set
    clock.advance(31)
    watcher.tick()                        # charge 1, restore #2, marker re-set
    clock.advance(31)
    watcher.tick()                        # charge 2 -> quarantined, NO dispatch

    record = intent.get()["local/lemonade"]
    assert record["failures"] == 2
    assert record["quarantined"] is True
    assert lemonade.loaded == ["extra.m.gguf", "extra.m.gguf"]
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-quarantined" in kinds


def test_serving_tick_clears_the_unverified_marker(tmp_path):
    """A restore that reaches `serving` is verified: the marker clears next
    to note_healthy, so a LATER genuine outage starts clean — its first
    restore charges nothing and logs no failure."""
    watcher, events_path, intent, lemonade, clock, world = _unverified_setup(tmp_path)

    watcher.tick()                        # restore #1, marker set
    world._snapshot = _world(lemonade=_lem(state="loaded", model="extra.m.gguf"))
    clock.advance(31)
    watcher.tick()                        # serving: marker cleared
    assert intent.get()["local/lemonade"]["last_healthy_ts"] is not None

    world._snapshot = _world(lemonade=_lem(state="unloaded"))
    clock.advance(31)
    watcher.tick()                        # new incident's FIRST restore

    assert lemonade.loaded == ["extra.m.gguf", "extra.m.gguf"]
    assert intent.get()["local/lemonade"]["failures"] == 0
    assert not [e for e in tail_events(events_path)
                if e["kind"] == "lifecycle-restore-failed"]


def test_cooldown_skip_does_not_charge_the_budget(tmp_path):
    """The 30 s cooldown skip stays a non-action: no event, no failure
    charged, marker untouched."""
    watcher, events_path, intent, lemonade, clock, _ = _unverified_setup(tmp_path)

    watcher.tick()                        # restore #1 at t=0
    clock.advance(10)                     # inside _RESTORE_COOLDOWN_S
    watcher.tick()                        # cooldown skip

    assert intent.get()["local/lemonade"]["failures"] == 0
    assert lemonade.loaded == ["extra.m.gguf"]
    assert not [e for e in tail_events(events_path)
                if e["kind"] == "lifecycle-restore-failed"]


def test_load_in_flight_skip_does_not_charge_the_budget(tmp_path):
    """The lemonade load_in_flight skip stays a non-action too — but it only
    DEFERS the reckoning: once the in-flight load clears and the key is
    still down, the charge fires on the real dispatch."""
    watcher, events_path, intent, lemonade, clock, _ = _unverified_setup(tmp_path)

    watcher.tick()                        # restore #1, marker set
    lemonade._in_flight = True
    clock.advance(31)
    watcher.tick()                        # in-flight skip: no charge

    assert intent.get()["local/lemonade"]["failures"] == 0

    lemonade._in_flight = False
    watcher.tick()                        # real dispatch -> charge fires

    assert intent.get()["local/lemonade"]["failures"] == 1


def test_fresh_watcher_does_not_charge_for_a_pre_restart_restore(tmp_path):
    """Documents the in-memory choice: a deck restart forgets at most one
    pending charge. A fresh watcher over a stale loaded-intent dispatches
    its first restore without charging — `failures` seeded away from the
    0 default so 'unchanged' is provable, not coincidental."""
    intent = _intent(tmp_path, key="local/lemonade", model="extra.m.gguf",
                     engine="lemonade")
    intent.note_failure("local/lemonade")  # pre-restart history: failures=1
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world(lemonade=_lem(state="unloaded"))),
        FakeRegistry(), _policy(), lemonade=lemonade, intent_store=intent,
    )

    watcher.tick()                        # first restore after "restart"

    assert lemonade.loaded == ["extra.m.gguf"]
    assert intent.get()["local/lemonade"]["failures"] == 1  # unchanged
    assert not [e for e in tail_events(events_path)
                if e["kind"] == "lifecycle-restore-failed"]


# ===========================================================================
# CATALOG HARVEST (task 8) — _derive_pass harvests each configurable
# engine's option catalog, once per observed engine version.
#
# Watcher._configurable_engines returns its constructor pairs VERBATIM as of
# task 8 (C2) — no internal pairing with _LOCAL_NODE happens inside Watcher
# anymore; app.main builds one real production pair per control:"swap"
# registry node (N1), each `(that node's id, "vllm")`. Every test below
# (except the C2 remote-pair tests at the end of this section) injects
# `configurable_engines=[(_LOCAL_NODE, "hipfire")]` purely to keep exercising
# _harvest_catalogs's machinery against a local engine_exec double; it is not
# a claim that hipfire is a real harvest target (F2, 2026-08-07: hipfire is
# confirmed not vLLM-backed, so C1/C2 both have no valid LOCAL harvest
# target — see Watcher._configurable_engines' docstring in app/arbiter.py).
# ===========================================================================


def _recording_exec(calls, version):
    """A fake ``engine_exec``: a callable with a mutable ``.version`` that
    records every ``(node, engine, interpreter, source)`` call it receives
    and returns ``(self.version, PROBE_OUTPUT)`` — mutable so a test can
    simulate an engine upgrade between derive passes."""

    class _Exec:
        def __init__(self) -> None:
            self.version = version

        def __call__(self, node, engine, interpreter, source):
            calls.append((node, engine, interpreter, source))
            return self.version, PROBE_OUTPUT

    return _Exec()


def test_harvest_runs_once_and_caches_by_version(tmp_path, monkeypatch):
    store = CharacteristicsStore(tmp_path / "c.json")
    execs = []
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=_recording_exec(execs, version="0.26.0"),
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(execs) == 1
    assert store.entry(f"engine/{_LOCAL_NODE}/hipfire")["option_catalog"]["value"]["engine_version"] == "0.26.0"


def test_harvest_reruns_when_the_engine_version_changes(tmp_path, monkeypatch):
    store = CharacteristicsStore(tmp_path / "c.json")
    execs = []
    exec_fn = _recording_exec(execs, version="0.26.0")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, engine_exec=exec_fn,
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()
    exec_fn.version = "0.27.0"
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(execs) == 2


def test_harvest_failure_leaves_no_catalog_and_does_not_raise(tmp_path, monkeypatch):
    """An engine that is down has no catalog. Supported state, not an error."""
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=_raises(EngineError("not running")),
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()

    assert "option_catalog" not in store.entry(f"engine/{_LOCAL_NODE}/hipfire")


# --- F3, Important (final branch review, 2026-08-07) -------------------------
#
# The exec-failure branch above used to be fully silent -- indistinguishable
# from "engine parked" forever, and the branch's own recorded deploy hazard
# (a hand-merged live compose losing the proxy's -allowPOST lines -> 403 on
# exec) lands exactly there. It now logs a deduped harvest-failed event.
#
# That deploy hazard no longer applies to the LOCAL docker path: task 11
# removed the socket-proxy exec rules outright, because C2 left them with
# zero callers (harvest reads the node-agent's file cache now). The branch
# stays live and worth this test -- engine_exec is an injection seam, and
# the exec that still runs through it is SparkCatalogExec's remote
# node-agent call, whose failures are network/agent-side rather than
# proxy-allowlist-side.


def test_harvest_exec_failure_logs_a_deduped_harvest_failed_event(tmp_path):
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        characteristics_store=store,
        engine_exec=_raises(EngineError("403 Forbidden")),
        configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    failures = [e for e in tail_events(events_path) if e["kind"] == "harvest-failed"]
    assert len(failures) == 1
    assert failures[0]["detail"] == {"key": f"engine/{_LOCAL_NODE}/hipfire", "reason": "EngineError"}


def test_harvest_failed_dedup_resets_when_the_failure_kind_changes(tmp_path):
    """A different failure in between resets the suppression -- same dedup
    semantics as every other _DEDUP_KINDS event (see the tick-error dedup
    test above)."""
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        characteristics_store=store,
        engine_exec=_raises(EngineError("403 Forbidden")),
        configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()  # EngineError -> logged
    watcher._last_derive_at = None
    watcher._derive_pass()  # EngineError again -> deduped

    watcher._engine_exec = _raises(
        GuardError("container 'ods-hipfire' is not in the park allowlist"))
    watcher._last_derive_at = None
    watcher._derive_pass()  # GuardError -> resets + logged

    failures = [e for e in tail_events(events_path) if e["kind"] == "harvest-failed"]
    assert [f["detail"]["reason"] for f in failures] == ["EngineError", "GuardError"]


# --- fix round 1 (2026-08-07) ------------------------------------------------


def test_harvest_survives_guarderror_when_the_engine_is_not_allowlisted(tmp_path):
    """GuardError (a non-allowlisted container) is deliberately NOT an
    EngineError subclass (app.engines' module docstring) -- an operator who
    narrows park_allowlist must not turn every derive pass into a raised
    GuardError. Same supported-no-catalog posture as an EngineError/
    BusyError failure."""
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=_raises(GuardError("container 'ods-hipfire' is not in the park allowlist")),
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()  # must not raise

    assert "option_catalog" not in store.entry(f"engine/{_LOCAL_NODE}/hipfire")


def test_harvest_bare_engine_exec_without_version_peek_still_writes_once(tmp_path):
    """`engine_exec` without a `.version` attribute (e.g. a bare function,
    as opposed to `_recording_exec`'s object) has no cheap peek to skip the
    exec on, so the exec runs on every non-throttled pass -- but the
    post-call version comparison still stops a redundant WRITE when the
    freshly observed version matches what's already cached. Covers the
    fallback-compare branch the peek-based tests above don't exercise."""
    store = CharacteristicsStore(tmp_path / "c.json")
    writes = []
    real_put_fields = store.put_fields

    def _counting_put_fields(key, fields):
        writes.append((key, fields))
        return real_put_fields(key, fields)

    store.put_fields = _counting_put_fields

    calls = []

    def bare_exec(node, engine, interpreter, source):
        calls.append(1)
        return "0.26.0", PROBE_OUTPUT

    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, engine_exec=bare_exec,
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(calls) == 2   # no peek available -> the exec runs every pass
    assert len(writes) == 1  # ...but the post-call compare stops the 2nd write


def test_harvest_reruns_against_the_real_dockerengineexec_when_the_image_id_changes(tmp_path):
    """The dictated re-harvest test above (test_harvest_reruns_when_the_
    engine_version_changes) passes only because its double's `.version` is
    hand-mutated by the test -- it cannot catch a real bug like keying
    "version" on the wrong Docker inspect field (Config.Image, a floating
    tag that stays IDENTICAL across an image rebuild + container recreate,
    instead of the resolved Image content ID -- see
    app.engines.docker_ctl.DockerCtl.image_ref's docstring for the exact
    failure this was). This wires the REAL DockerCtl + DockerEngineExec
    through a mocked Docker Engine API and proves the property against
    them, not a hand-rolled double."""
    import httpx

    from app.engines.docker_ctl import DockerCtl, DockerEngineExec

    image_id = {"value": "sha256:old"}
    exec_creates = []

    def handler(request):
        if request.url.path == "/containers/ods-hipfire/json":
            return httpx.Response(200, json={"Image": image_id["value"]}, request=request)
        if request.url.path == "/containers/ods-hipfire/exec":
            exec_creates.append(1)
            return httpx.Response(201, json={"Id": "e1"}, request=request)
        if request.url.path == "/exec/e1/start":
            frame = (bytes([1, 0, 0, 0]) + len(PROBE_OUTPUT).to_bytes(4, "big")
                     + PROBE_OUTPUT.encode())
            return httpx.Response(200, content=frame, request=request)
        raise AssertionError(f"unexpected path {request.url.path!r}")

    dockerctl = DockerCtl("http://docker-ctl:2375", ["ods-hipfire"],
                           transport=httpx.MockTransport(handler))
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=DockerEngineExec(dockerctl, "ods-hipfire"),
                       configurable_engines=[(_LOCAL_NODE, "hipfire")])

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(exec_creates) == 1  # unchanged image id -> peek skips the 2nd exec

    image_id["value"] = "sha256:new"
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(exec_creates) == 2  # changed image id -> re-harvested
    assert (store.entry(f"engine/{_LOCAL_NODE}/hipfire")["option_catalog"]["value"]["engine_version"]
            == "sha256:new")


# --- F1/F2, live-deploy fix round (2026-08-07) -------------------------------
#
# The live deploy's one real harvest pass wrote "engine/autarch/hipfire" —
# `_configurable_engines` used to pair "hipfire" with `settings.node_label`
# (the display label, "autarch" via MODEL_DECK_NODE_LABEL) instead of the
# node id every reader keys on (`app.observe._LOCAL_NODE`, "local",
# app/observe.py:28; see GET /api/settings/catalog/{node}/{engine}, app/
# routers/settings.py:78-82). Unit tests never caught it because
# node_label's default ("local") happens to equal the node id, making
# label == id in every test that never overrode node_label.
#
# Task 8 (C2) moved pair CONSTRUCTION out of Watcher entirely --
# _configurable_engines returns whatever pair it's handed, verbatim, so
# there is no internal "pair with X" step left for node_label to leak into.
# That half of the original bug is now covered where the construction
# actually happens: tests.test_engines.
# test_build_watcher_routes_spark_catalog_by_node_id_not_label (app.main's
# real wiring, asserting it uses the registry node's id, never node_label).
# This test keeps its ORIGINAL regression duty at the Watcher layer: even with
# node_label set away from "local" and handed a real node-id pair, the
# harvest path must consume that pair as given and never consult
# settings.node_label at all -- a future Watcher change that reintroduced a
# node_label fallback (e.g. "use the pair's node, falling back to
# node_label") would still be caught here.


def test_harvest_key_uses_node_id_not_node_label(tmp_path):
    """Regression for the live-deploy defect above, at the Watcher layer:
    even with settings.node_label set away from its "local" default, a
    pair built from the node id must land the catalog under that node id
    (`engine/local/hipfire`, what the settings API actually reads) --
    never under the label (`engine/autarch/hipfire`, a key no API path
    reads) -- proving Watcher's harvest path never substitutes node_label
    for the pair it was actually given."""
    store = CharacteristicsStore(tmp_path / "c.json")
    execs = []
    watcher = _watcher(
        tmp_path=tmp_path, characteristics_store=store,
        engine_exec=_recording_exec(execs, version="0.26.0"),
        configurable_engines=[(_LOCAL_NODE, "hipfire")],
        node_label="autarch",
    )

    watcher._derive_pass()

    assert store.entry(f"engine/{_LOCAL_NODE}/hipfire")["option_catalog"] is not None
    assert store.entry("engine/autarch/hipfire") == {}


# --- C2: remote (node, engine) pairs, returned verbatim (task 8) ------------
#
# _configurable_engines() no longer pairs anything with _LOCAL_NODE itself —
# it returns whatever the constructor was handed, verbatim (see its
# docstring). These two exercise that seam with a pair a real production
# wiring would build (`(<swap node's registry id>, "vllm")`, app.main), and
# — unlike every test above, which reuses tests.test_harvest's PROBE_OUTPUT
# fixture —
# build the probe text by hand from app.harvest's REAL `_SENTINEL` wrapped
# around a real one-option payload, so `parse_probe_output` does the actual
# parsing. A hand-seeded `option_catalog` dict would exercise none of that
# and could hide exactly the kind of parsing regression C1's final branch
# review caught (see _harvest_catalogs' and parse_probe_output's docstrings).


def _sentinel_probe_output():
    """One-option probe text, real-shaped: _SENTINEL-bracketed JSON exactly
    as PROBE_SOURCE prints it (app/harvest.py), so parse_probe_output parses
    it for real rather than a fixture standing in for that parse."""
    payload = json.dumps({"options": [
        {"flags": ["--max-model-len"], "type": "int", "choices": None,
         "default": "None", "nargs": None, "cls": "_StoreAction",
         "help": "Model context length."},
    ]})
    return f"{_SENTINEL}\n{payload}\n{_SENTINEL}\n"


def test_watcher_harvests_a_remote_pair_end_to_end(tmp_path):
    """Watcher with configurable_engines=[('sparky', 'vllm')] and an
    engine_exec returning (image_id, REAL-SHAPED probe output) stores
    option_catalog under key 'engine/sparky/vllm' — build the probe output
    with app.harvest's real sentinel + a one-option payload so
    parse_probe_output does the parsing (no hand-seeded shapes; that
    masking bug is exactly what C1's final review caught)."""
    store = CharacteristicsStore(tmp_path / "c.json")
    probe_output = _sentinel_probe_output()

    def exec_fn(node, engine, interpreter, source):
        return "sha256:remote", probe_output

    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=exec_fn,
                       configurable_engines=[("sparky", "vllm")])

    watcher._derive_pass()

    catalog = store.entry("engine/sparky/vllm")["option_catalog"]
    assert catalog["value"]["engine_version"] == "sha256:remote"
    assert "max-model-len" in catalog["value"]["options"]
    # The pair form is consumed verbatim -- no _LOCAL_NODE re-pairing landed
    # a copy under the local-node key too.
    assert store.entry(f"engine/{_LOCAL_NODE}/vllm") == {}


def test_watcher_version_skip_prevents_refetch_after_store(tmp_path):
    """Second derive pass with the same image_id -> exec called, put_fields
    NOT called again (arbiter.py:934-935). engine_exec here is a bare
    function with no `.version` peek -- SparkCatalogExec's real shape (see
    its docstring: the node-agent has no cheap peek, so the exec runs every
    non-throttled pass) -- so this exercises the POST-call version compare,
    not the peek-based early skip test_harvest_runs_once_and_caches_by_
    version already covers."""
    store = CharacteristicsStore(tmp_path / "c.json")
    writes = []
    real_put_fields = store.put_fields

    def _counting_put_fields(key, fields):
        writes.append((key, fields))
        return real_put_fields(key, fields)

    store.put_fields = _counting_put_fields

    probe_output = _sentinel_probe_output()
    calls = []

    def exec_fn(node, engine, interpreter, source):
        calls.append(1)
        return "sha256:remote", probe_output

    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, engine_exec=exec_fn,
                       configurable_engines=[("sparky", "vllm")])

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(calls) == 2   # no .version peek -> the exec runs every pass
    assert len(writes) == 1  # ...but the post-call compare stops the 2nd write


def test_harvest_skips_the_spark_pair_while_observer_says_unreachable(tmp_path):
    """The spark pair's harvest_catalog_pair routes to
    SparkCatalogExec.__call__ -> SparkClient.get_catalog() — the identical
    node-agent GET _provenance_nodes's guard already avoids. A derive pass
    must not block on it (or run engine_exec at all for that pair) while
    sparky is down; a non-spark pair would still be unaffected (none is
    exercised here, since production's only remote pair IS spark's --
    see _configurable_engines' docstring)."""
    store = CharacteristicsStore(tmp_path / "c.json")
    spark = FakeSpark(raises=EngineError("connection refused"))
    calls = []

    def exec_fn(node, engine, interpreter, source):
        calls.append((node, engine))
        return "sha256:remote", _sentinel_probe_output()

    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=exec_fn,
                       configurable_engines=[("sparky", "vllm")],
                       **_sparky_wiring(spark))

    watcher._node_observers.observer_for("sparky").status()  # seed the cache: unreachable
    watcher._derive_pass()

    assert calls == []


# --- provenance pass -------------------------------------------------------

class _FakeDockerCtl:
    """Just the one read the provenance pass makes."""

    def __init__(self, bodies):
        self.bodies = dict(bodies)
        self.inspected = []

    def inspect(self, name):
        self.inspected.append(name)
        try:
            return self.bodies[name]
        except KeyError:
            raise EngineError(f"no such container {name}") from None


class _FakeCatalogUnits:
    def __init__(self, units):
        self._units = units

    def units(self):
        return list(self._units)


def _prov_store(tmp_path):
    from app.provenance import ProvenanceStore

    return ProvenanceStore(tmp_path / "provenance.json", tmp_path / "prov-history.jsonl")


def _one_container(tmp_path, bodies=None, **kwargs):
    """A watcher whose only provenance source is one local container."""
    store = _prov_store(tmp_path)
    docker = _FakeDockerCtl(bodies if bodies is not None else {
        "ods-hipfire": {"Image": "sha256:a", "Config": {"Image": "ods-hipfire:latest"}}})
    watcher = _watcher(tmp_path, provenance_store=store, dockerctl=docker,
                       park_allowlist=["ods-hipfire"], **kwargs)
    return watcher, store, docker


def test_provenance_pass_records_local_containers(tmp_path):
    watcher, store, _docker = _one_container(tmp_path)

    watcher.tick()

    entry = store.entry("oci:local:ods-hipfire")
    assert entry["current"]["version"] == "sha256:a"
    assert entry["current"]["label"] == "ods-hipfire:latest"
    assert entry["role"] == "engine"


def test_provenance_pass_marks_an_unreadable_container_unavailable_not_gone(tmp_path):
    # app.catalog's retention rule: a source going away must not erase what
    # was already recorded about it.
    clock = _FakeClock()
    watcher, store, docker = _one_container(tmp_path, clock=clock)
    watcher.tick()

    docker.bodies.clear()
    clock.advance(10_000)
    watcher.tick()

    entry = store.entry("oci:local:ods-hipfire")
    assert entry["current"]["version"] == "sha256:a"
    assert entry["current"]["verification"] == "unavailable"


def test_provenance_pass_is_throttled(tmp_path):
    watcher, _store, docker = _one_container(tmp_path, clock=_FakeClock())

    watcher.tick()
    watcher.tick()

    assert docker.inspected == ["ods-hipfire"]   # the second tick did no I/O


def test_provenance_pass_runs_again_once_the_interval_elapses(tmp_path):
    clock = _FakeClock()
    watcher, _store, docker = _one_container(tmp_path, clock=clock)

    watcher.tick()
    clock.advance(10_000)
    watcher.tick()

    assert docker.inspected == ["ods-hipfire", "ods-hipfire"]


def test_provenance_pass_is_absent_without_a_store(tmp_path):
    watcher = _watcher(tmp_path, dockerctl=_FakeDockerCtl({}))

    watcher.tick()   # must not raise

    assert not (tmp_path / "provenance.json").exists()


def test_provenance_pass_records_catalog_units_as_weights(tmp_path):
    store = _prov_store(tmp_path)
    catalog = _FakeCatalogUnits([{
        "id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
        "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}])
    watcher = _watcher(tmp_path, provenance_store=store, catalog=catalog)

    watcher.tick()

    entry = store.entry("file:local:m.gguf")
    assert entry["role"] == "weights"
    assert entry["current"]["verification"] == "consistent"


# --- node vocabulary, provenance edition (live-deploy defect, 2026-08-08) ---
#
# The SAME defect the harvest path shipped on 2026-08-07 (see the F1/F2
# block above), in a module written after that warning: the local collectors
# keyed artifact ids on `settings.node_label` -- the display string, e.g.
# "autarch" via MODEL_DECK_NODE_LABEL, documented as presentation-only at
# app/settings.py:31-34 and split from the id at app/routers/status.py:27
# (`{"id": "local", "label": node_label}`) -- instead of the node id every
# other node-scoped key in the deck uses (`app.observe._LOCAL_NODE`).
#
# Sparky's collector got it right (keyed by the registry node id, never a
# label -- since generalized to every swap node in
# app.arbiter.Watcher._provenance_nodes), so the ledger keyed local
# artifacts by label and remote ones by id: one namespace, two
# vocabularies. Nothing refused it because
# `origins._NODE_RE` validates the node as a slug, and "autarch" is a
# perfectly good slug.
#
# Every unit test missed it for the documented reason -- node_label DEFAULTS
# to "local", making label == id -- so these two pin the distinction by
# setting node_label away from its default. Only the live box, where the
# label really is "autarch", could tell the difference before this.


def test_provenance_pass_keys_containers_on_node_id_not_node_label(tmp_path):
    """Regression: with node_label set away from "local", a local engine
    must still land under the node id."""
    watcher, store, _docker = _one_container(tmp_path, node_label="autarch")

    watcher.tick()

    assert store.entry(f"oci:{_LOCAL_NODE}:ods-hipfire") is not None
    assert store.entry("oci:autarch:ods-hipfire") is None, (
        "artifact keyed on the display label -- a rename of "
        "MODEL_DECK_NODE_LABEL would orphan every declared origin and its "
        "history, in the one file that is their only home (D13)")


def test_provenance_pass_keys_weights_on_node_id_not_node_label(tmp_path):
    """Same vocabulary, the other local collector."""
    store = _prov_store(tmp_path)
    catalog = _FakeCatalogUnits([{
        "id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
        "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}])
    watcher = _watcher(tmp_path, provenance_store=store, catalog=catalog,
                       node_label="autarch")

    watcher.tick()

    assert store.entry(f"file:{_LOCAL_NODE}:m.gguf") is not None
    assert store.entry("file:autarch:m.gguf") is None


def test_a_provenance_failure_is_logged_not_raised(tmp_path):
    """A collector raising must be caught inside the pass. If it escaped to
    tick()'s supervisor catch it would look identical in the log while
    silently being the LAST thing in the tick — which is fine today only
    because provenance runs last. Catching it here keeps that an accident
    rather than a dependency."""
    class _Exploding:
        def units(self):
            raise RuntimeError("catalog on fire")

    store = _prov_store(tmp_path)
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        provenance_store=store, catalog=_Exploding())

    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path, 50)]
    assert "provenance-pass-error" in kinds
    assert "tick-error" not in kinds


def test_provenance_marks_unavailable_by_artifact_id_not_container_name(tmp_path):
    """The retention path must target the id the entry was CREATED with.

    Identity is now the image repository, so the container name is no longer
    that id — `ods-comfyui` runs `ignatberesnev/comfyui-gfx1151`, and the two
    differ. Every pre-existing fixture used `ods-hipfire`, whose name and
    repository happen to be identical, which is exactly why the suite stayed
    green while this path marked an id that does not exist: a silent no-op
    leaving the artifact reporting `exact` forever with its container gone.
    """
    clock = _FakeClock()
    store = _prov_store(tmp_path)
    docker = _FakeDockerCtl({"ods-comfyui": {
        "Image": "sha256:a",
        "Config": {"Image": "ignatberesnev/comfyui-gfx1151:v0.2"}}})
    watcher = _watcher(tmp_path, provenance_store=store, dockerctl=docker,
                       park_allowlist=["ods-comfyui"], clock=clock)
    watcher.tick()
    assert store.entry("oci:local:ignatberesnev/comfyui-gfx1151") is not None

    docker.bodies.clear()
    clock.advance(10_000)
    watcher.tick()

    entry = store.entry("oci:local:ignatberesnev/comfyui-gfx1151")
    assert entry["current"]["verification"] == "unavailable"
    assert entry["current"]["version"] == "sha256:a", (
        "an unreachable source must RETAIN its last known version, not blank it")


def test_provenance_unavailable_marking_uses_the_same_id_the_entry_was_created_with(tmp_path):
    """An id assembled two different ways in one pass is an id that will
    eventually disagree with itself — and a mismatch here would silently
    no-op instead of retaining the last known version."""
    clock = _FakeClock()
    watcher, store, docker = _one_container(tmp_path, clock=clock)
    watcher.tick()
    created = set(store.get())

    docker.bodies.clear()
    clock.advance(10_000)
    watcher.tick()

    assert set(store.get()) == created
    assert store.entry(next(iter(created)))["current"]["verification"] == "unavailable"


# --- provenance pass: watch seeding (Task 9) --------------------------------
#
# Wiring for provenance_collect.merge_seeded_watch into _provenance_pass.
# The brief only says, in prose, to call set_watch "where _provenance_pass
# writes each oci artifact" -- these tests pin the two things prose left
# open: the entry merge_seeded_watch needs is the STORED entry (with
# `origin`/`watch`, neither of which a freshly-observed collector entry
# carries), fetched via store.entry() AFTER observe(), and the call must not
# grow the history file on a pass where nothing changed.

def test_provenance_pass_seeds_a_watch_from_a_derivable_declared_origin(tmp_path):
    """An operator declares a digest-pinned origin (PUT /origin, simulated
    here as a direct store call); the very next collector pass must add the
    channel watch without the operator also typing PUT /watch.

    ``ods-hipfire`` observes as repository ``aeon-7/ods-hipfire`` here (a
    registry-qualified reference, unlike ``_one_container``'s bareword
    default) -- identity is the repository (see this file's THE ARTIFACT IS
    THE IMAGE section), so the artifact id follows the reference, not the
    container name.
    """
    reference = "ghcr.io/aeon-7/ods-hipfire:slim@sha256:deadbeef"
    watcher, store, _docker = _one_container(tmp_path, bodies={
        "ods-hipfire": {"Image": "sha256:a", "Config": {"Image": reference}}})
    store.declare_origin(
        "oci:local:aeon-7/ods-hipfire", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "aeon-7/ods-hipfire",
                "reference": reference})

    watcher.tick()

    watch = store.entry("oci:local:aeon-7/ods-hipfire")["watch"]
    assert len(watch) == 1
    assert watch[0] == {"id": "channel", "check": "oci_channel", "derived": True,
                        "label": "slim", "registry": "ghcr.io",
                        "repository": "aeon-7/ods-hipfire", "reference": "slim",
                        "pinned": "sha256:deadbeef", "order": None}


def test_provenance_pass_derives_no_watch_without_a_declared_origin(tmp_path):
    """The ordinary case (no PUT /origin yet, D8's gap): nothing to derive
    from, so watch stays empty -- never a guess from the observed label."""
    watcher, store, _docker = _one_container(tmp_path)
    watcher.tick()
    assert store.entry("oci:local:ods-hipfire")["watch"] == []


def test_provenance_pass_leaves_a_declared_watch_source_alone(tmp_path):
    """declared-over-derived through the real collector pass: a hand-put
    source must survive repeated ticks even though the origin is derivable.

    It does NOT suppress the derived source beside it -- the two answer
    different questions (a git tag set vs. the digest a channel resolves to)
    and the live seed data pairs exactly that way on aeon-vllm."""
    reference = "ghcr.io/aeon-7/ods-hipfire:slim@sha256:deadbeef"
    watcher, store, _docker = _one_container(tmp_path, bodies={
        "ods-hipfire": {"Image": "sha256:a", "Config": {"Image": reference}}})
    store.declare_origin(
        "oci:local:aeon-7/ods-hipfire", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "aeon-7/ods-hipfire",
                "reference": reference})
    declared = [{"id": "upstream", "check": "git_compare",
                "remote": "https://github.com/a/b", "ref": "main",
                "pinned": "abc", "order": None}]
    store.set_watch("oci:local:aeon-7/ods-hipfire", declared)

    watcher.tick()

    watch = store.entry("oci:local:aeon-7/ods-hipfire")["watch"]
    assert watch[0] == declared[0], "the declared source must survive verbatim"
    assert [s["id"] for s in watch] == ["upstream", "channel"]


def test_provenance_pass_does_not_append_history_for_an_unchanged_seeded_watch(tmp_path):
    """Risk 3: an unconditional set_watch call every pass would append a
    'declared' history line every provenance_interval_s forever for a watch
    that never changes. Two ticks over an unchanged, already-seeded origin
    must leave exactly one 'watch' record."""
    clock = _FakeClock()
    reference = "ghcr.io/aeon-7/ods-hipfire:slim@sha256:deadbeef"
    watcher, store, _docker = _one_container(
        tmp_path, clock=clock,
        bodies={"ods-hipfire": {"Image": "sha256:a", "Config": {"Image": reference}}})
    store.declare_origin(
        "oci:local:aeon-7/ods-hipfire", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "aeon-7/ods-hipfire",
                "reference": reference})

    watcher.tick()
    clock.advance(10_000)
    watcher.tick()
    clock.advance(10_000)
    watcher.tick()

    from app import provenance_history
    records = provenance_history.history_for(
        tmp_path / "prov-history.jsonl", "oci:local:aeon-7/ods-hipfire")
    watch_records = [r for r in records if r["field"] == "watch"]
    assert len(watch_records) == 1


def test_provenance_pass_does_not_resurrect_a_deliberately_cleared_watch(tmp_path):
    """The end-to-end version of the resurrection risk: an operator watches,
    gets checked at least once (recorded via record_update, the only writer
    of `update`), then explicitly clears the watch. The next collector pass
    must leave it cleared, not silently re-add the derived source."""
    clock = _FakeClock()
    reference = "ghcr.io/aeon-7/ods-hipfire:slim@sha256:deadbeef"
    watcher, store, _docker = _one_container(
        tmp_path, clock=clock,
        bodies={"ods-hipfire": {"Image": "sha256:a", "Config": {"Image": reference}}})
    store.declare_origin(
        "oci:local:aeon-7/ods-hipfire", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "aeon-7/ods-hipfire",
                "reference": reference})

    watcher.tick()   # seeds "channel"
    assert store.entry("oci:local:aeon-7/ods-hipfire")["watch"] != []

    # Simulate an update-check pass having run at least once, then the
    # operator clearing the watch via PUT /watch {sources: []}.
    store.record_update("oci:local:aeon-7/ods-hipfire", [
        {"id": "channel", "status": "current", "current": "sha256:deadbeef",
         "latest": "sha256:deadbeef", "detail": {}, "note": None}])
    store.set_watch("oci:local:aeon-7/ods-hipfire", [])
    assert store.entry("oci:local:aeon-7/ods-hipfire")["watch"] == []

    clock.advance(10_000)
    watcher.tick()

    assert store.entry("oci:local:aeon-7/ods-hipfire")["watch"] == [], (
        "a deliberately cleared, already-checked watch must not be "
        "resurrected by the next collector pass")


# --- provenance pass: the SPARK half of watch seeding -----------------------
#
# app/arbiter.py:1182 (the spark call to _provenance_seed_watch) had ZERO
# coverage: deleting the line left the whole suite green, while deleting the
# local equivalent at :1130 is killed by three tests. Sparky's images are the
# digest-pinned ones this feature was designed around, so the untested half
# was the important half.


class _FakeSpark:
    """The three node-agent reads Watcher._provenance_nodes makes for one
    node (client.status()/get_compose()/get_catalog()) and nothing else.
    ``calls`` counts all three, so a test can assert none of them ran."""

    def __init__(self, composes, catalog=None, raises=None):
        self.composes = composes
        self.catalog = catalog
        self.raises = raises
        self.calls = 0

    def status(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return {"profiles": [{"name": name} for name in sorted(self.composes)]}

    def get_compose(self, profile):
        self.calls += 1
        return self.composes[profile]

    def get_catalog(self):
        self.calls += 1
        return {"catalog": self.catalog}


def _spark_compose(image):
    return f"services:\n  vllm:\n    image: {image}\n"


_SPARK_REFERENCE = "ghcr.io/aeon-7/aeon-vllm-ultimate:v0.26.0@sha256:deadbeef"
_SPARK_ARTIFACT = "oci:sparky:aeon-7/aeon-vllm-ultimate"


def _spark_watcher(tmp_path, **kwargs):
    store = _prov_store(tmp_path)
    spark = _FakeSpark({"heretic": _spark_compose(_SPARK_REFERENCE)})
    watcher = _watcher(tmp_path, provenance_store=store, **_sparky_wiring(spark), **kwargs)
    return watcher, store


def test_provenance_pass_seeds_a_watch_for_a_spark_artifact_too(tmp_path):
    """Same guarantee as the local path, on the node it matters most for."""
    watcher, store = _spark_watcher(tmp_path)
    store.declare_origin(
        _SPARK_ARTIFACT, kind="oci", node="sparky", role="engine",
        origin={"registry": "ghcr.io",
                "repository": "aeon-7/aeon-vllm-ultimate",
                "reference": _SPARK_REFERENCE})

    watcher.tick()

    watch = store.entry(_SPARK_ARTIFACT)["watch"]
    assert [s["id"] for s in watch] == ["channel"]
    assert watch[0]["pinned"] == "sha256:deadbeef"
    assert watch[0]["registry"] == "ghcr.io"


def test_provenance_pass_derives_no_spark_watch_without_a_declared_origin(tmp_path):
    """D8 on the spark path too: an observed image is not an origin."""
    watcher, store = _spark_watcher(tmp_path)

    watcher.tick()

    assert store.entry(_SPARK_ARTIFACT)["watch"] == []


def test_provenance_spark_skips_probes_while_observer_says_unreachable(tmp_path):
    """With the observer caching 'unreachable', _provenance_nodes must make
    ZERO direct spark calls — the backoff exists precisely so a down node
    costs nothing between provenance passes either."""
    store = _prov_store(tmp_path)
    spark = _FakeSpark({"heretic": _spark_compose(_SPARK_REFERENCE)},
                       raises=EngineError("down"))
    watcher = _watcher(tmp_path, provenance_store=store, **_sparky_wiring(spark))

    watcher._node_observers.observer_for("sparky").status()  # seed the cache: unreachable
    spark.calls = 0  # only calls made by the pass under test should count
    watcher._provenance_pass()

    assert spark.calls == 0


def test_a_watch_source_set_watch_refuses_does_not_abandon_the_rest_of_the_pass(
        tmp_path):
    """`_provenance_seed_watch` -> `set_watch` -> `validate_watch` raises
    BadWatch on a hand-edited source (here: a tag check with no `order`).
    Unhandled, that escaped into `_provenance_pass`'s catch-all
    (app/arbiter.py:1106) and abandoned everything after it -- local weights
    plus the entire sparky sweep -- once every provenance_interval_s,
    forever. That contradicts the method's own docstring at
    app/arbiter.py:1077-1080 ("best-effort per source ... never blanks the
    others").

    The oci collector runs FIRST (app/arbiter.py:1103-1105), so the weights
    entry existing at all is the proof the pass carried on."""
    reference = "ghcr.io/aeon-7/ods-hipfire:slim@sha256:deadbeef"
    artifact_id = "oci:local:aeon-7/ods-hipfire"
    catalog = _FakeCatalogUnits([{
        "id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
        "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}])
    watcher, store, _docker = _one_container(
        tmp_path, catalog=catalog,
        bodies={"ods-hipfire": {"Image": "sha256:a", "Config": {"Image": reference}}})
    store.declare_origin(
        artifact_id, kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "aeon-7/ods-hipfire",
                "reference": reference})

    # A source no write path could have produced: `git_tags` with no `order`.
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data[artifact_id]["watch"] = [{"id": "hand", "check": "git_tags",
                                   "remote": "https://github.com/a/b",
                                   "pinned": "v1.0.0"}]
    path.write_text(json.dumps(data))

    watcher.tick()

    assert store.entry("file:local:m.gguf") is not None, (
        "one unvalidatable watch source abandoned the rest of the pass")
    kinds = [e["kind"] for e in tail_events(tmp_path / "events.jsonl")]
    assert "provenance-seed-watch-failed" in kinds, "refused, but silently"
    assert "provenance-pass-error" not in kinds


def test_failed_unload_with_a_malformed_prior_record_still_isolates(tmp_path):
    """[T7 review Important-1] The isolation handler could itself raise.

    `prior` comes from IntentStore.get(); before the boundary gate, a
    malformed record on disk reached it verbatim, so put_back's validation
    raised ValueError from INSIDE the `except EngineError` block — producing
    exactly what T7 exists to prevent: tick-error, no 'unload-failed' event,
    reconcile skipped. Fixed at the boundary (IntentStore._load gates each
    record), so this drives the whole path with a hand-corrupted intent.json
    and requires the tick to survive it.
    """
    from app.intent import IntentStore

    path = tmp_path / "intent.json"
    path.write_text(json.dumps({
        "local/lemonade": {"state": "loaded"},           # no engine, no model
        "local/hipfire": {"state": "loaded", "model": None, "engine": "hipfire"},
    }))
    intent = IntentStore(path)

    lemonade = FakeLemonade(raise_on_unload=EngineError("lemonade unreachable"))
    hipfire = FakeHipfire(state="parked")
    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.foo.gguf", idle_s=1000),
        hipfire=_hip(state="parked"),
        default_route=None,
    )
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot), FakeRegistry(), _policy(lem_idle=900),
        lemonade=lemonade, hipfire=hipfire, intent_store=intent,
    )

    watcher.tick()  # must not raise

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "tick-error" not in kinds        # the whole point
    assert "unload-failed" in kinds         # the diagnostic survived
    assert "resume" in hipfire.calls        # the tick carried on into reconcile


def test_failed_unload_does_not_revert_an_operator_action_that_raced_it(tmp_path):
    """[T7 review m1] The rollback is compare-and-swap, not last-write-wins.

    An operator can record a deliberate load in the seconds the unload call
    is hanging. Blindly putting `prior` back would silently revert THEIR
    action — the same class of bug as the pull-through supersession hole
    task 6 closed. The fake writes an operator record and then raises,
    reproducing exactly that interleave.
    """
    from app.intent import IntentStore

    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/lemonade", state="loaded", model="extra.old.gguf",
                  engine="lemonade", actor="operator")

    class RacingLemonade(FakeLemonade):
        def unload(self, model):
            self.unloaded.append(model)
            # The operator's action lands mid-call, after this arm's
            # speculative pre-record.
            intent.record("local/lemonade", state="loaded",
                          model="extra.NEW.gguf", engine="lemonade",
                          actor="operator")
            raise EngineError("lemonade unreachable")

    snapshot = _world(
        lemonade=_lem(state="loaded", model="extra.old.gguf", idle_s=1000),
        default_route=None,
    )
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot), FakeRegistry(), _policy(lem_idle=900),
        lemonade=RacingLemonade(), intent_store=intent,
    )

    watcher.tick()

    record = intent.get()["local/lemonade"]
    assert record["model"] == "extra.NEW.gguf"   # the operator's, not prior's
    assert record["actor"] == "operator"
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "unload-failed" in kinds
    assert "unload-rollback-skipped" in kinds


def test_repeated_identical_load_failures_are_deduped(tmp_path):
    """One degraded backend used to write an identical load-failed line
    every 2 s tick for the whole outage — the reachable events-trim-thrash
    input [T10 review]. The failure memo is per-kind, NOT the one-slot
    _last_event_key: this fixture's ticks interleave a free_comfyui event
    with every failure, exactly the shape that defeated a global
    last-event check. Identical failures log ONCE across the outage."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB < footprint
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade = FakeLemonade(raise_on_load=EngineError("connection refused"))
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(), lemonade=lemonade, comfy=FakeComfy(),
    )

    watcher.tick()
    watcher.tick()
    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.count("load-failed") == 1


def test_load_failure_after_recovery_is_logged_again(tmp_path):
    """fail -> recover -> identical fail must log BOTH failures — the
    dedup-key-never-cleared flap-blindness the T9-fix review named as this
    codebase's forbidden dedup class. The successful load in between
    re-arms the memo explicitly."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade = FakeLemonade(raise_on_load=EngineError("connection refused"))
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(), lemonade=lemonade, comfy=FakeComfy(),
    )

    watcher.tick()                       # fail #1 -> logged
    lemonade._raise_on_load = None
    watcher.tick()                       # recovery: load succeeds
    lemonade._raise_on_load = EngineError("connection refused")
    watcher.tick()                       # fail #2, identical -> must log

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.count("load-failed") == 2


def test_load_failure_detail_is_bounded(tmp_path):
    """EngineError carries raw HTTP bodies; a body over events.py's
    2500 B/line hysteresis ratio used to ride into the log verbatim. The
    fixture is 4000 chars — deliberately ABOVE that ratio, where the
    pre-fix behavior and the truncated one visibly differ."""
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    lemonade = FakeLemonade(raise_on_load=EngineError("x" * 4000))
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(), lemonade=lemonade, comfy=FakeComfy(),
    )

    watcher.tick()

    failed = next(e for e in tail_events(events_path)
                  if e["kind"] == "load-failed")
    assert len(failed["detail"]["error"]) < 600
    assert failed["detail"]["error"].endswith("[truncated]")


# ===========================================================================
# sglang-omni Task 6 — remote engine observation inside a real tick.
#
# The registry state is HAND-BUILT (conftest's HandBuiltRegistry): the Task 5
# write gate refuses a LEMONADE-kind declaration on a node-agent entry (still
# true after Task 7, which made only sglang-omni remote-capable), and these
# tests deliberately use one — the remote observation path must work for ANY
# kind, not just the one it was built alongside. The gate is not weakened. Fixture discipline: node "nimbus" (NOT
# "sparky"), resource "gguf-r" (NOT "omni"), resource != kind name, GPU 4.
# ===========================================================================

_REMOTE_ENGINE = {
    "resource": "gguf-r", "kind": "lemonade",
    "connection": {"url": "http://gguf-r:8080",
                   "metrics_url": "http://gguf-r:8081/metrics",
                   "container": "gguf-r"},
    "gpu_index": 4,
    "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900},
}


class _DownAgent:
    """A node-agent that is not answering — the normal state of a box that
    is powered off."""

    def __init__(self) -> None:
        self.closed = False

    def gpu(self):
        raise NodeAgentUnreachable("connection refused")

    def close(self) -> None:
        self.closed = True


class _RemoteWatcher:
    """The Task 6 counterpart of _TwoGgufWatcher: one local lemonade-kind
    resource plus one engine DECLARED on a remote node whose agent is down."""

    def __init__(self, watcher, clients, intent_store, events_path, world,
                 registry_store, agent_factory, remote_clients, gpus,
                 litellm, footprints) -> None:
        self.watcher = watcher
        self.clients = clients
        self.intent_store = intent_store
        self.events_path = events_path
        self.world = world
        self.registry_store = registry_store
        self.agent_factory = agent_factory
        self.remote_clients = remote_clients
        self._gpus = gpus
        self._litellm = litellm
        self._footprints = footprints

    def remote_half(self) -> dict:
        from app.node_clients import remote_world_half

        return remote_world_half(self.registry_store, self.agent_factory,
                                 self.remote_clients, self.world,
                                 self._footprints)


@pytest.fixture
def remote_engine_watcher(tmp_path, hand_built_registry):
    from app.intent import IntentStore
    from app.node_clients import RemoteEngineClients, RemoteObserver
    from app.state import World

    entries = [
        {"id": "local", "label": "This Box", "agent_kind": "local",
         "control": "none", "engines": [_TWO_GGUF_ENGINES[1]]},   # gguf-b only
        {"id": "nimbus", "label": "Nimbus Box", "agent_kind": "node-agent",
         "address": "http://nimbus:7720", "control": "none",
         "engines": [dict(_REMOTE_ENGINE)]},
    ]
    registry_store = hand_built_registry(entries, {"nimbus": "key-nimbus"})

    agents = []

    def agent_factory(address, credential):
        agent = _DownAgent()
        agents.append((address, credential, agent))
        return agent

    agent_factory.opened = agents

    litellm = _FakeLiteLLMRoutes()
    footprints = FakeRegistry(footprints={"b.gguf": 5 * GIB})
    clients = _FakeLocalClients(
        {"gguf-b": _PendingCapableLemonade(litellm, footprints, loaded=None)})
    intent_store = IntentStore(tmp_path / "intent.json")
    world = World()
    gpus = [{"index": 3, "vram_total": 100, "vram_used": 10, "pids": {}}]
    remote_clients = RemoteEngineClients(
        registry_store, lambda entry, credential, engine: _UnreachableRemote())

    watcher, events_path = _make_watcher(
        tmp_path, world, footprints, _policy(),
        node_store=registry_store, local_clients=clients,
        intent_store=intent_store, litellm=litellm,
        read_gpus=lambda drm, kfd: gpus,
        remote_engine_clients=remote_clients,
        node_agent_client_factory=agent_factory,
        # RE-EXPRESSED (sglang-omni Task 7): the tick reaches the remote
        # half THROUGH the shared observer now (app.main wires one), so the
        # fixture supplies one — a real RemoteObserver on its real default
        # TTL, not a pass-through, because the pacing is part of what these
        # tests describe.
        remote_observer=RemoteObserver(),
    )
    return _RemoteWatcher(watcher, clients, intent_store, events_path, world,
                          registry_store, agent_factory, remote_clients, gpus,
                          litellm, footprints)


class _UnreachableRemote:
    """A remote engine client whose every probe fails — what a real one does
    while its box is off."""

    def load_in_flight(self) -> bool:
        return False

    def status(self) -> dict:
        raise EngineError("nimbus is not answering")

    def activity(self):
        return None

    def close(self) -> None:
        pass


def test_remote_engine_appears_in_the_world_with_its_node_id(
        remote_engine_watcher):
    tenant = remote_engine_watcher.remote_half()["remote_tenants"]["nimbus/gguf-r"]

    assert tenant["node_id"] == "nimbus"
    assert tenant["resource"] == "gguf-r"
    assert tenant["gpu_index"] == 4


def test_remote_engine_on_a_down_agent_observes_unknown(remote_engine_watcher):
    """Unknown means "we failed to look", never "nothing is loaded" — the
    difference decides whether the reconciler restores something that may
    already be running on that box."""
    from app.observe import observe_remote

    world = dict(remote_engine_watcher.remote_half())

    observed = observe_remote(world)

    assert observed["nimbus/gguf-r"] == {
        "reachable": False, "loaded": False, "model": None,
        "transitioning": False}


def test_tick_survives_a_down_remote_agent_and_still_reconciles(
        remote_engine_watcher):
    """A down remote box must not kill the tick (the N1 lesson: malformed /
    unreachable HEALS, it never kills).

    Two vacuity guards, because "no tick-error was logged" is worthless on a
    tick that never ran the code in question: the agent factory must have
    been ASKED (the tick really did reach the remote half), and the local
    resource's restore must have fired (the tick really did get past the
    remote half into reconciliation).
    """
    w = remote_engine_watcher
    w.intent_store.record("local/gguf-b", state="loaded", model="b.gguf",
                          engine="lemonade", actor="operator")

    w.watcher.tick()

    assert [address for address, _c, _a in w.agent_factory.opened] == [
        "http://nimbus:7720"]
    assert w.clients.fake("gguf-b").loaded == ["b.gguf"]
    assert not any(e["kind"] == "tick-error" for e in tail_events(w.events_path))


def test_ticks_pace_the_remote_gpu_probe(remote_engine_watcher):
    """sglang-omni Task 7 — the pacing obligation, inside the tick loop.

    A powered-off remote node is the NORMAL state, and its GPU read is a 5 s
    transport timeout. Unpaced, EVERY ~2 s tick paid it. Two ticks in the
    same instant now cost one probe (the watcher shares the deck's
    RemoteObserver with the HTTP paths, so they still describe one world).

    Vacuity guard: the probe must have happened at all, and the tick must
    still have reached reconciliation past the remote half."""
    w = remote_engine_watcher
    w.intent_store.record("local/gguf-b", state="loaded", model="b.gguf",
                          engine="lemonade", actor="operator")

    w.watcher.tick()
    w.watcher.tick()

    assert [address for address, _c, _a in w.agent_factory.opened] == [
        "http://nimbus:7720"]
    assert w.clients.fake("gguf-b").loaded == ["b.gguf"]


# ===========================================================================
# sglang-omni Task 9 — remote engine RECONCILE + IDLE RELEASE, inside a real
# tick.
#
# Task 6 proved a remote engine is OBSERVED; this section proves the deck can
# act on one: restore it when it dies, hold off while it boots, release it
# when it goes idle, and quarantine it when restores keep failing — each
# through the node-agent client the observation half already uses.
#
# The declaration goes through the REAL NodeStore write gate (sglang-omni is
# remote_capable since Task 7), not conftest's HandBuiltRegistry: these are
# the paths an operator's own declaration takes, and the deck-wide resource
# uniqueness gate this task adds sits on exactly that write.
#
# Fixture discipline ([[defaults-that-hide-bugs]]): node "nimbus" (never the
# live-seeded "sparky"), resource "song-r" (never "omni"), GPU 4, idle_ttl
# 120 (never the declared 900 the local lemonade seed uses).
# ===========================================================================

_OMNI_ENGINE = {
    "resource": "song-r", "kind": "sglang-omni",
    "connection": {"url": "http://nimbus:8008"},
    "gpu_index": 4,
    "container_consent": True,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}

_OMNI_KEY = "nimbus/song-r"
_OMNI_IDLE_TTL = 120


def _omni_policy(pinned=False, idle_ttl=_OMNI_IDLE_TTL):
    return {"song-r": {"priority": 5, "pinned": pinned, "idle_ttl": idle_ttl}}


def _ago(seconds: float) -> str:
    """An ISO timestamp `seconds` in the past — what an intent record's
    `updated_ts` looks like that long after the operator acted. Real wall
    clock deliberately: the warming window is measured against
    ``datetime.now(UTC)`` (app.engine_kinds._within), not the injectable
    monotonic clock the watcher's own cooldowns use."""
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


class _FakeOmni:
    """SglangOmniClient-shaped fake: the three calls the sglang-omni adapter
    makes (status/up/down) plus close(). `ups`/`downs` count the ATTEMPT
    before the guard raises, so a test can tell "never dispatched" apart from
    "dispatched and raised" (the FakeSpark lesson from N1 task 1)."""

    def __init__(self, *, healthy=True, busy=0, reachable=True,
                 raise_on_status=None, raise_on_up=None, raise_on_down=None):
        self.healthy = healthy
        self.busy = busy
        self.reachable = reachable
        self.status_calls = 0
        self.ups = 0
        self.downs = 0
        self._raise_on_status = raise_on_status
        self._raise_on_up = raise_on_up
        self._raise_on_down = raise_on_down

    def status(self) -> dict:
        self.status_calls += 1
        if self._raise_on_status is not None:
            raise self._raise_on_status
        return {"reachable": self.reachable, "healthy": self.healthy,
                "busy_requests": self.busy}

    def up(self) -> None:
        self.ups += 1
        if self._raise_on_up is not None:
            raise self._raise_on_up

    def down(self) -> None:
        self.downs += 1
        if self._raise_on_down is not None:
            raise self._raise_on_down

    def close(self) -> None:
        pass


class _FakeAgent:
    """node-agent stub for the ONE read the remote half makes per node: the
    GPU pool (app.node_clients.read_remote_gpus), which is also that node's
    liveness probe."""

    def __init__(self) -> None:
        self.calls = 0

    def gpu(self) -> dict:
        self.calls += 1
        return {"gpus": [{"index": 4, "memory_total_mb": 131072,
                          "memory_used_mb": 65536}]}

    def close(self) -> None:
        pass


class _OmniWatcher:
    """remote_omni_watcher's return value: the watcher plus the handles its
    tests drive it through."""

    def __init__(self, watcher, omni, agent, intent_store, events_path, clock,
                 node_store, world) -> None:
        self.watcher = watcher
        self.omni = omni
        self.agent = agent
        self.intent_store = intent_store
        self.events_path = events_path
        self.clock = clock
        self.node_store = node_store
        self.world = world

    def events(self, kind: str) -> list[dict]:
        return [e for e in tail_events(self.events_path) if e["kind"] == kind]

    def tick_error_free(self) -> bool:
        """The vacuity guard every proof-of-absence in this section pairs
        with: a tick that died in the supervisor catch actuates nothing for
        reasons that have nothing to do with the rule under test."""
        return not self.events("tick-error")

    @property
    def policy_store(self):
        """The watcher's own FakePolicyStore — the same instance the real
        forget route's third write lands on (sglang-omni Task 11)."""
        return self.watcher._policy_store

    def derived_status(self, key: str = _OMNI_KEY) -> str:
        """The lifecycle status the deck derives for `key`, through the SAME
        two stages the board and the reconcile pass share
        (``join_warming`` then ``derive_status`` —
        app.routers.build_lifecycle_view and app.arbiter._reconcile_pass both
        run exactly this pair over one snapshot).

        COSTS ONE FRESH OBSERVATION of its own: it re-assembles the remote
        half rather than reading back whatever the last tick held, so a test
        asserting on ``omni.status_calls`` must capture that count BEFORE
        calling this (the sglang-omni Task 11 proofs do, into a local).
        """
        from app.lifecycle import derive_status, join_warming
        from app.observe import merge_observations, observe_remote

        world = self.watcher._world.snapshot([], [], _FakeLocalClients({}),
                                             _FakeLiteLLMRoutes(), FakeRegistry())
        world.update(self.watcher._remote_observer.half(
            self.node_store, lambda address, credential: self.agent,
            self.watcher._remote_engine_clients, self.watcher._world,
            FakeRegistry()))
        intents = self.intent_store.get()
        observed = join_warming(merge_observations(observe_remote(world)), world,
                                intents)
        return derive_status(intents.get(key), observed[key])["status"]

    def forget(self, node_id: str = "nimbus", resource: str = "song-r") -> None:
        """Forget one declared engine through the REAL route
        (app.routers.nodes.forget_engine), called directly with a
        Request-shaped stand-in.

        Deliberately NOT the store-level mirror the LOCAL §6.2 proof uses
        (`_TwoGgufWatcher.forget` above, whose own docstring says why a
        mirror was right there: its fixture wires no policy store, and the
        E1-era route was local-only). This proof is about a call that must
        never happen — the node-agent engine channel — and a mirror can only
        ever prove the MIRROR makes no engine call. So the code under test is
        the route itself: its three writes (declaration, then intent, then
        policy), its `_reobserve` cache drop, and, crucially, the absence of
        any client lookup anywhere inside it.
        """
        from types import SimpleNamespace

        from app.routers.nodes import forget_engine

        deck = {
            "node_store": self.node_store,
            "intent_store": self.intent_store,
            "policy_store": self.policy_store,
            "events_path": self.events_path,
            "remote_observer": self.watcher._remote_observer,
        }
        forget_engine(node_id, resource, SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(deck=deck))))


@pytest.fixture
def remote_omni_watcher(tmp_path):
    """A watcher over ONE engine declared on a node-agent entry: nimbus/song-r
    (sglang-omni kind), observed healthy and idle by default. No local engine
    is declared at all — the remote half must stand on its own."""
    from app.intent import IntentStore
    from app.node_clients import RemoteEngineClients, RemoteObserver
    from app.node_store import NodeStore
    from app.state import World

    node_store = NodeStore(tmp_path / "nodes.json",
                           tmp_path / "node_credentials.json")
    node_store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    node_store.add({"id": "nimbus", "label": "Nimbus Box",
                    "agent_kind": "node-agent", "address": "http://nimbus:7720",
                    "control": "none"}, credential="key-nimbus")
    node_store.update("nimbus", {"engines": [dict(_OMNI_ENGINE)]})

    clock = _FakeClock()
    omni = _FakeOmni()
    agent = _FakeAgent()
    remote_clients = RemoteEngineClients(
        node_store, lambda entry, credential, engine: omni)
    intent_store = IntentStore(tmp_path / "intent.json")
    world = World(clock=clock)

    watcher, events_path = _make_watcher(
        tmp_path, world, FakeRegistry(), _omni_policy(),
        node_store=node_store, local_clients=_FakeLocalClients({}),
        intent_store=intent_store, litellm=_FakeLiteLLMRoutes(),
        read_gpus=lambda drm, kfd: [],
        remote_engine_clients=remote_clients,
        node_agent_client_factory=lambda address, credential: agent,
        # The real observer on its real TTL/backoff, driven by the SAME
        # clock as the world and the watcher: the pacing is part of what
        # these tests describe (a cached half is what a second tick in the
        # same instant sees).
        remote_observer=RemoteObserver(clock=clock),
        clock=clock,
    )
    return _OmniWatcher(watcher, omni, agent, intent_store, events_path, clock,
                        node_store, world)


# --- obligation 1: the remote restore path ---------------------------------


def test_remote_restore_dispatches_through_the_declaring_nodes_client(
        remote_omni_watcher):
    """R9 closed: intent loaded x observed down (past the warm window) ->
    the deck asks THAT node's agent to bring THAT engine up.

    Before this task every non-spark key resolved through the LOCAL client
    map, which holds nothing for a remote resource, so the attempt raised
    "no restore handler" and charged the failure budget instead."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(1200))

    w.watcher.tick()

    assert w.omni.ups == 1
    assert [e["detail"]["key"] for e in w.events("lifecycle-restore")] == [
        _OMNI_KEY]
    assert w.tick_error_free()


def test_remote_restore_without_health_charges_the_budget_on_redispatch(
        remote_omni_watcher):
    """Part A's no-raise hole, proven generic: `up()` returns fine (202 by
    contract — nothing observes the result), the engine never becomes
    healthy, and the NEXT dispatch for the still-unverified key charges the
    failure budget. No new code was expected here; the proof is that the
    machinery keys on the REMOTE key like any other."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(1200))

    w.watcher.tick()
    w.clock.advance(31)          # past _RESTORE_COOLDOWN_S and the remote TTL
    w.watcher.tick()

    assert w.omni.ups == 2
    failures = [e["detail"] for e in w.events("lifecycle-restore-failed")]
    assert [f["key"] for f in failures] == [_OMNI_KEY]
    assert failures[0]["failures"] == 1


def test_two_failed_remote_restores_quarantine_and_stop_retrying(
        remote_omni_watcher):
    """The failure budget's end state for a remote key: after FAILURE_BUDGET
    charges the key is quarantined and the reconciler stops dispatching —
    the crash-loop this budget exists to end."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(1200))

    for _ in range(4):
        w.watcher.tick()
        w.clock.advance(31)

    assert [e["detail"]["key"] for e in w.events("lifecycle-quarantined")] == [
        _OMNI_KEY]
    assert w.intent_store.get()[_OMNI_KEY]["quarantined"] is True
    # Two dispatches, then silence: the third tick's redispatch charge is
    # what quarantines, and nothing dispatches after it.
    assert w.omni.ups == 2


def test_a_remote_restore_that_raises_charges_the_budget_and_quarantines(
        remote_omni_watcher):
    """The RAISING half of the failure budget, for a remote key: the node
    agent refuses (409 pending, 503 swap-ctl disabled, transport down — all
    EngineError by the client's own taxonomy), which is a restore that
    visibly failed rather than one that merely never became healthy. Two of
    those and the key is quarantined."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.omni._raise_on_up = EngineError("nimbus: a request is already pending")
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(1200))

    w.watcher.tick()
    w.clock.advance(31)
    w.watcher.tick()

    assert w.omni.ups == 2
    failures = [e["detail"] for e in w.events("lifecycle-restore-failed")]
    assert [f["failures"] for f in failures] == [1, 2]
    assert "already pending" in failures[0]["error"]
    assert w.intent_store.get()[_OMNI_KEY]["quarantined"] is True


class _VanishingClients:
    """RemoteEngineClients-shaped, answering the real client for the first
    `answers` lookups and None afterwards — the credential-vanished-mid-flight
    race app.node_clients documents (design §9), landed precisely between a
    tick's observation phase and its restore phase.

    That gap is the only way the restore path's not-operable branch is
    reachable at all: an engine whose client is already None when the world
    is assembled observes `unknown` -> `unreachable`, which the reconciler
    never acts on. Contrived on purpose — the branch is real, and pinning it
    with a fixture that merely undeclares the engine would prove nothing
    (no observation, no restore, assertion satisfied vacuously)."""

    def __init__(self, client, answers: int) -> None:
        self._client = client
        self._answers = answers
        self.calls = 0

    def client_for(self, node_id, resource):
        self.calls += 1
        return self._client if self.calls <= self._answers else None

    def retire_absent(self, keep_pairs) -> None:
        pass


def test_a_restore_for_an_inoperable_remote_node_is_a_named_failure(
        remote_omni_watcher):
    """`RemoteEngineClients.client_for` is repair-shaped: it answers None for
    a pair that is not operable right now rather than raising. The restore
    path must turn that into ONE charged, NAMED failure — naming the node,
    since "no restore handler for engine 'sglang-omni'" alone cannot say
    which box went dark — and never a crash inside the tick."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(1200))
    clients = _VanishingClients(w.omni, answers=1)   # the observation, then dark
    w.watcher._remote_engine_clients = clients

    w.watcher.tick()

    assert clients.calls == 2            # observed, then asked to restore
    assert w.omni.ups == 0
    failures = [e["detail"] for e in w.events("lifecycle-restore-failed")]
    assert [f["failures"] for f in failures] == [1]
    assert "nimbus" in failures[0]["error"]
    assert w.tick_error_free()


# --- obligation 3: the warming join ----------------------------------------


def test_a_cold_boot_inside_the_warm_window_draws_no_restore(
        remote_omni_watcher):
    """GF4: a cold start takes ~3.5-4.5 min and observes EXACTLY like a
    death. Read as one, the reconciler restores it every cooldown and
    quarantines it before it ever finishes booting.

    Three vacuity guards, because "no restore" is worthless on a tick that
    never looked: the engine must have been PROBED, the tick must not have
    died, and the sibling test below must show the same fixture restoring
    once the window lapses."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(60))

    for _ in range(3):
        w.watcher.tick()
        w.clock.advance(31)

    assert w.omni.ups == 0
    assert w.events("lifecycle-restore") == []
    assert w.omni.status_calls > 0
    assert w.tick_error_free()


def test_restores_resume_once_the_warm_window_lapses(remote_omni_watcher):
    """The other half of the pair above: an intent recorded longer ago than
    the boot window is no longer a boot in flight — it is the 26-hour
    hipfire failure, and the deck must act on it."""
    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(601))

    w.watcher.tick()

    assert w.omni.ups == 1


def test_a_warming_remote_engine_reports_warming_not_down(remote_omni_watcher):
    """The board and the reconciler must agree about what they are looking
    at: the same join that holds the restore off is what makes /api/lifecycle
    say `warming` rather than `down` for those four minutes."""
    from app.lifecycle import derive_status, join_warming
    from app.observe import merge_observations, observe_remote

    w = remote_omni_watcher
    w.omni.healthy = False
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(60))
    world = w.watcher._world.snapshot([], [], _FakeLocalClients({}),
                                      _FakeLiteLLMRoutes(), FakeRegistry())
    world.update(w.watcher._remote_observer.half(
        w.node_store, lambda address, credential: w.agent,
        w.watcher._remote_engine_clients, w.watcher._world, FakeRegistry()))

    observed = join_warming(merge_observations(observe_remote(world)), world,
                            w.intent_store.get())

    assert observed[_OMNI_KEY]["transitioning"] is True
    assert derive_status(w.intent_store.get()[_OMNI_KEY],
                         observed[_OMNI_KEY])["status"] == "warming"


# --- obligation 4: idle release for a remote engine ------------------------


def _idle_past_ttl(w) -> None:
    """Drive `w`'s remote engine to an observation that is idle PAST its
    idle_ttl: one tick establishes the clock baseline (every kind's
    first-ever observation does), then time passes. Also expires the
    RemoteObserver's TTL, so the second tick really re-observes."""
    w.watcher.tick()
    w.clock.advance(_OMNI_IDLE_TTL + 80)


def test_remote_idle_release_unloads_through_the_declaring_nodes_client(
        remote_omni_watcher):
    """The idle rule reaches a remote engine at all — the arbiter iterates
    `remote_tenants` explicitly (it inherits NO coverage from the local
    `tenants` loop, which is a different world key entirely) and actuates
    through that node's own client.

    Whoever actuates, records: the unload writes intent for the NODE-keyed
    resource, actor "deck" (this is the arbiter's own automatic action, never
    an operator's)."""
    w = remote_omni_watcher
    _idle_past_ttl(w)

    w.watcher.tick()

    assert w.omni.downs == 1
    record = w.intent_store.get()[_OMNI_KEY]
    assert record["state"] == "unloaded"
    assert record["actor"] == "deck"
    assert record["engine"] == "sglang-omni"
    assert w.tick_error_free()


def test_an_unavailable_busy_count_never_unloads_a_running_render(
        remote_omni_watcher):
    """THE RENDER-PROTECTION HEADLINE (design §4). MiniMax-Music3 has no
    metrics endpoint at this pin (GF3), so "in flight" is the agent's own
    count of established connections — and `None` means it could not take
    that count. A song render runs for MINUTES holding its connection, so an
    unavailable indicator must fail toward ALIVE: reading it as 0 would
    unload the engine out from under a render in progress.

    Two vacuity guards: the engine was really probed (this is not a tick that
    never looked), and the sibling test above proves the SAME fixture with a
    real 0 does unload — so "nothing emitted" here is the busy count's doing,
    not an inert fixture."""
    w = remote_omni_watcher
    w.omni.busy = None
    _idle_past_ttl(w)

    for _ in range(3):
        w.watcher.tick()
        w.clock.advance(31)

    assert w.omni.downs == 0
    # Nothing was recorded either: whoever actuates records FIRST, so an
    # intent record would be evidence of an unload this assertion missed.
    assert w.intent_store.get() == {}
    assert w.omni.status_calls > 0
    assert w.tick_error_free()


def test_a_deliberately_unloaded_remote_engine_is_left_alone(
        remote_omni_watcher):
    """The D9 invariant, remote edition: intent `unloaded` is a DELIBERATE
    park, and the deck must fight it exactly never — not with a restore (a
    parked status is inert by construction, app.reconcile) and not with a
    second unload.

    Both observations a park can produce are covered: the engine actually
    down (status `parked`), and the engine still up (status `unexpected` —
    someone else acted, which the deck reports and never auto-corrects).
    Deliberately NOT covered: an observation idle PAST its ttl, where an
    idle-release unload is the deck moving TOWARD the recorded intent rather
    than fighting it — that is the sibling test above, not a violation of
    this one."""
    w = remote_omni_watcher
    w.intent_store.record(_OMNI_KEY, state="unloaded", model=None,
                          engine="sglang-omni", actor="operator",
                          now=_ago(30))

    # 31 s per tick clears the restore cooldown AND the observer's 10 s TTL,
    # so every tick here is a REAL re-observation. Total elapsed since the
    # last `down` (which re-arms the idle clock) stays well under the 120 s
    # idle_ttl on purpose: this test is about the two statuses named above,
    # and an idle-past-ttl observation would legitimately emit an unload —
    # keep that margin if you add ticks.
    for observed_healthy in (False, True):
        w.omni.healthy = observed_healthy
        for _ in range(3):
            w.watcher.tick()
            w.clock.advance(31)

    assert (w.omni.ups, w.omni.downs) == (0, 0)
    assert w.omni.status_calls > 0
    assert w.intent_store.get()[_OMNI_KEY]["state"] == "unloaded"
    assert w.tick_error_free()


def test_one_idle_observation_unloads_once_however_many_ticks_see_it(
        remote_omni_watcher):
    """The comfyui free-spam lesson, in the one place it bites hardest: a
    remote engine's observation is TTL-CACHED (app.node_clients.RemoteObserver,
    10 s) and its container takes seconds to stop, so the very same
    idle-past-ttl record is what every tick in that window reads. Unguarded
    that is one unload dispatch per ~2 s tick until the engine finally
    disappears.

    Two mechanisms, both proven here: acting on a remote engine drops the
    cached half (so the next tick re-observes rather than re-reading the
    record it just acted on), and the actuation re-arms that resource's idle
    clock (so the fresh observation is not instantly idle-past-ttl again
    while the container is still going down — the fake keeps reporting
    healthy-and-idle exactly to model that window)."""
    w = remote_omni_watcher
    _idle_past_ttl(w)

    w.watcher.tick()          # emits the unload
    w.watcher.tick()          # same instant: must not re-emit
    w.clock.advance(11)       # past the observer's TTL: a REAL re-observation
    w.watcher.tick()

    assert w.omni.downs == 1


# ===========================================================================
# sglang-omni Task 11 — the §6 COEXISTENCE proofs, fourth kind.
#
# The binding invariant (E1 design §6, Tim: "an alternative, not a
# replacement"), already proven at the top of this file for the LOCAL
# lemonade kind (test_natively_loaded_model_is_left_alone /
# test_forget_then_tick_never_calls_the_engine, the T13 section). These are
# the remote editions, and the fourth kind is what makes them worth writing
# twice: the thing an operator launches by hand off-box is the whole ENGINE
# (a compose up on sparky), not a model inside an engine the deck already
# talks to — and every touch the deck could make is now an HTTP call to
# another machine's agent, which is exactly what "never touched" has to mean
# here.
#
# Both are proofs of ABSENCE driven through a REAL watcher.tick(), and each
# carries the vacuity guards this section's `tick_error_free` docstring
# describes plus one more that matters more for a remote engine than for a
# local one: the engine must have been PROBED. "Nothing was actuated" is free
# on a tick that never looked, and a remote engine has three separate ways of
# never being looked at (node in backoff, observation TTL still warm, client
# not operable).
#
# The pacing (`clock.advance(11)` per tick) is load-bearing for the same
# reason: the RemoteObserver caches the assembled half for 10 s, so ticks in
# the same instant would prove only that the cache works.
#
# NOT covered here, deliberately: idle release past the declared ttl. §6.4
# names it as the ONE deliberate touch-point ("it unloads an idle
# lemonade-class model whoever loaded it"), so it is not a violation of §6.1
# — its own proofs are the obligation-4 pair above. Both proofs below keep
# their total elapsed time well inside the declared idle_ttl (120 s) so that
# rule is never in play; keep that margin if you add ticks.
# ===========================================================================

# Enough ticks that a reconciler with any opinion at all would have acted
# several times over, and few enough (at 11 s each) to stay inside the ttl.
_COEXIST_TICKS = 6

# A SECOND engine on the same node, declared by the §6.2 proof only. It is
# never forgotten, so its probes are what prove the post-forget ticks really
# did reach the remote half — nimbus's GPU pool is read only for nodes that
# declare something (app.node_clients.remote_world_half), so a node whose
# last engine was just forgotten stops being probed entirely, and "no calls"
# would otherwise be satisfiable by a tick that stopped looking at the box.
# Its own name/GPU are away from song-r's ([[defaults-that-hide-bugs]]), and
# it deliberately gets NO policy row: an engine the arbiter has no row for is
# never an idle-release candidate, so the neighbour cannot actuate anything
# while it stands in as the witness.
_OMNI_NEIGHBOUR = {
    "resource": "tune-r", "kind": "sglang-omni",
    "connection": {"url": "http://nimbus:8009"},
    "gpu_index": 5,
    "container_consent": True,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}


def test_natively_loaded_sglang_omni_is_left_alone(remote_omni_watcher):
    """Spec §6.1, fourth kind: an engine somebody brought up BY HAND on its
    own box — declared to the deck, observed healthy, with NO intent record
    anywhere — is watched and never touched.

    This is the state a box is in the moment an operator declares an engine
    that is already running, and the state it stays in for anyone who keeps
    using `docker compose up` on sparky directly. The deck's own docstrings
    call the derived status `unmanaged` — "loaded but the Deck has no intent
    for it", an ADOPT CANDIDATE — and adopting it is exactly what must not
    happen: a restore-shaped touch would be the deck claiming an engine
    nobody handed it, and an unload-shaped one would be the deck switching
    off a render somebody started.

    Four vacuity guards, because every assertion below is an absence:

    * the engine was really PROBED, once per tick (so this is not a tick that
      never looked, nor one served entirely from the observation cache);
    * the tick did not die in the supervisor catch;
    * nothing was RECORDED either — whoever actuates records first, so an
      intent record would be evidence of an actuation these counters missed;
    * and the status the deck derived really is `unmanaged`. That last one is
      the guard specific to a REMOTE engine: an unreachable node observes
      `unknown` -> `unreachable`, which the reconciler is inert about for
      reasons that have nothing to do with this rule, and a fixture that
      drifted that way would pass everything above while proving nothing.
    """
    w = remote_omni_watcher
    assert w.intent_store.get() == {}

    for _ in range(_COEXIST_TICKS):
        w.watcher.tick()
        w.clock.advance(11)

    # Captured BEFORE derived_status(), which takes a fresh observation of
    # its own (see its docstring).
    probes = w.omni.status_calls

    assert (w.omni.ups, w.omni.downs) == (0, 0)   # nothing actuated
    assert w.intent_store.get() == {}              # nothing recorded
    assert probes == _COEXIST_TICKS
    assert w.tick_error_free()
    assert w.derived_status() == "unmanaged"


def test_forget_remote_engine_never_calls_the_engine(remote_omni_watcher):
    """Spec §6.2, remote edition: forget is BOOKKEEPING. It drops the
    deck's three records of an engine — the declaration, the intent record,
    the stored policy row — and says nothing to the engine at all, which on
    another box means: not one node-agent request.

    Removing a declaration is an operator saying "stop watching this", never
    "shut this down". The engine keeps serving whatever it was serving; the
    only thing that ends is the deck's opinion about it. Getting this wrong
    on a remote engine is worse than getting it wrong locally — a forget that
    tidied up after itself would stop a render on a machine the operator was
    not even looking at.

    Driven through the REAL route (see `_OmniWatcher.forget`), with a
    prior intent record and a live policy row in place so that all three
    drops are DETECTABLE — asserting a key is absent from a store it was
    never in proves nothing.

    Three vacuity guards: the engine was really probed while it was still
    declared (so "no calls" describes a change, not a fixture that never
    called anything), the deck went on probing its NEIGHBOUR on the same node
    afterwards (so the ticks really did keep reaching the remote half — see
    `_OMNI_NEIGHBOUR`), and the tick never died.

    `close()` is not counted as touching the engine, and is not: retiring an
    undeclared pair (app.node_clients.RemoteEngineClients.retire_absent)
    releases the deck's OWN http client. It sends nothing.
    """
    from app.node_clients import RemoteEngineClients

    w = remote_omni_watcher
    neighbour = _FakeOmni()
    w.node_store.update("nimbus", {"engines": [dict(_OMNI_ENGINE),
                                               dict(_OMNI_NEIGHBOUR)]})
    w.watcher._remote_engine_clients = RemoteEngineClients(
        w.node_store,
        lambda entry, credential, engine:
            neighbour if engine["resource"] == _OMNI_NEIGHBOUR["resource"] else w.omni)
    w.intent_store.record(_OMNI_KEY, state="loaded", model=None,
                          engine="sglang-omni", actor="operator", now=_ago(30))

    w.watcher.tick()                 # while still declared: the deck DOES probe it
    probes_while_declared = w.omni.status_calls
    neighbour_probes_before = neighbour.status_calls
    w.clock.advance(11)

    assert probes_while_declared > 0
    assert _OMNI_KEY in w.intent_store.get()
    assert "song-r" in w.policy_store.get()

    w.forget()

    for _ in range(_COEXIST_TICKS):
        w.watcher.tick()
        w.clock.advance(11)

    # The engine: not one further call, of any kind.
    assert w.omni.status_calls == probes_while_declared
    assert (w.omni.ups, w.omni.downs) == (0, 0)
    # The deck's three records of it: gone.
    assert [e["resource"] for e in w.node_store.get("nimbus")["engines"]] == [
        _OMNI_NEIGHBOUR["resource"]]
    assert _OMNI_KEY not in w.intent_store.get()
    assert "song-r" not in w.policy_store.get()
    # ...and the ticks that saw none of that really ran, and really looked:
    # one real probe of the surviving engine per tick, on the same node.
    assert neighbour.status_calls == neighbour_probes_before + _COEXIST_TICKS
    assert w.tick_error_free()


# --- obligation 2: per-action isolation ------------------------------------


def test_an_unrecognized_action_is_logged_and_the_rest_still_run(tmp_path):
    """RE-EXPRESSED (was test_execute_raises_on_action_for_a_non_legacy_resource,
    T5 review fix). Its intent is kept exactly: an action `_execute` cannot
    dispatch must never vanish silently — no actuation, no log line — which
    is what a bare unmatched if-chain would do.

    What changes is the BLAST RADIUS, which is this task's obligation. The
    raise it used to assert escaped into `tick()`'s broad supervisor catch,
    taking the REMAINING actions and the whole reconcile/derive/provenance
    tail of that tick with it. Now the failure is logged as loudly as before
    and contained to its own action: the next one still runs."""
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        lemonade=lemonade,
    )

    watcher._execute(
        [{"type": "unload", "resource": "gguf-a", "model": "x.gguf"},
         {"type": "free", "resource": "img"},
         {"type": "unload", "resource": "lemonade", "model": "extra.m.gguf"}],
        None,
    )

    # The two undispatchable actions are each reported by name...
    failed = [e["detail"] for e in tail_events(events_path)
              if e["kind"] == "action-failed"]
    assert [f["resource"] for f in failed] == ["gguf-a", "img"]
    assert all("gguf-a" in f["error"] or "img" in f["error"] for f in failed)
    # ...and the action AFTER them still actuated.
    assert lemonade.unloaded == ["extra.m.gguf"]


def test_a_raising_free_still_blocks_the_pending_load_retrigger(tmp_path):
    """FIX ROUND 1 (I-1): isolation must not INVERT the eviction guard it
    wrapped.

    `_ComfyAdapter.execute_free` returns True — "an eviction raced" — for
    every failure it understands, and `_execute` reads that as "the VRAM is
    NOT confirmed reclaimed, so the pending load must not be re-triggered
    into a GPU that may still be full". An exception ESCAPING that adapter
    (exactly what per-action isolation was added to contain) carries the same
    meaning and must reach the same conclusion: the free did not happen, so
    the retrigger stays blocked.

    Before the fix the raise left `eviction_raced` False and the retrigger
    fired — a load pushed at a full GPU, which is strictly worse than the
    pre-isolation behavior (the tick died and the retrigger never ran).

    A non-EngineError (`RuntimeError`) on purpose: EngineError and GuardError
    are the two the adapter handles itself, so neither would reach the
    isolation guard under test."""
    lemonade = FakeLemonade()
    comfy = FakeComfy(raise_on_free=RuntimeError("docker proxy returned junk"))
    snapshot = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free < footprint
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
        default_route="extra.model.gguf",
    )
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(snapshot),
        FakeRegistry(footprints={"model.gguf": 19 * GIB}),
        _policy(), lemonade=lemonade, comfy=comfy,
    )

    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.count("action-failed") == 1     # reported, not swallowed
    assert comfy.freed == 0                       # the free really did fail
    assert lemonade.loaded == []                  # ...and nothing was loaded


def test_repeated_identical_action_failures_are_deduped(tmp_path):
    """FIX ROUND 1 (M-1): `action-failed` belongs to the FAILURE memo, not
    the one-slot `_last_event_key`.

    A structural mis-declaration fails identically on every ~2 s tick, and a
    contention tick emits a `noop` alongside it — the exact interleaving
    shape `_FAILURE_DEDUP_KINDS`' own comment records as what defeats a
    global last-event check (`test_repeated_identical_load_failures_are_deduped`
    pins the same property for load-failed). Under `_DEDUP_KINDS` the pair
    alternated and BOTH logged every tick, forever: the events-trim-thrash
    input [T10 review], from the guard that exists to make failures visible."""
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
    )
    actions = [{"type": "noop", "reason": "wont-fit"},
               {"type": "unload", "resource": "gguf-a", "model": "x.gguf"}]

    watcher._execute(actions, None)
    watcher._execute(actions, None)
    settled = len(tail_events(events_path))
    watcher._execute(actions, None)

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.count("action-failed") == 1
    # ...and the pair SETTLES: by the third identical tick nothing is
    # written at all. The one-slot `_last_event_key` still pays for the
    # interleaving once (the second tick's `noop` no longer matches its
    # neighbour, so it logs again) — that is this memo's own documented
    # weakness, and it is bounded here rather than repeating forever, which
    # is what it did while `action-failed` shared that slot.
    assert len(tail_events(events_path)) == settled
    assert kinds.count("noop") == 2


def test_action_failure_after_a_successful_actuation_is_logged_again(tmp_path):
    """The other half of the failure memo's contract, and the reason it is
    the right home: fail -> recover -> identical fail must log BOTH. A
    successful unload re-arms the memo (`_clear_failure_dedup`), so the
    flap-blindness the T9-fix review forbade cannot creep in behind this
    move."""
    lemonade = FakeLemonade()
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        lemonade=lemonade,
    )
    bad = {"type": "unload", "resource": "gguf-a", "model": "x.gguf"}
    good = {"type": "unload", "resource": "lemonade", "model": "extra.m.gguf"}

    watcher._execute([bad], None)          # fail #1 -> logged
    watcher._execute([good], None)         # a real unload succeeds: re-arm
    watcher._execute([bad], None)          # fail #2, identical -> must log

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert kinds.count("action-failed") == 2
    assert lemonade.unloaded == ["extra.m.gguf"]


def test_a_signature_mismatched_action_aborts_alone_not_the_whole_tick(
        tmp_path, hand_built_registry):
    """The T7 review's exact case, which is what made per-action isolation
    this task's obligation: `_execute` dispatches positionally, and a kind
    whose actuator does not take the keyword-only `node_id` a remote action
    must carry raises TypeError — not EngineError, so no adapter's own
    narrow handler catches it. Unisolated that killed the WHOLE tick:
    every remaining action plus the reconcile pass.

    Built with a lemonade-kind engine declared remotely (hand-built past the
    write gate, which refuses that kind off-box) precisely because it is the
    mismatch: the E1 kinds' actuators are local-only by signature. The
    reconcile pass running afterwards is the proof the tick survived."""
    from app.intent import IntentStore

    entries = [
        {"id": "local", "label": "This Box", "agent_kind": "local",
         "control": "none", "engines": [_TWO_GGUF_ENGINES[1]]},   # gguf-b
        {"id": "nimbus", "label": "Nimbus Box", "agent_kind": "node-agent",
         "address": "http://nimbus:7720", "control": "none",
         "engines": [dict(_REMOTE_ENGINE)]},                      # gguf-r
    ]
    registry_store = hand_built_registry(entries, {"nimbus": "key-nimbus"})
    litellm = _FakeLiteLLMRoutes()
    footprints = FakeRegistry(footprints={"b.gguf": 5 * GIB})
    clients = _FakeLocalClients(
        {"gguf-b": _PendingCapableLemonade(litellm, footprints, loaded="b.gguf")})
    intent_store = IntentStore(tmp_path / "intent.json")
    from app.node_clients import RemoteEngineClients

    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), footprints, _policy(),
        node_store=registry_store, local_clients=clients,
        intent_store=intent_store, litellm=litellm,
        remote_engine_clients=RemoteEngineClients(
            registry_store, lambda entry, credential, engine: _FakeOmni()),
    )

    actuated = watcher._execute(
        [{"type": "unload", "resource": "gguf-r", "node": "nimbus",
          "model": None},
         {"type": "unload", "resource": "gguf-b", "model": "b.gguf"}],
        None,
    )

    failed = [e["detail"] for e in tail_events(events_path)
              if e["kind"] == "action-failed"]
    assert [(f["node"], f["resource"]) for f in failed] == [("nimbus", "gguf-r")]
    assert "node_id" in failed[0]["error"]
    # The action after it actuated, and its key came back for the caller's
    # own reconcile-suppression set — i.e. `_execute` returned normally.
    assert clients.fake("gguf-b").unloaded == ["b.gguf"]
    assert actuated == {"local/gguf-b"}
