"""Tests for app.arbiter — the priority arbiter (pure ``decide``) + watcher.

``decide(world, policy, pending_load) -> list[dict]`` is a pure function: no
I/O, deterministic, and every guard below is tested for the ABSENCE of the
dangerous action, not merely the presence of the safe one — this module
decides real GPU unloads on a live box.

Action dicts:
    {"type": "unload_lemonade", "model": <str>}
    {"type": "free_comfyui"}
    {"type": "noop", "reason": <str>}

The Watcher tests drive one ``tick()`` at a time against stub deps (no real
thread needed for behavior coverage); a single thread test proves
start()/stop() runs at least one tick and joins promptly.
"""

import json
import threading
import time

from app.arbiter import HealSuppressor, Watcher, decide
from app.characteristics import CharacteristicsStore
from app.engines import EngineError, GuardError
from app.events import tail_events

# The reconcile pass is the first watcher code to call the hipfire client, so
# it needs a real fake rather than the bare object() the arbitration tests
# pass. Reuse test_api's rather than growing a second, subtly different one.
from tests.test_api import FakeHipfire

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
    return {
        "gpus": gpus if gpus is not None else [_gpu()],
        "tenants": {
            "lemonade": lemonade if lemonade is not None else _lem(),
            "comfyui": comfyui if comfyui is not None else _comfy(),
            "hipfire": hipfire if hipfire is not None else _hip(),
        },
        "externals": [],
        "default_route": default_route,
        "placement": placement
        if placement is not None
        else {"hipfire": 0, "lemonade": 1, "comfyui": 1},
    }


def _policy(lem_pinned=False, lem_idle=900, comfy_pinned=False, comfy_idle=300):
    return {
        "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
        "lemonade": {"priority": 50, "pinned": lem_pinned, "idle_ttl": lem_idle},
        "comfyui": {"priority": 40, "pinned": comfy_pinned, "idle_ttl": comfy_idle},
    }


_VALID_ACTION_TYPES = {"unload_lemonade", "free_comfyui", "noop"}


def _assert_only_valid_actions(actions):
    """No action type may ever target hipfire or an external — the only way
    to prove that guard is to assert by omission across every result."""
    for a in actions:
        assert a["type"] in _VALID_ACTION_TYPES, a


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

    assert result == [{"type": "unload_lemonade", "model": model}]
    _assert_only_valid_actions(result)


def test_idle_release_no_unload_when_lemonade_pinned():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=5000),
    )

    result = decide(world, _policy(lem_pinned=True), None)

    assert "unload_lemonade" not in _types(result)


def test_idle_release_no_unload_when_idle_ttl_zero():
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=99999),
    )

    result = decide(world, _policy(lem_idle=0), None)

    assert "unload_lemonade" not in _types(result)


def test_idle_release_no_unload_when_idle_s_none():
    """idle_s None means activity metrics are unavailable, which is NOT the
    same as idle — never release on missing telemetry."""
    world = _world(
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=None),
    )

    result = decide(world, _policy(lem_idle=900), None)

    assert "unload_lemonade" not in _types(result)


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

    assert result == [{"type": "free_comfyui"}]
    _assert_only_valid_actions(result)


def test_idle_release_no_free_when_comfy_holds_no_vram():
    """TTL expiry alone must not free: with comfy's GPU essentially empty
    (used below the slack allowance), the free is a guaranteed no-op that
    would just re-arm the TTL and spam the event ring forever."""
    world = _world(
        gpus=[_gpu(index=1, used=60 * 1024**2)],
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free_comfyui" not in _types(result)


def test_idle_release_no_free_when_gpu_usage_is_lemonades():
    """Usage fully accounted for by a loaded lemonade on the same GPU is not
    comfy's to reclaim — freeing comfy would be a no-op."""
    world = _world(
        gpus=[_gpu(index=1, used=10 * GIB)],
        lemonade=_lem(state="loaded", model="extra.m.gguf", footprint=10 * GIB, idle_s=10),
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free_comfyui" not in _types(result)


def test_idle_release_frees_comfyui_when_gpu_unresolvable():
    """Placement pointing at a GPU we have no telemetry for must NOT suppress
    the free — unknown usage is not proof there's nothing to reclaim."""
    world = _world(
        gpus=[_gpu(index=0, used=0)],
        comfyui=_comfy(state="idle", queue=0, idle_s=301),
        placement={"hipfire": 0, "lemonade": 1, "comfyui": 1},
    )

    result = decide(world, _policy(comfy_idle=300), None)

    assert result == [{"type": "free_comfyui"}]


def test_idle_release_no_free_when_comfy_pinned():
    world = _world(comfyui=_comfy(state="idle", queue=0, idle_s=5000))

    result = decide(world, _policy(comfy_pinned=True), None)

    assert "free_comfyui" not in _types(result)


def test_idle_release_no_free_when_comfy_busy():
    """A busy comfy (queue>0) is never freed even long 'past' a TTL."""
    world = _world(comfyui=_comfy(state="busy", queue=3, idle_s=5000))

    result = decide(world, _policy(comfy_idle=300), None)

    assert "free_comfyui" not in _types(result)


def test_idle_release_no_free_when_comfy_idle_ttl_zero():
    world = _world(comfyui=_comfy(state="idle", queue=0, idle_s=99999))

    result = decide(world, _policy(comfy_idle=0), None)

    assert "free_comfyui" not in _types(result)


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


def _pending(model="extra.model.gguf", footprint=19 * GIB, gpu_index=1):
    return {"model": model, "footprint": footprint, "gpu_index": gpu_index}


def test_contention_todays_incident_frees_comfyui():
    """The exact incident from the brief: gpu1 free 12 GiB, comfy idle 400 s
    holding ~21 GiB, pending 27B footprint 19 GiB -> [free_comfyui]."""
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=22 * GIB)],  # free = 12 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=400),
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "free_comfyui"}]
    _assert_only_valid_actions(result)


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
    assert "free_comfyui" not in _types(result)
    assert "unload_lemonade" not in _types(result)


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
    assert "unload_lemonade" not in _types(result)


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

    assert result == [{"type": "unload_lemonade", "model": "extra.other.gguf"}]


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
    assert "unload_lemonade" not in _types(result)


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

    assert result == [{"type": "free_comfyui"}]
    assert "unload_lemonade" not in _types(result)


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

    assert result == [{"type": "free_comfyui"}, {"type": "unload_lemonade", "model": "extra.other.gguf"}]


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
    assert "free_comfyui" not in _types(result)  # comfy NOT killed pointlessly


def test_contention_pinned_comfy_not_evicted():
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="idle", queue=0, idle_s=5000),
    )

    result = decide(world, _policy(comfy_pinned=True), _pending(footprint=19 * GIB, gpu_index=1))

    assert result == [{"type": "noop", "reason": "wont-fit"}]
    assert "free_comfyui" not in _types(result)


def test_contention_busy_comfy_never_freed():
    world = _world(
        gpus=[_gpu(index=1, total=34 * GIB, used=29 * GIB)],  # free = 5 GiB
        lemonade=_lem(state="unloaded"),
        comfyui=_comfy(state="busy", queue=5, idle_s=0),
    )

    result = decide(world, _policy(), _pending(footprint=19 * GIB, gpu_index=1))

    assert "free_comfyui" not in _types(result)


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
    _assert_only_valid_actions(result)


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

    assert result == [{"type": "unload_lemonade", "model": "extra.other.gguf"}]


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

    def snapshot(self, gpus, lemonade, comfy, hipfire, litellm, registry):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._snapshot

    def note_comfy_freed(self):
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


class FakeLemonade:
    def __init__(self, raise_on_load=None, in_flight=False):
        self.unloaded = []
        self.loaded = []
        self._raise_on_load = raise_on_load
        self._in_flight = in_flight

    def load_in_flight(self):
        return self._in_flight

    def unload(self, model):
        self.unloaded.append(model)

    def load(self, model):
        if self._raise_on_load is not None:
            raise self._raise_on_load
        self.loaded.append(model)


class FakeComfy:
    def __init__(self, raise_guard=False):
        self.freed = 0
        self._raise_guard = raise_guard

    def free(self):
        if self._raise_guard:
            raise GuardError("queue raced non-empty")
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
    intent_store=None, spark=None, auto=True, characteristics_store=None,
    gguf_dir=None, clock=None, on_derive=None, engine_exec=None, **sett,
):
    events_path = tmp_path / "events.jsonl"
    watcher = Watcher(
        settings=_settings(**sett),
        world=world,
        lemonade=lemonade if lemonade is not None else FakeLemonade(),
        comfy=comfy if comfy is not None else FakeComfy(),
        hipfire=hipfire if hipfire is not None else object(),
        litellm=object(),
        registry=registry,
        policy_store=FakePolicyStore(policy, auto=auto),
        events_path=events_path,
        read_gpus=read_gpus if read_gpus is not None else RecordingReadGpus(),
        heal_suppressor=heal_suppressor,
        hostagent=hostagent,
        catalog=catalog,
        intent_store=intent_store,
        spark=spark,
        characteristics_store=characteristics_store,
        gguf_dir=gguf_dir,
        clock=clock if clock is not None else time.monotonic,
        on_derive=on_derive,
        engine_exec=engine_exec,
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
# I1 — watcher yields to an in-flight set apply
# ===========================================================================


def test_watcher_tick_noop_while_apply_holds_lock(tmp_path):
    """While a set apply holds the module lock, tick() is a clean no-op: no
    snapshot is taken and no event is logged."""
    import app.sets as sets_mod

    world = FakeWorld(_world())
    watcher, events_path = _make_watcher(tmp_path, world, FakeRegistry(), _policy())

    with sets_mod._apply_lock:
        watcher.tick()

    assert world.calls == 0  # no snapshot attempted
    assert tail_events(events_path) == []  # nothing logged


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
        self.calls.append(("swap", profile))
        if self.swap_fail is not None:
            raise self.swap_fail
        return {"id": "u1", "profile": profile}

    def models(self):
        self.models_calls += 1
        if self.models_raises is not None:
            raise self.models_raises
        return self._models_body


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
    watcher, _ = _reconcile_watcher(tmp_path, store, spark=spark)

    watcher.tick()

    assert spark.calls == [("swap", "heretic")]


def test_restore_dispatches_a_spark_swap_after_the_boot_window_expires(tmp_path):
    """swap_status stays 'done' forever after a successful swap — a model that
    died hours later must still be restored, not read as 'still booting'."""
    from datetime import UTC, datetime, timedelta

    old = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", swap_state="done", swap_ts=old)
    watcher, _ = _reconcile_watcher(tmp_path, store, spark=spark)

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
    watcher, _ = _reconcile_watcher(tmp_path, store, spark=spark)

    watcher.tick()

    assert spark.calls == []


def test_unreachable_spark_is_not_a_dead_model(tmp_path):
    """Failing to look is not the same as looking and seeing nothing."""
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(raises=EngineError("connection refused"))
    watcher, events_path = _reconcile_watcher(tmp_path, store, spark=spark)

    watcher.tick()

    assert spark.calls == []
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "tick-error" not in kinds


def test_spark_serving_the_intended_profile_is_left_alone(tmp_path):
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    spark = FakeSpark(profile="heretic", serving_model="heretic", endpoint_ok=True)
    watcher, _ = _reconcile_watcher(tmp_path, store, spark=spark)

    watcher.tick()

    assert spark.calls == []
    assert store.get()["sparky/slot0"]["last_healthy_ts"] is not None


def test_no_spark_configured_emits_no_phantom_key(tmp_path):
    store = _intent(tmp_path, key="sparky/slot0", model="heretic", engine="spark")
    watcher, events_path = _reconcile_watcher(tmp_path, store, spark=None)

    watcher.tick()

    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "lifecycle-restore" not in kinds
    assert "lifecycle-restore-failed" not in kinds


# ===========================================================================
# CHARACTERISTICS DERIVE PASS (task 6) — throttled by settings.derive_interval_s
# ===========================================================================


def test_derive_pass_runs_on_first_tick(tmp_path, monkeypatch):
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       gguf_dir=_checkpoint_tree(tmp_path))

    watcher.tick()

    assert store.get(), "expected at least one derived entry"


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
                       gguf_dir=tmp_path / "does-not-exist", spark=spark)

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
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, spark=spark)

    watcher.tick()  # must not raise

    assert store.get()["model/heretic"]["served"]["value"] is True


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
    assert actions == [{"type": "unload_lemonade", "model": "extra.other.gguf"}]

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


# ===========================================================================
# CATALOG HARVEST (task 8) — _derive_pass harvests each configurable
# engine's option catalog, once per observed engine version.
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
                       engine_exec=_recording_exec(execs, version="0.26.0"))

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(execs) == 1
    assert store.entry("engine/local/hipfire")["option_catalog"]["value"]["engine_version"] == "0.26.0"


def test_harvest_reruns_when_the_engine_version_changes(tmp_path, monkeypatch):
    store = CharacteristicsStore(tmp_path / "c.json")
    execs = []
    exec_fn = _recording_exec(execs, version="0.26.0")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, engine_exec=exec_fn)

    watcher._derive_pass()
    exec_fn.version = "0.27.0"
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(execs) == 2


def test_harvest_failure_leaves_no_catalog_and_does_not_raise(tmp_path, monkeypatch):
    """An engine that is down has no catalog. Supported state, not an error."""
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store,
                       engine_exec=_raises(EngineError("not running")))

    watcher._derive_pass()

    assert "option_catalog" not in store.entry("engine/local/hipfire")


# --- F3, Important (final branch review, 2026-08-07) -------------------------
#
# The exec-failure branch above used to be fully silent -- indistinguishable
# from "engine parked" forever, and the branch's own recorded deploy hazard
# (a hand-merged live compose losing the proxy's -allowPOST lines -> 403 on
# exec) lands exactly there. It now logs a deduped harvest-failed event.


def test_harvest_exec_failure_logs_a_deduped_harvest_failed_event(tmp_path):
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        characteristics_store=store,
        engine_exec=_raises(EngineError("403 Forbidden")))

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    failures = [e for e in tail_events(events_path) if e["kind"] == "harvest-failed"]
    assert len(failures) == 1
    assert failures[0]["detail"] == {"key": "engine/local/hipfire", "reason": "EngineError"}


def test_harvest_failed_dedup_resets_when_the_failure_kind_changes(tmp_path):
    """A different failure in between resets the suppression -- same dedup
    semantics as every other _DEDUP_KINDS event (see the tick-error dedup
    test above)."""
    store = CharacteristicsStore(tmp_path / "c.json")
    watcher, events_path = _make_watcher(
        tmp_path, FakeWorld(_world()), FakeRegistry(), _policy(),
        characteristics_store=store,
        engine_exec=_raises(EngineError("403 Forbidden")))

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
                       engine_exec=_raises(GuardError("container 'ods-hipfire' is not in the park allowlist")))

    watcher._derive_pass()  # must not raise

    assert "option_catalog" not in store.entry("engine/local/hipfire")


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

    watcher = _watcher(tmp_path=tmp_path, characteristics_store=store, engine_exec=bare_exec)

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
                       engine_exec=DockerEngineExec(dockerctl, "ods-hipfire"))

    watcher._derive_pass()
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(exec_creates) == 1  # unchanged image id -> peek skips the 2nd exec

    image_id["value"] = "sha256:new"
    watcher._last_derive_at = None
    watcher._derive_pass()

    assert len(exec_creates) == 2  # changed image id -> re-harvested
    assert (store.entry("engine/local/hipfire")["option_catalog"]["value"]["engine_version"]
            == "sha256:new")
