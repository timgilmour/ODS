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

import threading
import time

from app.arbiter import Watcher, decide
from app.engines import GuardError
from app.events import tail_events

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


def _world(gpus=None, lemonade=None, comfyui=None, hipfire=None, default_route=None):
    return {
        "gpus": gpus if gpus is not None else [_gpu()],
        "tenants": {
            "lemonade": lemonade if lemonade is not None else _lem(),
            "comfyui": comfyui if comfyui is not None else _comfy(),
            "hipfire": hipfire if hipfire is not None else _hip(),
        },
        "externals": [],
        "default_route": default_route,
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
    world = _world(comfyui=_comfy(state="idle", queue=0, idle_s=301))

    result = decide(world, _policy(comfy_idle=300), None)

    assert result == [{"type": "free_comfyui"}]
    _assert_only_valid_actions(result)


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

    def snapshot(self, gpus, lemonade, comfy, hipfire, litellm, registry):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._snapshot


class FakeRegistry:
    def __init__(self, footprints=None, missing=()):
        self._footprints = footprints or {}
        self._missing = set(missing)

    def footprint(self, key):
        if key in self._missing:
            raise FileNotFoundError(key)
        return self._footprints[key]


class FakePolicyStore:
    def __init__(self, policy):
        self._policy = policy

    def get(self):
        return self._policy


class FakeLemonade:
    def __init__(self):
        self.unloaded = []
        self.loaded = []

    def unload(self, model):
        self.unloaded.append(model)

    def load(self, model):
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


def _make_watcher(tmp_path, world, registry, policy, lemonade=None, comfy=None, read_gpus=None, **sett):
    events_path = tmp_path / "events.jsonl"
    watcher = Watcher(
        settings=_settings(**sett),
        world=world,
        lemonade=lemonade if lemonade is not None else FakeLemonade(),
        comfy=comfy if comfy is not None else FakeComfy(),
        hipfire=object(),
        litellm=object(),
        registry=registry,
        policy_store=FakePolicyStore(policy),
        events_path=events_path,
        read_gpus=read_gpus if read_gpus is not None else RecordingReadGpus(),
    )
    return watcher, events_path


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
    kinds = [e["kind"] for e in tail_events(events_path)]
    assert "free_comfyui" in kinds
    assert read_gpus.calls  # read_gpus was actually invoked with the sysfs roots


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


def test_watcher_tick_error_swallowed_and_loop_survives(tmp_path):
    world = FakeWorld(snapshot=None, raises=KeyError("boom"))
    watcher, events_path = _make_watcher(tmp_path, world, FakeRegistry(), _policy())

    watcher.tick()  # first tick errors...
    watcher.tick()  # ...and the second still runs (loop survives)

    events = tail_events(events_path)
    tick_errors = [e for e in events if e["kind"] == "tick-error"]
    assert len(tick_errors) == 2
    assert "boom" in tick_errors[0]["detail"]["error"]


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
