"""Tests for app.state — the World snapshot assembler.

World.snapshot() takes already-fetched GPU data (a read_gpus() result) plus
the four engine clients and the footprint registry, and assembles one
point-in-time WorldState dict. It owns no I/O of its own; all stubs below
are plain classes standing in for app.engines.* / app.registry.Registry so
these tests exercise only app.state's assembly/aggregation logic.

A World instance keeps two bits of in-memory state across snapshot() calls
(lemonade's last-seen activity counter/timestamp, comfyui's last-busy
timestamp) so idle clocks are tested by driving a fake, manually-advanced
clock rather than sleeping in real time.
"""

from app.engines import EngineError
from app.state import World

# --- stubs -------------------------------------------------------------


class FakeClock:
    """Injectable clock: starts at 0.0, advances only when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, secs: float) -> None:
        self._now += secs


class StubLemonade:
    def __init__(self, loaded: str | None = None, activity: int | None = 0) -> None:
        self._loaded = loaded
        self._activity = activity

    def status(self) -> dict:
        return {"loaded": self._loaded}

    def activity(self) -> int | None:
        return self._activity


class RaisingLemonade:
    def status(self) -> dict:
        raise EngineError("lemonade unreachable")

    def activity(self) -> int | None:
        return None  # activity() never raises per app.engines.lemonade contract


class StubComfy:
    def __init__(self, queue: int = 0) -> None:
        self._queue = queue

    def queue_len(self) -> int:
        return self._queue


class RaisingComfy:
    def queue_len(self) -> int:
        raise EngineError("comfyui unreachable")


class StubHipfire:
    def __init__(self, state: str = "parked") -> None:
        self._state = state
        self.stats_body = {"queue_depth": 0, "requests_served": 0}
        self.stats_error = None
        self.stats_calls = 0

    def status(self) -> str:
        return self._state

    def stats(self) -> dict:
        self.stats_calls += 1
        if self.stats_error is not None:
            raise self.stats_error
        return dict(self.stats_body)


class RaisingHipfire:
    def status(self) -> str:
        raise EngineError("hipfire unreachable")

    def stats(self) -> dict:
        raise EngineError("hipfire unreachable")


class StubLiteLLM:
    def __init__(self, routes: dict | None = None) -> None:
        self._routes = routes or {}

    def route_table(self) -> dict:
        return dict(self._routes)


class RaisingLiteLLM:
    def route_table(self) -> dict:
        raise EngineError("litellm unreachable")


class StubRegistry:
    def __init__(self, footprints: dict | None = None, missing: tuple = ()) -> None:
        self._footprints = footprints or {}
        self._missing = set(missing)

    def footprint(self, model_file: str) -> int:
        if model_file in self._missing:
            raise FileNotFoundError(model_file)
        return self._footprints.get(model_file, 0)


def _gpu(index=0, total=34_000_000_000, used=0, pids=None):
    return {"index": index, "vram_total": total, "vram_used": used, "pids": pids or {}}


def _healthy_kwargs(**overrides):
    kwargs = dict(
        gpus=[_gpu()],
        lemonade=StubLemonade(),
        comfy=StubComfy(),
        hipfire=StubHipfire(),
        litellm=StubLiteLLM(),
        registry=StubRegistry(),
    )
    kwargs.update(overrides)
    return kwargs


# --- snapshot shape ------------------------------------------------------


def test_snapshot_has_expected_top_level_keys():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs())

    assert set(result) == {"gpus", "tenants", "externals", "default_route"}
    assert set(result["tenants"]) == {"lemonade", "comfyui", "hipfire"}


def test_snapshot_gpus_computes_free_from_total_minus_used():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, total=34_000_000_000, used=10_000_000_000)]

    result = world.snapshot(**_healthy_kwargs(gpus=gpus))

    assert result["gpus"] == [
        {"index": 0, "total": 34_000_000_000, "used": 10_000_000_000, "free": 24_000_000_000}
    ]


def test_snapshot_gpus_handles_multiple_gpus_in_order():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, total=100, used=10), _gpu(index=1, total=200, used=20)]

    result = world.snapshot(**_healthy_kwargs(gpus=gpus))

    assert [g["index"] for g in result["gpus"]] == [0, 1]


# --- per-engine EngineError -> tenant "unknown" ---------------------------


def test_lemonade_engineerror_sets_tenant_unknown_with_none_fields():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(lemonade=RaisingLemonade()))

    assert result["tenants"]["lemonade"] == {
        "state": "unknown",
        "model": None,
        "footprint": None,
        "idle_s": None,
    }


def test_comfyui_engineerror_sets_tenant_unknown_with_none_fields():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(comfy=RaisingComfy()))

    assert result["tenants"]["comfyui"] == {"state": "unknown", "queue": None, "idle_s": None}


def test_hipfire_engineerror_sets_tenant_unknown_with_zero_footprint():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(hipfire=RaisingHipfire()))

    assert result["tenants"]["hipfire"] == {
        "state": "unknown", "model": None, "footprint": 0, "queue_depth": None
    }


def test_snapshot_itself_never_raises_when_all_engines_fail():
    world = World(clock=FakeClock())

    result = world.snapshot(
        **_healthy_kwargs(
            lemonade=RaisingLemonade(),
            comfy=RaisingComfy(),
            hipfire=RaisingHipfire(),
            litellm=RaisingLiteLLM(),
        )
    )

    assert result["tenants"]["lemonade"]["state"] == "unknown"
    assert result["tenants"]["comfyui"]["state"] == "unknown"
    assert result["tenants"]["hipfire"]["state"] == "unknown"
    assert result["default_route"] is None


def test_bare_keyerror_from_malformed_body_propagates():
    class MalformedLemonade:
        def status(self):
            return {}  # missing "loaded" key entirely -> KeyError, not EngineError

        def activity(self):
            return None

    world = World(clock=FakeClock())

    try:
        world.snapshot(**_healthy_kwargs(lemonade=MalformedLemonade()))
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError to propagate for malformed body")


# --- lemonade state/model/footprint --------------------------------------


def test_lemonade_state_unloaded_when_nothing_loaded():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(loaded=None)))

    tenant = result["tenants"]["lemonade"]
    assert tenant["state"] == "unloaded"
    assert tenant["model"] is None
    assert tenant["footprint"] is None


def test_lemonade_footprint_strips_leading_extra_prefix_before_registry_lookup():
    world = World(clock=FakeClock())
    registry = StubRegistry(footprints={"Qwen3.5-27B-Q4_K_M.gguf": 27_000_000_000})
    lemonade = StubLemonade(loaded="extra.Qwen3.5-27B-Q4_K_M.gguf")

    result = world.snapshot(**_healthy_kwargs(lemonade=lemonade, registry=registry))

    tenant = result["tenants"]["lemonade"]
    assert tenant["state"] == "loaded"
    assert tenant["model"] == "extra.Qwen3.5-27B-Q4_K_M.gguf"
    assert tenant["footprint"] == 27_000_000_000


def test_lemonade_footprint_none_when_registry_raises_filenotfounderror():
    world = World(clock=FakeClock())
    registry = StubRegistry(missing={"missing.gguf"})
    lemonade = StubLemonade(loaded="extra.missing.gguf")

    result = world.snapshot(**_healthy_kwargs(lemonade=lemonade, registry=registry))

    assert result["tenants"]["lemonade"]["footprint"] is None


# --- lemonade idle clock ---------------------------------------------------


def test_lemonade_idle_s_zero_on_first_snapshot():
    clock = FakeClock(start=1000.0)
    world = World(clock=clock)

    result = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=5)))

    assert result["tenants"]["lemonade"]["idle_s"] == 0


def test_lemonade_idle_s_advances_only_while_counter_is_static():
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))
    clock.advance(30)
    second = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))
    clock.advance(30)
    third = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))

    assert second["tenants"]["lemonade"]["idle_s"] == 30
    assert third["tenants"]["lemonade"]["idle_s"] == 60


def test_lemonade_idle_s_resets_when_counter_changes():
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))
    clock.advance(45)
    world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))
    clock.advance(5)
    result = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=999)))

    assert result["tenants"]["lemonade"]["idle_s"] == 0


def test_lemonade_idle_s_none_when_activity_returns_none():
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=10)))
    clock.advance(30)
    result = world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(activity=None)))

    assert result["tenants"]["lemonade"]["idle_s"] is None


# --- comfyui state/idle clock ----------------------------------------------


def test_comfy_state_idle_when_queue_empty():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))

    assert result["tenants"]["comfyui"] == {"state": "idle", "queue": 0, "idle_s": 0}


def test_comfy_state_busy_when_queue_nonempty():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=3)))

    assert result["tenants"]["comfyui"]["state"] == "busy"
    assert result["tenants"]["comfyui"]["queue"] == 3
    assert result["tenants"]["comfyui"]["idle_s"] == 0


def test_comfy_idle_s_grows_after_queue_drains():
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=2)))  # busy, last_activity=0
    clock.advance(20)
    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))  # now idle

    assert result["tenants"]["comfyui"]["state"] == "idle"
    assert result["tenants"]["comfyui"]["idle_s"] == 20


def test_comfy_idle_s_resets_when_queue_becomes_nonempty_again():
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))
    clock.advance(50)
    world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))
    clock.advance(5)
    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=1)))

    assert result["tenants"]["comfyui"]["state"] == "busy"
    assert result["tenants"]["comfyui"]["idle_s"] == 0


# --- hipfire state/model/footprint -----------------------------------------


def test_hipfire_footprint_is_hipfire_footprint_constant_when_running():
    from app.registry import HIPFIRE_FOOTPRINT

    world = World(clock=FakeClock())
    routes = {"hipfire": "openai/qwen36-35b-a3b.mq4"}

    result = world.snapshot(**_healthy_kwargs(hipfire=StubHipfire("running"), litellm=StubLiteLLM(routes)))

    tenant = result["tenants"]["hipfire"]
    assert tenant["state"] == "running"
    assert tenant["model"] == "qwen36-35b-a3b.mq4"
    assert tenant["footprint"] == HIPFIRE_FOOTPRINT


def test_hipfire_footprint_zero_when_not_running():
    for state in ("parked", "loading"):
        result = World(clock=FakeClock()).snapshot(**_healthy_kwargs(hipfire=StubHipfire(state)))
        assert result["tenants"]["hipfire"]["footprint"] == 0
        assert result["tenants"]["hipfire"]["state"] == state


def test_hipfire_snapshot_includes_queue_depth_when_running():
    """The snapshot polls /stats while running — this is also what feeds the
    HipfireClient conversation-activity tracker every watcher tick."""
    hipfire = StubHipfire("running")
    hipfire.stats_body = {"queue_depth": 2, "requests_served": 9}

    result = World(clock=FakeClock()).snapshot(**_healthy_kwargs(hipfire=hipfire))

    assert result["tenants"]["hipfire"]["queue_depth"] == 2
    assert hipfire.stats_calls == 1


def test_hipfire_snapshot_skips_stats_when_not_running():
    for state in ("parked", "loading"):
        hipfire = StubHipfire(state)
        result = World(clock=FakeClock()).snapshot(**_healthy_kwargs(hipfire=hipfire))
        assert result["tenants"]["hipfire"]["queue_depth"] is None
        assert hipfire.stats_calls == 0


def test_hipfire_snapshot_queue_depth_none_when_stats_raises():
    """/stats failing must not take down the snapshot: state stays honest,
    queue_depth is simply unknown."""
    hipfire = StubHipfire("running")
    hipfire.stats_error = EngineError("stats unreachable")

    result = World(clock=FakeClock()).snapshot(**_healthy_kwargs(hipfire=hipfire))

    assert result["tenants"]["hipfire"]["state"] == "running"
    assert result["tenants"]["hipfire"]["queue_depth"] is None


def test_hipfire_model_strips_openai_prefix():
    world = World(clock=FakeClock())
    routes = {"hipfire": "openai/qwen36-35b-a3b.mq4"}

    result = world.snapshot(**_healthy_kwargs(hipfire=StubHipfire("parked"), litellm=StubLiteLLM(routes)))

    assert result["tenants"]["hipfire"]["model"] == "qwen36-35b-a3b.mq4"


def test_hipfire_model_none_when_route_table_has_no_hipfire_entry():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(hipfire=StubHipfire("running"), litellm=StubLiteLLM({})))

    assert result["tenants"]["hipfire"]["model"] is None


def test_hipfire_state_preserved_when_litellm_fails_but_model_is_none():
    """A litellm failure means we can't learn hipfire's *model name*, but
    hipfire's own operational state came from HipfireClient.status()
    independently and should not be downgraded to "unknown" just because
    the unrelated litellm call failed."""
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(hipfire=StubHipfire("running"), litellm=RaisingLiteLLM()))

    tenant = result["tenants"]["hipfire"]
    assert tenant["state"] == "running"
    assert tenant["model"] is None
    from app.registry import HIPFIRE_FOOTPRINT

    assert tenant["footprint"] == HIPFIRE_FOOTPRINT


# --- default_route ----------------------------------------------------------


def test_default_route_strips_openai_prefix():
    world = World(clock=FakeClock())
    routes = {"default": "openai/extra.Qwen3.5-27B-Q4_K_M.gguf"}

    result = world.snapshot(**_healthy_kwargs(litellm=StubLiteLLM(routes)))

    assert result["default_route"] == "extra.Qwen3.5-27B-Q4_K_M.gguf"


def test_default_route_none_when_missing_from_route_table():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(litellm=StubLiteLLM({})))

    assert result["default_route"] is None


def test_default_route_none_on_litellm_engineerror():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(litellm=RaisingLiteLLM()))

    assert result["default_route"] is None


# --- externals: minimal heuristic -------------------------------------------


_ONE_GIB = 1024**3


def test_externals_flags_fat_pid_when_nothing_loaded_anywhere():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, pids={4242: 2 * _ONE_GIB})]

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded=None),
            comfy=StubComfy(queue=0),
            hipfire=StubHipfire("parked"),
        )
    )

    assert result["externals"] == [{"pid": 4242, "gpu": 0, "bytes": 2 * _ONE_GIB}]


def test_externals_ignores_pids_at_or_below_one_gib_floor():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, pids={4242: _ONE_GIB})]  # exactly at floor, not above

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded=None),
            comfy=StubComfy(queue=0),
            hipfire=StubHipfire("parked"),
        )
    )

    assert result["externals"] == []


def test_externals_empty_when_lemonade_reports_loaded():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, pids={4242: 2 * _ONE_GIB})]

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded="extra.some-model.gguf"),
            comfy=StubComfy(queue=0),
            hipfire=StubHipfire("parked"),
        )
    )

    assert result["externals"] == []


def test_externals_empty_when_hipfire_running():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, pids={4242: 2 * _ONE_GIB})]

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded=None),
            comfy=StubComfy(queue=0),
            hipfire=StubHipfire("running"),
        )
    )

    assert result["externals"] == []


def test_externals_empty_when_comfyui_busy():
    world = World(clock=FakeClock())
    gpus = [_gpu(index=0, pids={4242: 2 * _ONE_GIB})]

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded=None),
            comfy=StubComfy(queue=1),
            hipfire=StubHipfire("parked"),
        )
    )

    assert result["externals"] == []


def test_externals_across_multiple_gpus():
    world = World(clock=FakeClock())
    gpus = [
        _gpu(index=0, pids={100: 2 * _ONE_GIB}),
        _gpu(index=1, pids={200: 3 * _ONE_GIB}),
    ]

    result = world.snapshot(
        **_healthy_kwargs(
            gpus=gpus,
            lemonade=StubLemonade(loaded=None),
            comfy=StubComfy(queue=0),
            hipfire=StubHipfire("parked"),
        )
    )

    assert sorted(result["externals"], key=lambda e: e["pid"]) == [
        {"pid": 100, "gpu": 0, "bytes": 2 * _ONE_GIB},
        {"pid": 200, "gpu": 1, "bytes": 3 * _ONE_GIB},
    ]
