"""Tests for app.state — the World snapshot assembler.

World.snapshot() takes already-fetched GPU data (a read_gpus() result) plus
a declaration (`engines`), a LocalClients-shaped `clients` (E1 Task 3: see
app.local_clients.LocalClients — `_FakeLocalClients` below stands in for
it, per the task brief's fixture), the litellm client and the footprint
registry, and assembles one point-in-time WorldState dict. It owns no I/O
of its own; all stubs below are plain classes standing in for
app.engines.* / app.registry.Registry so these tests exercise only
app.state's assembly/aggregation logic.

These tests deliberately keep `_ENGINES`' resource names equal to their
kind names ("lemonade"/"comfyui"/"hipfire") — this file's job is pinning
each KIND's per-engine logic (the moved-verbatim observe() bodies: registry
footprint lookup, HIPFIRE_FOOTPRINT, litellm route-table model lookup),
which is not generic across kinds, so a resource!=kind name would buy
nothing here. The generalization itself (resource, not kind, keys
everything; two same-kind resources are independent) is what
tests/test_engine_kinds.py's Step-1 tests pin, with fixture-rule names.

A World instance keeps one piece of in-memory state across snapshot()
calls — `self._mem`, keyed by resource (lemonade's last-seen activity
counter/timestamp/loaded model, comfyui's last-busy timestamp) — so idle
clocks are tested by driving a fake, manually-advanced clock rather than
sleeping in real time.
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

    def load_in_flight(self) -> bool:
        return False

    def status(self) -> dict:
        return {"loaded": self._loaded}

    def activity(self) -> int | None:
        return self._activity


class RaisingLemonade:
    def load_in_flight(self) -> bool:
        return False

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


class _FakeLocalClients:
    """Stand-in for app.local_clients.LocalClients: a plain dict wrapper
    (the task brief's prescribed shape — `client_for` only)."""

    def __init__(self, clients: dict) -> None:
        self._clients = clients

    def client_for(self, resource: str):
        return self._clients.get(resource)


# The declaration these tests snapshot against: resource names equal kind
# names (see module docstring for why), gpu_index values matching the old
# _DEFAULT_PLACEMENT this generalizes away from (hipfire=0, lemonade=1,
# comfyui=1) so test_snapshot_includes_default_placement's assertion (a
# pre-E1 holdover, now proving placement DERIVES from the declaration
# rather than a World constructor arg) needs no numeric changes.
_ENGINES = [
    {"resource": "lemonade", "kind": "lemonade",
     "connection": {"url": "u", "metrics_url": "m", "container": "c"},
     "gpu_index": 1,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
    {"resource": "comfyui", "kind": "comfyui",
     "connection": {"url": "u"}, "gpu_index": 1,
     "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
    {"resource": "hipfire", "kind": "hipfire",
     "connection": {"container": "c"}, "gpu_index": 0,
     "policy_defaults": {"priority": 100, "pinned": True, "idle_ttl": 0}},
]


def _healthy_kwargs(**overrides):
    """Builds World.snapshot()'s kwargs from the SAME per-engine-client
    overrides every test below already passes (`lemonade=`, `comfy=`,
    `hipfire=`, plus `gpus=`/`litellm=`/`registry=`) — only this helper's
    internals changed for the new (engines, clients) interface, so every
    call site in this file (and every assertion against
    `result["tenants"]["lemonade"|"comfyui"|"hipfire"]`) is untouched."""
    kwargs = dict(
        gpus=[_gpu()],
        lemonade=StubLemonade(),
        comfy=StubComfy(),
        hipfire=StubHipfire(),
        litellm=StubLiteLLM(),
        registry=StubRegistry(),
    )
    kwargs.update(overrides)
    return dict(
        gpus=kwargs["gpus"],
        engines=_ENGINES,
        clients=_FakeLocalClients({
            "lemonade": kwargs["lemonade"],
            "comfyui": kwargs["comfy"],
            "hipfire": kwargs["hipfire"],
        }),
        litellm=kwargs["litellm"],
        registry=kwargs["registry"],
    )


# --- snapshot shape ------------------------------------------------------


def test_snapshot_has_expected_top_level_keys():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs())

    assert set(result) == {"gpus", "tenants", "externals", "default_route",
                           "routes_known", "placement"}
    assert set(result["tenants"]) == {"lemonade", "comfyui", "hipfire"}


# --- placement ---------------------------------------------------------


def test_snapshot_derives_placement_from_declared_engines():
    # Pre-E1 this pinned World's own placement CONSTRUCTOR ARG default
    # (_DEFAULT_PLACEMENT); E1 Task 3 kills that arg — placement is now
    # DERIVED from each declared entry's own gpu_index, so this now proves
    # the derivation, not a hardcoded default (_ENGINES' own gpu_index
    # values happen to equal the old default, see this file's _ENGINES
    # comment).
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs())

    # placement (INST I1 Task 2, D-I1-2): now the declared GPU LIST, even
    # for a legacy scalar declaration.
    assert result["placement"] == {"hipfire": [0], "lemonade": [1], "comfyui": [1]}


def test_snapshot_placement_follows_declared_gpu_index_not_a_fixed_default():
    # Was test_snapshot_respects_custom_placement (World(placement=...)).
    # E1 Task 3 kills that constructor arg entirely: "custom placement" is
    # now just "a different declaration", proven by re-declaring the same
    # three resources on different GPUs.
    world = World(clock=FakeClock())
    engines = [{**e, "gpu_index": {"hipfire": 2, "lemonade": 0, "comfyui": 0}[e["resource"]]}
               for e in _ENGINES]
    kwargs = _healthy_kwargs()
    kwargs["engines"] = engines

    result = world.snapshot(**kwargs)

    # placement (INST I1 Task 2, D-I1-2): now the declared GPU LIST.
    assert result["placement"] == {"hipfire": [2], "lemonade": [0], "comfyui": [0]}


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

    # gains "engine"/"gpu_index"/"gpu_indices" (E1 Task 3 stamped
    # engine/gpu_index; INST I1 Task 2 adds gpu_indices, D-I1-2's list
    # spelling — a one-element list for this legacy scalar declaration)
    # alongside the unchanged fields.
    assert result["tenants"]["lemonade"] == {
        "state": "unknown",
        "model": None,
        "footprint": None,
        "idle_s": None,
        "engine": "lemonade",
        "gpu_index": 1,
        "gpu_indices": [1],
    }


def test_comfyui_engineerror_sets_tenant_unknown_with_none_fields():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(comfy=RaisingComfy()))

    assert result["tenants"]["comfyui"] == {
        "state": "unknown", "queue": None, "idle_s": None,
        "engine": "comfyui", "gpu_index": 1, "gpu_indices": [1],
    }


def test_hipfire_engineerror_sets_tenant_unknown_with_zero_footprint():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(hipfire=RaisingHipfire()))

    assert result["tenants"]["hipfire"] == {
        "state": "unknown", "model": None, "footprint": 0, "queue_depth": None,
        "engine": "hipfire", "gpu_index": 0, "gpu_indices": [0],
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
        def load_in_flight(self):
            return False

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


def test_snapshot_reports_loading_while_a_load_is_in_flight():
    class LoadingLemonade:
        def load_in_flight(self):
            return True

        def status(self):
            raise AssertionError("status() must not be consulted mid-load")

        def activity(self):
            return None

    world = World()

    result = world.snapshot(**_healthy_kwargs(lemonade=LoadingLemonade()))

    assert result["tenants"]["lemonade"] == {
        "state": "loading", "model": None, "footprint": None, "idle_s": None,
        "engine": "lemonade", "gpu_index": 1, "gpu_indices": [1],
    }


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


def test_lemonade_idle_s_resets_on_load_even_when_counter_value_repeats():
    # llama.cpp's token counters restart at 0 on every load, so a fresh load
    # reports the same counter value (0) the previous resident model ended
    # with. The load transition itself must reset the idle clock — otherwise
    # the new model inherits the hours-old clock and the arbiter evicts it on
    # the first tick after load.
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(
        **_healthy_kwargs(lemonade=StubLemonade(loaded="extra.a.gguf", activity=0))
    )
    clock.advance(7 * 3600)  # idle-released hours ago; llama process gone
    world.snapshot(**_healthy_kwargs(lemonade=StubLemonade(loaded=None, activity=None)))
    clock.advance(60)
    result = world.snapshot(
        **_healthy_kwargs(lemonade=StubLemonade(loaded="extra.b.gguf", activity=0))
    )

    assert result["tenants"]["lemonade"]["idle_s"] == 0


# --- comfyui state/idle clock ----------------------------------------------


def test_comfy_state_idle_when_queue_empty():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))

    assert result["tenants"]["comfyui"] == {
        "state": "idle", "queue": 0, "idle_s": 0, "engine": "comfyui",
        "gpu_index": 1, "gpu_indices": [1],
    }


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


def test_comfy_idle_clock_rearms_when_freed():
    # A successful comfy free re-arms the idle TTL. Without this, idle_s only
    # grows once comfy is idle, so the idle-release rule re-fires
    # free_comfyui on every watcher tick (2s) for as long as comfy stays
    # idle — flooding the event ring and spamming comfy's /free endpoint.
    clock = FakeClock(start=0.0)
    world = World(clock=clock)

    world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))
    clock.advance(500)  # idle well past a 300s TTL
    world.note_freed("comfyui")  # was note_comfy_freed() (E1 Task 3)
    clock.advance(30)
    result = world.snapshot(**_healthy_kwargs(comfy=StubComfy(queue=0)))

    assert result["tenants"]["comfyui"]["idle_s"] == 30


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


# --- routes_known: "None default_route" is ambiguous without it -------------


def test_routes_known_true_when_litellm_answers():
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(litellm=StubLiteLLM({})))

    assert result["routes_known"] is True


def test_routes_known_false_on_litellm_engineerror():
    """default_route=None means two different things — "no default route is
    configured" and "we could not reach litellm to ask". Storage guards must
    fail closed on the second, so the snapshot distinguishes them."""
    world = World(clock=FakeClock())

    result = world.snapshot(**_healthy_kwargs(litellm=RaisingLiteLLM()))

    assert result["default_route"] is None
    assert result["routes_known"] is False


# ===========================================================================
# sglang-omni Task 6 — World.snapshot_remote: DECLARED engines on registry
# entries OTHER than the local one.
#
# Kept in its OWN half of the World (remote_tenants / a per-node GPU pool),
# never merged into `tenants`/`gpus`: a remote engine's gpu_index addresses
# ITS OWN node's GPU list, so folding remote tenants into the local
# resource-keyed map would make the arbiter's co-residency and eviction
# arithmetic (which matches `tenant["gpu_index"]` against the LOCAL gpu
# list) silently compare two different machines' GPU 0.
#
# Fixture discipline ([[defaults-that-hide-bugs]]): node id "nimbus" (NOT
# "sparky"), resource "gguf-r" (NOT "omni"), resource != kind name,
# gpu_index 4 (past every local fixture's 0/1/2/3).
# ===========================================================================

_REMOTE_ENGINES = [
    {"resource": "gguf-r", "kind": "lemonade", "node_id": "nimbus",
     "connection": {"url": "http://gguf-r:8080",
                    "metrics_url": "http://gguf-r:8081/metrics",
                    "container": "gguf-r"},
     "gpu_index": 4,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
]

_NIMBUS_POOL = [{"index": 4, "total": 40, "used": 10, "free": 30}]


class _FakeRemoteClients:
    """RemoteEngineClients-shaped fake: `client_for(node_id, resource)`
    only, answering None for a pair it has no client for (that class's own
    "not operable is a state, not an error" posture)."""

    def __init__(self, clients: dict) -> None:
        self._clients = clients

    def client_for(self, node_id: str, resource: str):
        return self._clients.get((node_id, resource))


def _remote_kwargs(client=None, pools=None, engines=None):
    return dict(
        engines=_REMOTE_ENGINES if engines is None else engines,
        clients=_FakeRemoteClients(
            {} if client is None else {("nimbus", "gguf-r"): client}),
        gpu_pools={"nimbus": _NIMBUS_POOL} if pools is None else pools,
        registry=StubRegistry(footprints={"r.gguf": 7}),
    )


def test_snapshot_remote_keys_tenants_by_node_and_resource():
    world = World(clock=FakeClock())

    tenants = world.snapshot_remote(
        **_remote_kwargs(client=StubLemonade(loaded="r.gguf")))

    assert list(tenants) == ["nimbus/gguf-r"]


def test_snapshot_remote_tenant_carries_its_node_id_and_resource():
    """The world entry has to say WHICH node it came from: everything
    downstream builds its observation/intent key from these two fields
    rather than from the map key."""
    world = World(clock=FakeClock())

    tenant = world.snapshot_remote(
        **_remote_kwargs(client=StubLemonade(loaded="r.gguf")))["nimbus/gguf-r"]

    assert tenant["node_id"] == "nimbus"
    assert tenant["resource"] == "gguf-r"
    assert tenant["engine"] == "lemonade"
    assert tenant["gpu_index"] == 4
    assert tenant["state"] == "loaded"


def test_snapshot_remote_unreadable_gpu_pool_makes_that_nodes_engines_unknown():
    """The node-agent's GPU read IS the node's liveness probe (the
    node-observer precedent: that probe alone governs). Failing it means we
    could not look at the node at all — its engines are 'unknown', which is
    NOT 'nothing is loaded', and the engines are not probed one by one
    behind an already-dead agent."""
    client = StubLemonade(loaded="r.gguf")
    world = World(clock=FakeClock())

    tenants = world.snapshot_remote(
        **_remote_kwargs(client=client, pools={"nimbus": None}))

    assert tenants["nimbus/gguf-r"]["state"] == "unknown"


def test_snapshot_remote_unknown_record_keeps_its_own_kinds_shape():
    """The unknown record is produced by the KIND's own observe(), never
    synthesized here — app.state does not know that a lemonade-kind record
    carries `model`/`footprint` while a comfyui-kind one carries `queue`,
    and a synthesized guess is how a downstream KeyError gets in."""
    world = World(clock=FakeClock())

    tenant = world.snapshot_remote(
        **_remote_kwargs(pools={"nimbus": None}))["nimbus/gguf-r"]

    assert tenant["state"] == "unknown"
    assert tenant["model"] is None
    assert tenant["footprint"] is None


def test_snapshot_remote_engine_with_no_operable_client_is_unknown():
    """A declared engine whose client cannot be built (credential gone,
    kind with no remote constructor) is still DECLARED — it must appear, as
    unknown, not vanish from the world."""
    world = World(clock=FakeClock())

    tenants = world.snapshot_remote(**_remote_kwargs(client=None))

    assert tenants["nimbus/gguf-r"]["state"] == "unknown"


def test_snapshot_remote_missing_pool_entry_is_unknown_not_a_crash():
    world = World(clock=FakeClock())

    tenants = world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded="r.gguf"), pools={}))

    assert tenants["nimbus/gguf-r"]["state"] == "unknown"


def test_snapshot_remote_idle_memory_is_independent_of_a_same_named_local_one():
    """A local resource and a remote one may share a name; their idle
    clocks must not be the same entry."""
    clock = FakeClock()
    world = World(clock=clock)
    local_client = StubLemonade(loaded=None, activity=1)
    local_engines = [{"resource": "gguf-r", "kind": "lemonade",
                      "connection": {"url": "u", "metrics_url": "m",
                                     "container": "c"},
                      "gpu_index": 1,
                      "policy_defaults": {"priority": 50, "pinned": False,
                                          "idle_ttl": 900}}]

    world.snapshot(gpus=[_gpu()], engines=local_engines,
                   clients=_FakeLocalClients({"gguf-r": local_client}),
                   litellm=StubLiteLLM(), registry=StubRegistry())
    clock.advance(60)
    world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))
    clock.advance(5)

    local = world.snapshot(gpus=[_gpu()], engines=local_engines,
                           clients=_FakeLocalClients({"gguf-r": local_client}),
                           litellm=StubLiteLLM(), registry=StubRegistry())
    remote = world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))

    assert local["tenants"]["gguf-r"]["idle_s"] == 65
    assert remote["nimbus/gguf-r"]["idle_s"] == 5


def test_local_snapshot_does_not_prune_remote_idle_memory():
    """ONE World instance is shared by the arbiter tick and the HTTP paths.
    A caller that snapshots only the local half must not wipe the remote
    half's idle clocks — that would reset every remote resource's clock on
    every tick that skipped the remote pass."""
    clock = FakeClock()
    world = World(clock=clock)

    world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))
    clock.advance(30)
    world.snapshot(**_healthy_kwargs())          # local half only
    clock.advance(0)

    remote = world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))

    assert remote["nimbus/gguf-r"]["idle_s"] == 30


def test_snapshot_remote_drops_memory_for_an_undeclared_engine():
    """Same rule the local half follows: a resource that left the
    declaration loses its clock — re-declaring it starts fresh."""
    clock = FakeClock()
    world = World(clock=clock)

    world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))
    clock.advance(30)
    world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1), engines=[]))
    clock.advance(5)
    remote = world.snapshot_remote(**_remote_kwargs(
        client=StubLemonade(loaded=None, activity=1)))

    assert remote["nimbus/gguf-r"]["idle_s"] == 0


# ===========================================================================
# sglang-omni Task 7, fix round 1 (review finding 1) — an UNSEEN outage must
# not accrue idle time.
#
# `snapshot_remote` returns the kind's `unknown()` WITHOUT running the
# adapter's `observe()` when the node's GPU pool is unreadable or no client
# is operable — so the adapter's own "a non-idle answer re-arms the clock"
# rule, which lives inside observe(), never runs on exactly the path a
# powered-off box takes. Left alone, the first observation after the box
# comes back reads the whole dark window as observed idle time and the deck
# unloads an engine it just spent ~4 minutes (GF4) booting.
#
# Fixture discipline: node "nimbus" (never "sparky"), resource "song-r"
# (never "omni"), idle_ttl 120 (never the declared default 900), GPU 4.
# ===========================================================================

_REMOTE_OMNI_ENGINES = [
    {"resource": "song-r", "kind": "sglang-omni", "node_id": "nimbus",
     "connection": {"url": "http://127.0.0.1:8008"}, "gpu_index": 4,
     "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120}},
]

_OMNI_POLICY = {"priority": 5, "pinned": False, "idle_ttl": 120}


class StubOmni:
    """SglangOmniClient-shaped: status() answering the node-agent's wire
    dict."""

    def __init__(self, healthy: bool = True, busy_requests: int | None = 0) -> None:
        self._healthy = healthy
        self._busy = busy_requests

    def status(self) -> dict:
        return {"reachable": True, "healthy": self._healthy,
                "busy_requests": self._busy}


def _omni_snapshot(world, *, dark=False):
    """One remote snapshot of the single declared sglang-omni engine —
    `dark` = the node's agent did not answer (pool None), the path that
    skips the adapter's observe() entirely."""
    return world.snapshot_remote(
        engines=_REMOTE_OMNI_ENGINES,
        clients=_FakeRemoteClients({("nimbus", "song-r"): StubOmni()}),
        gpu_pools={"nimbus": None if dark else _NIMBUS_POOL},
        registry=StubRegistry(),
    )["nimbus/song-r"]


def _omni_idle_action(tenant):
    from app.engine_kinds import ENGINE_KINDS

    return ENGINE_KINDS["sglang-omni"].idle_action(tenant, _OMNI_POLICY, None, 0)


def test_remote_idle_clock_accrues_while_the_node_is_actually_watched():
    """Positive control for the regression below: a genuinely idle engine
    DOES pass its TTL and become an idle-release candidate. Without this,
    the "must not fire" assertion below could pass for the wrong reason."""
    clock = FakeClock(start=1000.0)
    world = World(clock=clock)
    _omni_snapshot(world)

    clock.advance(121.0)
    tenant = _omni_snapshot(world)

    assert tenant["idle_s"] == 121.0
    assert _omni_idle_action(tenant) == {"type": "unload", "model": None}


def test_remote_idle_clock_restarts_after_an_unseen_outage():
    """THE regression (fix round 1, finding 1): time the deck could not
    observe is not time it observed the engine idle.

    Reviewer's reproduction: idle past its TTL, then the node goes dark for
    several ticks (RemoteObserver's backoff stretches that to minutes), then
    it answers healthy and idle again. Before the fix the first healthy tick
    reported idle_s 601.0 and idle_action emitted an unload — against an
    engine that may have just finished a ~4-minute boot."""
    clock = FakeClock(start=1000.0)
    world = World(clock=clock)
    _omni_snapshot(world)
    clock.advance(121.0)
    _omni_idle_action(_omni_snapshot(world))          # (fires — see above)

    for _ in range(5):                                 # the box goes dark
        clock.advance(96.0)
        assert _omni_snapshot(world, dark=True)["state"] == "unknown"
    clock.advance(2.0)
    tenant = _omni_snapshot(world)                     # ...and comes back

    assert tenant["state"] == "idle"
    assert tenant["idle_s"] == 0.0
    assert _omni_idle_action(tenant) is None


def test_remote_unknown_does_not_wipe_a_different_engines_idle_clock():
    """The reset is per TENANT, not a blanket wipe: an engine on a node
    that is still answering keeps accruing while another one goes dark."""
    engines = _REMOTE_OMNI_ENGINES + [
        {**_REMOTE_OMNI_ENGINES[0], "resource": "song-b", "node_id": "cirrus"}]
    clients = _FakeRemoteClients({("nimbus", "song-r"): StubOmni(),
                                  ("cirrus", "song-b"): StubOmni()})
    clock = FakeClock(start=1000.0)
    world = World(clock=clock)

    def snap(nimbus_pool):
        return world.snapshot_remote(
            engines=engines, clients=clients,
            gpu_pools={"nimbus": nimbus_pool, "cirrus": _NIMBUS_POOL},
            registry=StubRegistry())

    snap(_NIMBUS_POOL)
    clock.advance(60.0)
    snap(None)                                          # only nimbus is dark
    clock.advance(30.0)
    tenants = snap(_NIMBUS_POOL)

    assert tenants["nimbus/song-r"]["idle_s"] == 0.0     # restarted
    assert tenants["cirrus/song-b"]["idle_s"] == 90.0    # untouched
