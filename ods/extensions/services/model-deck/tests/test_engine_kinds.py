"""Declaration validation (E1 spec §1). Kinds and their connection
schemas live in app/engine_kinds.py — the single module allowed to
know engine names (spec §8)."""

import pytest

from app.engine_kinds import KNOWN_KINDS, validate_engines


def _entry(**over):
    # Fixture rule: resource deliberately NOT an engine name, GPU not 0/1.
    e = {"resource": "gguf-a", "kind": "lemonade",
         "connection": {"url": "http://gguf-a:8080",
                        "metrics_url": "http://gguf-a:8001/metrics",
                        "container": "ods-gguf-a"},
         "gpu_index": 3,
         "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}}
    e.update(over)
    return e


def test_known_kinds_are_exactly_the_three_adapters():
    assert set(KNOWN_KINDS) == {"lemonade", "comfyui", "hipfire"}


def test_valid_list_passes():
    validate_engines([_entry()])


def test_unknown_kind_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        validate_engines([_entry(kind="vllm")])


def test_missing_required_connection_field_refused():
    with pytest.raises(ValueError, match="url"):
        validate_engines([_entry(connection={"container": "c"})])


def test_unknown_connection_field_refused():
    with pytest.raises(ValueError, match="extra"):
        validate_engines([_entry(connection={"url": "http://x",
                                             "metrics_url": "http://x/m",
                                             "container": "c",
                                             "port": 9})])


def test_duplicate_resource_refused():
    with pytest.raises(ValueError, match="duplicate"):
        validate_engines([_entry(), _entry()])


def test_gpu_index_must_be_nonnegative_int():
    with pytest.raises(ValueError, match="gpu_index"):
        validate_engines([_entry(gpu_index=True)])   # bool is not an int here
    with pytest.raises(ValueError, match="gpu_index"):
        validate_engines([_entry(gpu_index=-1)])


def test_policy_defaults_exact_fields():
    with pytest.raises(ValueError, match="policy_defaults"):
        validate_engines([_entry(policy_defaults={"priority": 1})])


def test_resource_shape_refused_when_slashy():
    # resource keys build "local/<resource>" lifecycle keys — a slash
    # would forge a foreign key ([[literal-declared-inputs]]).
    with pytest.raises(ValueError, match="resource"):
        validate_engines([_entry(resource="local/evil")])


# ===========================================================================
# E1 Task 3: adapters (observe/active/arbiter_verbs/human_verbs/demand) +
# World generalization. Fixture rule: resources gguf-a/gguf-b/img/agent,
# GPUs 2 and 3 — none of this may match live topology (lemonade/comfyui/
# hipfire, GPUs 0/1); tests/test_state.py is where the per-KIND logic
# (which legitimately needs realistic kind test doubles) is pinned instead.
# ===========================================================================


class _FakeGguf:
    """Duck-types the lemonade client surface observe() touches."""
    def __init__(self):
        self._loaded = "m.gguf"; self._activity = 7
        self.load_in_flight_v = False
    def load_in_flight(self): return self.load_in_flight_v
    def status(self): return {"loaded": self._loaded}
    def activity(self): return self._activity


class _FakeLocalClients:
    """Dict wrapper mirroring app.local_clients.LocalClients' client_for."""
    def __init__(self, clients: dict) -> None:
        self._clients = clients

    def client_for(self, resource: str):
        return self._clients.get(resource)


class _FakeLitellm:
    def __init__(self, routes: dict | None = None) -> None:
        self._routes = routes or {}

    def route_table(self) -> dict:
        return dict(self._routes)


class _FakeRegistry:
    def __init__(self, footprints: dict | None = None) -> None:
        self._footprints = footprints or {}

    def footprint(self, key: str) -> int:
        if key not in self._footprints:
            raise FileNotFoundError(key)
        return self._footprints[key]


def _gpu(index, total=100, used=0):
    return {"index": index, "vram_total": total, "vram_used": used, "pids": {}}


_GPUS_2_3 = [_gpu(2), _gpu(3)]


def test_two_lemonade_kind_resources_have_independent_idle_clocks():
    """The generalization's point: resource keys the memory, not the kind."""
    from app.state import World
    t = {"v": 100.0}
    world = World(clock=lambda: t["v"])
    engines = [
        {"resource": "gguf-a", "kind": "lemonade",
         "connection": {"url": "u", "metrics_url": "m", "container": "c"},
         "gpu_index": 2,
         "policy_defaults": {"priority": 1, "pinned": False, "idle_ttl": 60}},
        {"resource": "gguf-b", "kind": "lemonade",
         "connection": {"url": "u2", "metrics_url": "m2", "container": "c2"},
         "gpu_index": 3,
         "policy_defaults": {"priority": 2, "pinned": False, "idle_ttl": 60}},
    ]
    a, b = _FakeGguf(), _FakeGguf()
    clients = _FakeLocalClients({"gguf-a": a, "gguf-b": b})
    world.snapshot(_GPUS_2_3, engines, clients, _FakeLitellm(), _FakeRegistry())
    t["v"] = 150.0
    a._activity = 8                        # a is active; b stays idle
    snap = world.snapshot(_GPUS_2_3, engines, clients, _FakeLitellm(), _FakeRegistry())
    assert snap["tenants"]["gguf-a"]["idle_s"] == 0.0
    assert snap["tenants"]["gguf-b"]["idle_s"] == 50.0
    assert snap["tenants"]["gguf-a"]["engine"] == "lemonade"
    assert snap["placement"] == {"gguf-a": 2, "gguf-b": 3}


def test_absent_engine_is_absent_everywhere():
    """Spec §1: absence representable — no entry, no tenant, no unknown."""
    from app.state import World
    world = World(clock=lambda: 0.0)
    snap = world.snapshot(_GPUS_2_3, [], _FakeLocalClients({}),
                          _FakeLitellm(), _FakeRegistry())
    assert snap["tenants"] == {}


class _FakeComfy:
    """Duck-types the comfy client surface observe() touches."""
    def __init__(self, queue=0):
        self._queue = queue

    def queue_len(self):
        return self._queue


def test_resource_redeclared_under_a_different_kind_gets_a_fresh_idle_clock():
    """Review fix (T3 round 2): a resource KEPT but re-declared under a
    DIFFERENT kind must not inherit the old kind's idle-clock bookkeeping.
    Different adapters reuse the same mem key NAME ("last_activity_time")
    for unrelated clocks — lemonade-kind's idle baseline is meaningless to
    a comfyui-kind observation, and reading it as one would silently
    report a wrong (often huge) idle_s on the very first tick under the
    new kind instead of the honest 0.0 a fresh resource always starts at
    (see test_comfy_state_idle_when_queue_empty's shape in
    tests/test_state.py for that baseline elsewhere).

    Direction matters here, and this is deliberately lemonade-THEN-comfyui,
    not the reverse: lemonade's own observe() has a second, independent
    "loaded value changed" transition check (`loaded != mem.get(
    "last_loaded")`) that happens to reset the clock anyway on ANY mem
    that lacks a "last_loaded" key — which comfyui's observe() never
    writes — so a comfyui-then-lemonade transition passes with or without
    this fix and would be a [[defaults-that-hide-bugs]] test. Verified
    empirically both ways before choosing this direction."""
    from app.state import World

    t = {"v": 1000.0}
    world = World(clock=lambda: t["v"])

    lemonade_engine = [{"resource": "img", "kind": "lemonade",
                        "connection": {"url": "u", "metrics_url": "m", "container": "c"},
                        "gpu_index": 2,
                        "policy_defaults": {"priority": 1, "pinned": False, "idle_ttl": 60}}]
    world.snapshot(_GPUS_2_3, lemonade_engine, _FakeLocalClients({"img": _FakeGguf()}),
                  _FakeLitellm(), _FakeRegistry())

    t["v"] = 1300.0  # 300s later — lemonade's own idle clock would read 300s stale here
    comfy_engine = [{"resource": "img", "kind": "comfyui",
                     "connection": {"url": "u"}, "gpu_index": 2,
                     "policy_defaults": {"priority": 1, "pinned": False, "idle_ttl": 60}}]
    snap = world.snapshot(_GPUS_2_3, comfy_engine,
                          _FakeLocalClients({"img": _FakeComfy(queue=0)}),
                          _FakeLitellm(), _FakeRegistry())

    assert snap["tenants"]["img"]["engine"] == "comfyui"
    assert snap["tenants"]["img"]["idle_s"] == 0.0  # fresh baseline, NOT 300.0


# --- adapter surface: active / arbiter_verbs / human_verbs / demand -------


def test_lemonade_adapter_active_verbs_and_demand():
    from app.engine_kinds import ENGINE_KINDS
    lemonade = ENGINE_KINDS["lemonade"]
    assert lemonade.active({"state": "loaded"}) is True
    assert lemonade.active({"state": "unloaded"}) is False
    assert lemonade.arbiter_verbs() == frozenset({"unload"})
    assert lemonade.human_verbs() == frozenset({"load", "unload"})
    assert lemonade.demand() is True


def test_comfy_adapter_active_verbs_and_demand():
    from app.engine_kinds import ENGINE_KINDS
    comfy = ENGINE_KINDS["comfyui"]
    assert comfy.active({"state": "busy"}) is True
    assert comfy.active({"state": "idle"}) is False
    assert comfy.arbiter_verbs() == frozenset({"free"})
    assert comfy.human_verbs() == frozenset({"free"})
    assert comfy.demand() is False


def test_hipfire_adapter_active_verbs_and_demand():
    from app.engine_kinds import ENGINE_KINDS
    hipfire = ENGINE_KINDS["hipfire"]
    assert hipfire.active({"state": "running"}) is True
    assert hipfire.active({"state": "parked"}) is False
    # No arbiter verb at all — park stays human-only (structural omission
    # made explicit, spec §2).
    assert hipfire.arbiter_verbs() == frozenset()
    assert hipfire.human_verbs() == frozenset({"park", "resume"})
    assert hipfire.demand() is False


def test_hipfire_observe_looks_up_its_own_resource_name_in_routes():
    """The delegated mechanism choice (Task 3 brief): hipfire's model comes
    from routes.get(RESOURCE), not a hardcoded "hipfire" literal — proven
    here with a resource name that ISN'T "hipfire"."""
    from app.engine_kinds import ENGINE_KINDS

    class _FakeHipfireClient:
        def status(self):
            return "running"

        def stats(self):
            return {"queue_depth": 0}

    hipfire = ENGINE_KINDS["hipfire"]
    ctx = {"registry": _FakeRegistry(), "routes": {"agent": "openai/big-model"},
           "resource": "agent"}
    obs = hipfire.observe(_FakeHipfireClient(), {}, 0.0, ctx)
    assert obs["model"] == "big-model"
