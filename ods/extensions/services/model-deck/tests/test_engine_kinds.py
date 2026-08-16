"""Declaration validation (E1 spec §1). Kinds and their connection
schemas live in app/engine_kinds.py — the single module allowed to
know engine names (spec §8)."""

import pytest

from app.engine_kinds import KNOWN_KINDS, validate_engines
from app.engines import EngineError


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


# --- E1 Task 5: KNOWN_KINDS gains remote_capable; validate_engines gains
# a `remote` argument enforcing it ------------------------------------------


def test_known_kinds_declare_connection_and_remote_capable_shape():
    """Schema is now `kind -> {"connection": {...}, "remote_capable": bool}`
    — every known kind carries both keys, and (today) none is
    remote_capable (all three run in-process with the deck; Task 7 adds
    the first remote-capable kind)."""
    for kind, spec in KNOWN_KINDS.items():
        assert set(spec) == {"connection", "remote_capable"}, kind
        assert spec["remote_capable"] is False, kind


def test_validate_engines_default_is_not_remote():
    """`remote` defaults False — every existing caller (the local entry's
    own validation, and every test above that calls validate_engines with
    just one argument) is unaffected by this task."""
    validate_engines([_entry()])  # no error, same as remote=False


def test_validate_engines_remote_refuses_non_remote_capable_kind_naming_it():
    with pytest.raises(ValueError, match="lemonade") as exc:
        validate_engines([_entry()], remote=True)
    assert "not remote_capable" in str(exc.value)


def test_validate_engines_remote_accepts_an_empty_list():
    # Vacuously true: no kind is named, so nothing can fail the
    # remote_capable check.
    validate_engines([], remote=True)


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


# ===========================================================================
# sglang-omni Task 6 — `unknown()`: the kind's own "we failed to look"
# record, callable WITHOUT a client.
#
# The two call sites must agree by construction, so each case below proves
# `unknown()` against what that kind's own `observe()` returns when its
# client raises — never against a hand-written expected dict, which would
# just be a third copy of the same shape drifting on its own.
# ===========================================================================


class _RaisingClient:
    """Every probe surface any adapter's observe() touches, all raising —
    what a real client does with its engine unreachable. `load_in_flight`
    and `activity` answer instead of raising: their own clients' contracts
    say they never raise (app/engines/lemonade.py)."""

    def load_in_flight(self):
        return False

    def activity(self):
        return None

    def status(self):
        raise EngineError("unreachable")

    def queue_len(self):
        raise EngineError("unreachable")

    def stats(self):
        raise EngineError("unreachable")


@pytest.mark.parametrize("kind", ["lemonade", "comfyui", "hipfire"])
def test_adapter_unknown_matches_its_own_observe_on_engine_error(kind):
    from app.engine_kinds import ENGINE_KINDS

    adapter = ENGINE_KINDS[kind]
    observed = adapter.observe(_RaisingClient(), {}, 0.0,
                               {"registry": _FakeRegistry(), "routes": {},
                                "resource": "r-1"})

    assert adapter.unknown() == observed
    assert observed["state"] == "unknown"


@pytest.mark.parametrize("kind", ["lemonade", "comfyui", "hipfire"])
def test_adapter_unknown_is_not_a_bare_state_dict(kind):
    """The point of asking the KIND: each one's record carries different
    fields (model/footprint/idle_s vs queue/idle_s vs queue_depth), and a
    caller synthesizing `{"state": "unknown"}` would hand downstream a
    shape that KeyErrors."""
    from app.engine_kinds import ENGINE_KINDS

    assert set(ENGINE_KINDS[kind].unknown()) != {"state"}
