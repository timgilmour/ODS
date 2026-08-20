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
         "container_consent": True,
         "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}}
    e.update(over)
    return e


def test_known_kinds_are_exactly_the_four_adapters():
    """RE-EXPRESSED (sglang-omni Task 7, was
    ...are_exactly_the_three_adapters): the kind set is closed, and it is
    now four — E1's built-in triple plus the first kind that runs OFF the
    deck's own box. Still an exact-set assertion, not a superset one: a
    kind appearing in KNOWN_KINDS with no ENGINE_KINDS adapter (or the
    reverse) is exactly the drift this pins."""
    assert set(KNOWN_KINDS) == {"lemonade", "comfyui", "hipfire", "sglang-omni"}


def test_every_known_kind_has_an_adapter():
    from app.engine_kinds import ENGINE_KINDS

    assert set(ENGINE_KINDS) == set(KNOWN_KINDS)


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


def test_validate_engines_requires_container_consent():
    e = _entry()
    del e["container_consent"]
    with pytest.raises(ValueError, match="container_consent"):
        validate_engines([e])


def test_validate_engines_container_consent_must_be_bool():
    with pytest.raises(ValueError, match="container_consent"):
        validate_engines([_entry(container_consent="yes")])
    # Review fix round 1 (Finding 2): non-bool TRUTHY/FALSY values must be
    # refused too, not just the string case — a future truthiness refactor
    # (e.g. `if not consent:` instead of `isinstance(consent, bool)`) would
    # let 1/0 slip through as if they were True/False.
    with pytest.raises(ValueError, match="container_consent"):
        validate_engines([_entry(container_consent=1)])
    with pytest.raises(ValueError, match="container_consent"):
        validate_engines([_entry(container_consent=0)])


# --- Finding 1 (whole-branch review): comfyui gains an OPTIONAL
# connection.container, so declared_containers' OCI provenance sweep can
# enumerate ods-comfyui like every other kind — comfyui's schema used to be
# the one url-only kind with no way to carry a container name at all. The
# optionality mechanism is minimal: KNOWN_KINDS' schema dict already maps
# field -> required (bool); `False` was simply unused until now. A field
# marked False is never REQUIRED, but validate_engines still type-checks it
# (non-empty string) when the caller chose to include it — an optional field
# is not an unchecked one.

def _comfy_entry(**over):
    e = {"resource": "img", "kind": "comfyui",
         "connection": {"url": "http://comfyui:8188"},
         "gpu_index": 1,
         "container_consent": True,
         "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}}
    e.update(over)
    return e


def test_comfyui_connection_schema_declares_container_optional():
    assert KNOWN_KINDS["comfyui"]["connection"] == {"url": True, "container": False}


def test_comfyui_container_absent_still_passes():
    validate_engines([_comfy_entry()])


def test_comfyui_accepts_a_container_name():
    validate_engines([_comfy_entry(
        connection={"url": "http://comfyui:8188", "container": "ods-comfyui"})])


def test_comfyui_refuses_a_non_string_container():
    with pytest.raises(ValueError, match="container"):
        validate_engines([_comfy_entry(
            connection={"url": "http://comfyui:8188", "container": 5})])


def test_comfyui_refuses_an_empty_string_container():
    with pytest.raises(ValueError, match="container"):
        validate_engines([_comfy_entry(
            connection={"url": "http://comfyui:8188", "container": ""})])


def test_resource_shape_refused_when_slashy():
    # resource keys build "local/<resource>" lifecycle keys — a slash
    # would forge a foreign key ([[literal-declared-inputs]]).
    with pytest.raises(ValueError, match="resource"):
        validate_engines([_entry(resource="local/evil")])


def test_resource_shape_refuses_the_reserved_slot0_name():
    """'slot0' is the serving slot's own observation key
    (app.observe.slot_key: '<node>/slot0'), not a name any DECLARED engine
    may also claim — an engine named 'slot0' on a swap-capable node would
    collide with the slot at that same key across merge_observations
    (last-wins, so the engine silently vanishes), the intent store (one
    record per key, so a load overwrites the slot's swap intent), and the
    board (duplicate React keys). Checked in the resource-shape block,
    before the kind/remote_capable checks below it, so the refusal fires
    regardless of `remote` — this fixture's kind (lemonade) is not even
    remote_capable, and the point is precisely that the slot0 check never
    has to reach that gate to refuse it."""
    with pytest.raises(ValueError, match="slot0"):
        validate_engines([_entry(resource="slot0")])


def test_resource_shape_refuses_slot0_on_a_remote_declaration_too():
    """Same refusal on `remote=True`, and with a kind that IS remote_capable
    (sglang-omni, via `_omni_entry` below) — deliberately NOT the lemonade
    fixture above ([[defaults-that-hide-bugs]]): `_entry(resource="slot0")`
    under `remote=True` would raise "not remote_capable" first (lemonade
    isn't), and that message happens to also contain "slot0" (the resource
    name rides in every error string here), so it would pass this test for
    the wrong reason. A remote_capable kind removes that confound: the only
    thing left that can raise is the slot0 reservation itself."""
    with pytest.raises(ValueError, match="slot0"):
        validate_engines([_omni_entry(resource="slot0")], remote=True)


# --- E1 Task 5: KNOWN_KINDS gains remote_capable; validate_engines gains
# a `remote` argument enforcing it ------------------------------------------


def test_known_kinds_declare_remote_capability_per_kind():
    """RE-EXPRESSED (sglang-omni Task 7, was
    ...declare_connection_and_remote_capable_shape): the old blanket
    `spec["remote_capable"] is False` was true only while no kind could run
    off-box. It is now per-kind — the E1 triple are all sibling containers
    on THIS box and stay False; sglang-omni is the first True — so the
    assertion names which is which rather than asserting one answer for
    everything (a superset check like "at least one is True" would pass
    even if a local kind silently flipped).

    The SHAPE half of the old assertion moved to
    test_known_kinds_declare_where_each_kind_may_RUN (fix round 1), which
    owns the key set now that there are two run-location flags."""
    remote_capable = {"lemonade": False, "comfyui": False, "hipfire": False,
                      "sglang-omni": True}
    for kind, spec in KNOWN_KINDS.items():
        assert spec["remote_capable"] is remote_capable[kind], kind


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
    # placement (INST I1 Task 2, D-I1-2): now the declared GPU LIST, even
    # for a legacy scalar declaration — the sole reader is tests + the
    # api.ts comment (Task 8 updates the latter).
    assert snap["placement"] == {"gguf-a": [2], "gguf-b": [3]}


def test_snapshot_stamps_gpu_indices_and_a_derived_min_gpu_index():
    """INST I1 Task 2 (D-I1-2): a declaration spelled with the LIST stamps
    the tenant with BOTH the list and a DERIVED min scalar — one producer
    (app.state), so the two spellings cannot disagree. A legacy scalar
    declaration reads back as a one-element list with the same scalar.

    Written here, not tests/test_state.py: this is the file's own
    two-lemonade-kind-resource idiom (gguf-a/gguf-b, GPUs away from live
    topology) — test_state.py's fixtures pin per-KIND observe() logic with
    resource==kind names instead (see that file's module docstring)."""
    from app.state import World
    world = World(clock=lambda: 0.0)
    engines = [
        {"resource": "gguf-a", "kind": "lemonade",
         "connection": {"url": "u", "metrics_url": "m", "container": "c"},
         "gpu_index": 2,
         "policy_defaults": {"priority": 1, "pinned": False, "idle_ttl": 60}},
        {"resource": "gguf-b", "kind": "lemonade",
         "connection": {"url": "u2", "metrics_url": "m2", "container": "c2"},
         "gpu_indices": [3, 4],
         "policy_defaults": {"priority": 2, "pinned": False, "idle_ttl": 60}},
    ]
    clients = _FakeLocalClients({"gguf-a": _FakeGguf(), "gguf-b": _FakeGguf()})
    gpus = [_gpu(2), _gpu(3), _gpu(4)]

    snap = world.snapshot(gpus, engines, clients, _FakeLitellm(), _FakeRegistry())

    assert snap["tenants"]["gguf-b"]["gpu_indices"] == [3, 4]
    assert snap["tenants"]["gguf-b"]["gpu_index"] == 3
    assert snap["placement"]["gguf-b"] == [3, 4]
    # legacy scalar declaration -> one-element list, same scalar
    assert snap["tenants"]["gguf-a"]["gpu_indices"] == [2]
    assert snap["tenants"]["gguf-a"]["gpu_index"] == 2


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


def test_comfy_build_client_tolerates_the_new_optional_container_key():
    """Finding 1's NOTE (whole-branch review): `_ComfyAdapter.build_client`
    builds from `connection["url"]` alone and must not choke on the new
    optional `container` key riding alongside it — verified directly rather
    than only inferred from reading the adapter."""
    from app.engine_kinds import ENGINE_KINDS
    from app.engines.comfyui import ComfyClient

    comfy = ENGINE_KINDS["comfyui"]
    client = comfy.build_client(
        {"url": "http://comfyui:8188", "container": "ods-comfyui"},
        object(), None)
    assert isinstance(client, ComfyClient)


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


def test_hipfire_adapter_build_client_derives_guard_hosts_from_container_and_gateway_host():
    """E1 debt 2 (D-I1-5, INST I1 T3): the client's park guard must key on
    the resource's OWN declared names (container + optional gateway_host),
    never a substring of the kind name — proven at the adapter boundary
    where connection dicts actually flow in from nodes.json."""
    from app.engine_kinds import ENGINE_KINDS
    from app.engines.hipfire import HipfireClient
    from app.settings import Settings

    hipfire = ENGINE_KINDS["hipfire"]
    settings = Settings()

    client = hipfire.build_client(
        {"container": "deck-agent", "gateway_host": "agent"}, settings, None)
    assert isinstance(client, HipfireClient)
    assert client._guard_hosts == frozenset({"deck-agent", "agent"})

    no_gateway = hipfire.build_client({"container": "deck-agent"}, settings, None)
    assert no_gateway._guard_hosts == frozenset({"deck-agent"})


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


@pytest.mark.parametrize("kind", ["lemonade", "comfyui", "hipfire", "sglang-omni"])
def test_adapter_unknown_matches_its_own_observe_on_engine_error(kind):
    from app.engine_kinds import ENGINE_KINDS

    adapter = ENGINE_KINDS[kind]
    observed = adapter.observe(_RaisingClient(), {}, 0.0,
                               {"registry": _FakeRegistry(), "routes": {},
                                "resource": "r-1"})

    assert adapter.unknown() == observed
    assert observed["state"] == "unknown"


@pytest.mark.parametrize("kind", ["lemonade", "comfyui", "hipfire", "sglang-omni"])
def test_adapter_unknown_is_not_a_bare_state_dict(kind):
    """The point of asking the KIND: each one's record carries different
    fields (model/footprint/idle_s vs queue/idle_s vs queue_depth), and a
    caller synthesizing `{"state": "unknown"}` would hand downstream a
    shape that KeyErrors."""
    from app.engine_kinds import ENGINE_KINDS

    assert set(ENGINE_KINDS[kind].unknown()) != {"state"}


# ===========================================================================
# sglang-omni Task 7 — the fourth kind: `_SglangOmniAdapter`.
#
# The first kind that runs OFF this box: every probe and every verb rides a
# node-agent, so the "client" here duck-types app/engines/sglang_omni.py's
# SglangOmniClient (status/up/down), never a local HTTP engine.
#
# Fixture rule ([[defaults-that-hide-bugs]]): node "nimbus" (never the
# live-seeded "sparky"), resource "song-r" (never "omni"), GPU 4, and every
# policy fixture's `idle_ttl` is 120 — deliberately NOT the declared default
# 900, so a test can never pass by coinciding with a default.
# ===========================================================================

# GF5 (gate run 2026-08-16): ~62 GiB resident at idle on sparky. Written out
# here rather than imported from the adapter so the FACT is pinned by the
# test, not just the code agreeing with itself.
_OMNI_IDLE_FOOTPRINT = 62 * 1024**3

_OMNI_KEY = "nimbus/song-r"


class _FakeOmni:
    """Duck-types the SglangOmniClient surface the adapter touches.

    `status()` answers the node-agent's wire dict VERBATIM
    (`{"reachable", "healthy", "busy_requests"}`, app/engines/sglang_omni.py's
    documented contract) — never a deck-shaped dict, so a test cannot pass
    against a shape the wire never sends.
    """

    def __init__(self, *, reachable=True, healthy=True, busy_requests=0,
                 raises=None, verb_raises=None):
        self._status = {"reachable": reachable, "healthy": healthy,
                        "busy_requests": busy_requests}
        self._raises = raises
        self._verb_raises = verb_raises
        self.calls = []          # mutating verbs only

    def status(self):
        if self._raises is not None:
            raise self._raises
        return dict(self._status)

    def up(self):
        self.calls.append("up")
        if self._verb_raises is not None:
            raise self._verb_raises

    def down(self):
        self.calls.append("down")
        if self._verb_raises is not None:
            raise self._verb_raises

    def close(self):
        pass


def _omni():
    from app.engine_kinds import ENGINE_KINDS

    return ENGINE_KINDS["sglang-omni"]


def _observe(client, mem=None, now=0.0):
    """One adapter observation. `routes` is None on purpose: the litellm
    route table is the LOCAL box's, and World.snapshot_remote hands remote
    kinds None for it (that method's own docstring)."""
    return _omni().observe(client, {} if mem is None else mem, now,
                           {"registry": _FakeRegistry(), "routes": None,
                            "resource": "song-r"})


def _policy(*, pinned=False, idle_ttl=120, priority=5):
    return {"priority": priority, "pinned": pinned, "idle_ttl": idle_ttl}


def _obs(state, busy_requests, idle_s=0.0):
    return {"state": state, "busy_requests": busy_requests, "model": None,
            "idle_s": idle_s}


# --- declaration: the kind is known, remote-capable, one required field ----


def test_sglang_omni_connection_schema_requires_only_a_url():
    assert KNOWN_KINDS["sglang-omni"]["connection"] == {"url": True}


def _omni_entry(**over):
    e = {"resource": "song-r", "kind": "sglang-omni",
         "connection": {"url": "http://127.0.0.1:8008"},
         "gpu_index": 4,
         "container_consent": True,
         "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120}}
    e.update(over)
    return e


def test_validate_engines_accepts_sglang_omni_on_a_remote_declaration():
    """The point of the whole task: `remote=True` no longer refuses every
    kind. A declaration naming sglang-omni passes the gate the other three
    kinds still fail."""
    validate_engines([_omni_entry()], remote=True)


def test_sglang_omni_refuses_an_unknown_connection_field():
    """Refuse-never-coerce reaches the new kind too: only `url` is declared,
    so the node-agent-side facts (health_url, busy port, compose file) stay
    where A1 put them — the HOST-owned engines.json allowlist — and cannot
    be smuggled in through the deck's declaration.

    `remote=True` (fix round 1): the only place this kind may legally be
    declared at all, so the connection schema has to be checked there — on
    a local declaration the remote-only refusal now fires first, which would
    make this test pass for the wrong reason."""
    with pytest.raises(ValueError, match="extra"):
        validate_engines([_omni_entry(
            connection={"url": "http://127.0.0.1:8008", "container": "omni"})],
            remote=True)


# --- the four busy cases, one test each ------------------------------------
# ([[defaults-that-hide-bugs]]: indicator present-and-nonzero, present-zero,
# absent, and error are four DIFFERENT inputs with three different answers —
# collapsing them into one parametrized case would let "absent" ride on
# "present-zero"'s assertion.)


def test_sglang_omni_healthy_with_requests_in_flight_observes_busy():
    obs = _observe(_FakeOmni(busy_requests=2))

    assert obs["state"] == "busy"
    assert obs["busy_requests"] == 2


def test_sglang_omni_healthy_with_no_requests_in_flight_observes_idle():
    obs = _observe(_FakeOmni(busy_requests=0))

    assert obs["state"] == "idle"
    assert obs["busy_requests"] == 0


def test_sglang_omni_healthy_with_an_unavailable_busy_indicator_observes_busy():
    """design §4 / GF3: the busy signal is an agent-side established-
    connection count, and `None` means the agent could not take it. An
    unavailable indicator fails toward ALIVE — reading it as 0 would let
    idle-release kill a render that has been running for minutes."""
    obs = _observe(_FakeOmni(busy_requests=None))

    assert obs["state"] == "busy"
    assert obs["busy_requests"] is None


def test_sglang_omni_status_error_observes_unknown():
    """A failed probe is "we failed to look", never "nothing is loaded"."""
    obs = _observe(_FakeOmni(raises=EngineError("nimbus is not answering")))

    assert obs == _omni().unknown()
    assert obs["state"] == "unknown"


# --- the rest of the state map ---------------------------------------------


def test_sglang_omni_reachable_but_not_healthy_observes_down():
    """The engine answered the socket but /health is not 200 — observed,
    not merely unseen. `down` is what intent-loaded turns into a restore."""
    obs = _observe(_FakeOmni(reachable=True, healthy=False, busy_requests=0))

    assert obs["state"] == "down"


def test_sglang_omni_unreachable_engine_observes_down_not_unknown():
    """The AGENT answered — it is the ENGINE that is not up. That is a fact
    we observed, so it must not degrade to `unknown` (which means the deck
    could not look at all, and which the reconciler never acts on)."""
    obs = _observe(_FakeOmni(reachable=False, healthy=False, busy_requests=None))

    assert obs["state"] == "down"
    assert obs["busy_requests"] is None


def test_sglang_omni_never_reports_a_model_name():
    """GF2: the engine's own model id is `/model` — the container's mount
    path, not an identity. Never trusted, and the declaration carries no
    model either, so identity is None exactly as it is for every other
    single-model engine (derive_status's "loaded, no opinion which model"
    branch)."""
    assert _observe(_FakeOmni(busy_requests=0))["model"] is None
    assert _observe(_FakeOmni(busy_requests=7))["model"] is None


# --- the idle clock --------------------------------------------------------


def test_sglang_omni_idle_seconds_accrue_while_idle():
    mem = {}
    _observe(_FakeOmni(busy_requests=0), mem, now=1000.0)

    obs = _observe(_FakeOmni(busy_requests=0), mem, now=1120.0)

    assert obs["idle_s"] == 120.0


def test_sglang_omni_a_request_in_flight_rearms_the_idle_clock():
    """The render-protection invariant at the clock level: a song that
    starts after 10 minutes of idle resets the countdown, so idle-release
    can never fire on the tick after a render began."""
    mem = {}
    _observe(_FakeOmni(busy_requests=0), mem, now=1000.0)
    _observe(_FakeOmni(busy_requests=1), mem, now=1600.0)

    obs = _observe(_FakeOmni(busy_requests=0), mem, now=1605.0)

    assert obs["idle_s"] == 5.0


def test_sglang_omni_an_unavailable_busy_indicator_rearms_the_idle_clock():
    """Same re-arm as a real in-flight request: `None` is treated as busy
    everywhere, including the clock — otherwise a run of unavailable
    indicators would quietly accrue the idle time that authorizes an
    unload."""
    mem = {}
    _observe(_FakeOmni(busy_requests=0), mem, now=1000.0)
    _observe(_FakeOmni(busy_requests=None), mem, now=1600.0)

    obs = _observe(_FakeOmni(busy_requests=0), mem, now=1605.0)

    assert obs["idle_s"] == 5.0


def test_sglang_omni_a_down_engine_rearms_the_idle_clock():
    """No instant-evict of a just-booted engine: an engine that was DOWN for
    ten minutes and has only now come up healthy has been idle for seconds,
    not minutes. Accruing across the down window would put it past any sane
    idle_ttl on its very first healthy tick — the deck would unload what it
    just spent ~4 minutes (GF4) booting."""
    mem = {}
    _observe(_FakeOmni(busy_requests=0), mem, now=1000.0)
    _observe(_FakeOmni(reachable=False, healthy=False, busy_requests=None),
             mem, now=1600.0)

    obs = _observe(_FakeOmni(busy_requests=0), mem, now=1602.0)

    assert obs["idle_s"] == 2.0


def test_sglang_omni_a_failed_probe_rearms_the_idle_clock():
    """Same rule for the unknown arm: time we could not observe is not time
    we observed it idle."""
    mem = {}
    _observe(_FakeOmni(busy_requests=0), mem, now=1000.0)
    _observe(_FakeOmni(raises=EngineError("nimbus is not answering")),
             mem, now=1600.0)

    obs = _observe(_FakeOmni(busy_requests=0), mem, now=1602.0)

    assert obs["idle_s"] == 2.0


def test_sglang_omni_down_and_unknown_carry_no_idle_seconds():
    """`idle_s` is None, not 0.0, when "idle" is not a meaningful notion —
    0.0 would read as a real (freshly re-armed) clock."""
    assert _observe(_FakeOmni(healthy=False))["idle_s"] is None
    assert _omni().unknown()["idle_s"] is None


# --- verbs, activity, demand ----------------------------------------------


def test_sglang_omni_adapter_active_verbs_and_demand():
    omni = _omni()
    assert omni.active(_obs("busy", 1)) is True
    assert omni.active(_obs("idle", 0)) is True
    assert omni.active(_obs("down", None)) is False
    assert omni.active(omni.unknown()) is False
    assert omni.arbiter_verbs() == frozenset({"unload"})
    assert omni.human_verbs() == frozenset({"load", "unload"})
    # No pending-load inference for this kind: nothing routes chat traffic
    # at it, so there is no "someone asked for a model that isn't loaded"
    # signal to demand one from.
    assert omni.demand() is False


def test_sglang_omni_idle_still_counts_as_active():
    """An idle engine still holds ~62 GiB (GF5) — idle describes its
    REQUEST queue, not its residency, the same reading comfyui's adapter
    already documents."""
    assert _omni().active(_obs("idle", 0)) is True


def test_sglang_omni_never_uses_a_gguf_and_needs_no_container_restart():
    omni = _omni()
    assert omni.uses_gguf(_obs("idle", 0)) is None
    assert omni.restart_container(_omni_entry()) is None


# --- reclaimable: the busy-signal invariant lives HERE ---------------------


def test_sglang_omni_reclaimable_is_the_idle_footprint_when_idle_and_unbusy():
    assert _omni().reclaimable(_obs("idle", 0), None, 0) == _OMNI_IDLE_FOOTPRINT
    # Non-zero matters: idle_action's `reclaimable != 0` clause would
    # suppress every unload if this were 0.
    assert _OMNI_IDLE_FOOTPRINT > 0


def test_sglang_omni_reclaimable_is_none_while_busy():
    assert _omni().reclaimable(_obs("busy", 3), None, 0) is None


def test_sglang_omni_reclaimable_is_none_when_the_busy_indicator_is_unavailable():
    """The design §4 invariant, pinned at the ONE place it lives: an
    observation that says idle but carries no usable busy count is not a
    reclaim candidate. Asserted against a hand-built obs (not one observe()
    can produce) precisely so the guard is proven in `reclaimable` itself
    rather than inherited from observe's state mapping — a future call site
    that builds its own obs still cannot get an eviction out of it."""
    assert _omni().reclaimable(_obs("idle", None), None, 0) is None


def test_sglang_omni_reclaimable_is_none_when_down_or_unknown():
    omni = _omni()
    assert omni.reclaimable(_obs("down", None), None, 0) is None
    assert omni.reclaimable(omni.unknown(), None, 0) is None


def test_sglang_omni_reclaimable_ignores_the_gpu_snapshot():
    """Unlike comfyui's, this kind's footprint is DECLARED (GF5), not
    inferred from a GPU's unexplained usage — the GPU pool belongs to
    another machine, and co-resident footprints there are that box's own
    arithmetic. Both are accepted for interface uniformity and ignored."""
    assert _omni().reclaimable(
        _obs("idle", 0), {"index": 4, "total": 1, "used": 1, "free": 0},
        99 * 1024**3) == _OMNI_IDLE_FOOTPRINT


# --- idle_action: the full predicate, one clause per test ------------------


def test_sglang_omni_idle_action_unloads_once_past_the_ttl():
    action = _omni().idle_action(_obs("idle", 0, idle_s=121.0), _policy(), None, 0)

    assert action == {"type": "unload", "model": None}


def test_sglang_omni_idle_action_is_none_before_the_ttl():
    assert _omni().idle_action(_obs("idle", 0, idle_s=119.0), _policy(), None, 0) is None


def test_sglang_omni_idle_action_is_none_when_pinned():
    assert _omni().idle_action(_obs("idle", 0, idle_s=1000.0),
                               _policy(pinned=True), None, 0) is None


def test_sglang_omni_idle_action_is_none_when_the_ttl_is_zero():
    """idle_ttl 0 means "never idle-release" (the vocabulary every other
    kind already uses), not "release immediately"."""
    assert _omni().idle_action(_obs("idle", 0, idle_s=1000.0),
                               _policy(idle_ttl=0), None, 0) is None


def test_sglang_omni_idle_action_is_none_while_busy():
    assert _omni().idle_action(_obs("busy", 4, idle_s=1000.0), _policy(),
                               None, 0) is None


def test_sglang_omni_idle_action_is_none_when_the_busy_indicator_is_unavailable():
    """THE render-protection headline (design §4): a long-idle-looking
    observation whose busy count is unavailable must never authorize an
    unload — a five-minute song render is exactly the thing that would be
    killed."""
    assert _omni().idle_action(_obs("idle", None, idle_s=1000.0), _policy(),
                               None, 0) is None


def test_sglang_omni_idle_action_is_none_when_down_or_unknown():
    omni = _omni()
    assert omni.idle_action(_obs("down", None, idle_s=1000.0), _policy(), None, 0) is None
    assert omni.idle_action(omni.unknown(), _policy(), None, 0) is None


# --- actuation: whoever actuates, records — and records FIRST --------------


class _RecordingWorld:
    """The one World method an actuator calls (app.state.World.note_freed),
    recording the (resource, node_id) pair it was re-armed for."""

    def __init__(self) -> None:
        self.rearmed = []

    def note_freed(self, resource, node_id=None):
        self.rearmed.append((resource, node_id))


class _RecordingObserver:
    """The one RemoteObserver method an actuator calls."""

    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1


class _RecordingWatcher:
    """The slice of app.arbiter.Watcher the adapter actuators reach into.

    `_world`/`_remote_observer` joined it with Task 9: a successful remote
    unload re-arms that resource's idle clock and drops the cached remote
    half, the two obligations a comfyui-kind free already had (`note_freed`)
    sharpened by the observation being TTL-cached off-box. `_remote_observer`
    is None-able there, so this fake supplies a real recorder and the
    None case gets its own test below."""

    def __init__(self, intent_store, remote_observer=None) -> None:
        self._intent_store = intent_store
        self._world = _RecordingWorld()
        self._remote_observer = remote_observer
        self.logs = []
        self.dedup_clears = 0

    def _log(self, kind, detail):
        self.logs.append((kind, detail))

    def _clear_failure_dedup(self):
        self.dedup_clears += 1


class _IntentReadingOmni(_FakeOmni):
    """Reads the intent store the moment the verb lands, so "recorded
    BEFORE the engine call" is proven by ORDER, not by a post-hoc read that
    would pass either way."""

    def __init__(self, store, **kw):
        super().__init__(**kw)
        self._store = store
        self.intent_at_call = None

    def _snapshot(self):
        self.intent_at_call = self._store.get().get(_OMNI_KEY)

    def up(self):
        self._snapshot()
        super().up()

    def down(self):
        self._snapshot()
        super().down()


def _intent_store(tmp_path):
    from app.intent import IntentStore

    return IntentStore(tmp_path / "intent.json")


def test_sglang_omni_execute_unload_records_the_node_keyed_intent_first(tmp_path):
    store = _intent_store(tmp_path)
    watcher = _RecordingWatcher(store)
    client = _IntentReadingOmni(store)
    actuated = set()

    _omni().execute_unload(watcher, "song-r", client, None, actuated,
                           node_id="nimbus")

    assert client.calls == ["down"]
    assert client.intent_at_call["state"] == "unloaded"     # recorded FIRST
    record = store.get()[_OMNI_KEY]
    assert record["engine"] == "sglang-omni"
    assert record["actor"] == "deck"
    assert actuated == {_OMNI_KEY}


def test_sglang_omni_execute_unload_rolls_the_intent_back_when_the_engine_refuses(tmp_path):
    """A refused unload did not happen, so leaving `unloaded` standing would
    derive the inert 'unexpected' against a still-running engine."""
    store = _intent_store(tmp_path)
    store.record(_OMNI_KEY, state="loaded", model=None, engine="sglang-omni",
                 actor="operator")
    watcher = _RecordingWatcher(store)
    client = _FakeOmni(verb_raises=EngineError("agent said no"))

    _omni().execute_unload(watcher, "song-r", client, None, set(),
                           node_id="nimbus")

    assert store.get()[_OMNI_KEY]["state"] == "loaded"
    assert [kind for kind, _d in watcher.logs] == ["unload-failed"]
    assert watcher.logs[0][1]["resource"] == "song-r"
    assert watcher.logs[0][1]["node"] == "nimbus"


class _SupersedingOmni(_FakeOmni):
    """An operator records a NEW intent while the unload hangs, then the
    unload fails — the compare-and-swap case."""

    def __init__(self, store, **kw):
        super().__init__(**kw)
        self._store = store

    def down(self):
        self._store.record(_OMNI_KEY, state="loaded", model=None,
                           engine="sglang-omni", actor="operator")
        super().down()


def test_sglang_omni_execute_unload_rollback_is_skipped_when_the_intent_changed(tmp_path):
    """Compare-and-swap, not check-then-act: blindly putting the prior
    record back would silently revert an operator's deliberate action taken
    during the seconds the engine call hung."""
    store = _intent_store(tmp_path)
    store.record(_OMNI_KEY, state="unloaded", model=None, engine="sglang-omni",
                 actor="operator")
    watcher = _RecordingWatcher(store)
    client = _SupersedingOmni(store, verb_raises=EngineError("agent said no"))

    _omni().execute_unload(watcher, "song-r", client, None, set(),
                           node_id="nimbus")

    assert store.get()[_OMNI_KEY]["state"] == "loaded"      # operator's, kept
    assert [kind for kind, _d in watcher.logs] == [
        "unload-failed", "unload-rollback-skipped"]


def test_sglang_omni_execute_unload_rearms_the_failure_dedup_on_success(tmp_path):
    """fail -> recover -> fail must not swallow the second failure."""
    store = _intent_store(tmp_path)
    watcher = _RecordingWatcher(store)

    _omni().execute_unload(watcher, "song-r", _FakeOmni(), None, set(),
                           node_id="nimbus")

    assert watcher.dedup_clears == 1


def test_sglang_omni_execute_unload_rearms_the_clock_and_drops_the_cache(tmp_path):
    """Task 9: the comfyui-kind free's own two obligations, on a REMOTE
    engine. `down()` is a 202 and the observation is TTL-cached, so without
    both the same idle-past-ttl record drives one unload dispatch per tick
    until the container finally disappears (the free-spam lesson, with a
    whole engine at the other end). Re-armed for THIS node's key, never the
    local resource of the same name."""
    store = _intent_store(tmp_path)
    observer = _RecordingObserver()
    watcher = _RecordingWatcher(store, remote_observer=observer)

    _omni().execute_unload(watcher, "song-r", _FakeOmni(), None, set(),
                           node_id="nimbus")

    assert watcher._world.rearmed == [("song-r", "nimbus")]
    assert observer.invalidations == 1


def test_sglang_omni_execute_unload_does_neither_when_the_engine_refuses(tmp_path):
    """A refused unload changed nothing, so it must not re-arm the idle
    clock (which would delay the retry by a whole idle_ttl) nor spend the
    deck-wide cache drop — the same success-path-only discipline
    `_ComfyAdapter.execute_free` keeps for `note_freed`."""
    store = _intent_store(tmp_path)
    observer = _RecordingObserver()
    watcher = _RecordingWatcher(store, remote_observer=observer)

    _omni().execute_unload(watcher, "song-r",
                           _FakeOmni(verb_raises=EngineError("agent said no")),
                           None, set(), node_id="nimbus")

    assert watcher._world.rearmed == []
    assert observer.invalidations == 0


def test_sglang_omni_execute_unload_survives_a_watcher_without_an_observer(tmp_path):
    """`remote_observer=None` is a real Watcher state (every caller built
    before Task 7, and every unit test that wires no remote half): the
    actuation still happens, with nothing to invalidate."""
    store = _intent_store(tmp_path)
    watcher = _RecordingWatcher(store)
    client = _FakeOmni()

    _omni().execute_unload(watcher, "song-r", client, None, set(),
                           node_id="nimbus")

    assert client.calls == ["down"]
    assert watcher._world.rearmed == [("song-r", "nimbus")]


def test_sglang_omni_execute_load_records_the_intent_first_and_calls_up(tmp_path):
    store = _intent_store(tmp_path)
    watcher = _RecordingWatcher(store)
    client = _IntentReadingOmni(store)
    actuated = set()

    _omni().execute_load(watcher, "song-r", client, {"model": None}, actuated,
                         node_id="nimbus")

    assert client.calls == ["up"]
    assert client.intent_at_call["state"] == "loaded"
    assert store.get()[_OMNI_KEY]["actor"] == "deck"
    assert actuated == {_OMNI_KEY}


def test_sglang_omni_execute_load_leaves_the_intent_standing_when_the_engine_refuses(tmp_path):
    """Deliberate, and the same call the lemonade-kind load arm makes: a
    failed load derives 'down' and the reconciler retries it under the
    existing failure budget. Rolling back here would make the deck forget it
    ever wanted the engine up."""
    store = _intent_store(tmp_path)
    watcher = _RecordingWatcher(store)

    _omni().execute_load(watcher, "song-r", _FakeOmni(verb_raises=EngineError("no")),
                         {"model": None}, set(), node_id="nimbus")

    assert store.get()[_OMNI_KEY]["state"] == "loaded"
    assert [kind for kind, _d in watcher.logs] == ["load-failed"]


def test_sglang_omni_restore_brings_the_engine_up():
    client = _FakeOmni()

    _omni().restore(client, None)

    assert client.calls == ["up"]


# --- construction ----------------------------------------------------------


def test_sglang_omni_build_remote_client_binds_the_node_and_the_resource():
    from app.engines.sglang_omni import SglangOmniClient

    client = _omni().build_remote_client(
        "http://nimbus:7720", "key-nimbus", "song-r",
        {"url": "http://127.0.0.1:8008"})

    try:
        assert isinstance(client, SglangOmniClient)
    finally:
        client.close()


def test_sglang_omni_has_no_local_client_and_says_so(tmp_path):
    """`build_client` exists for protocol completeness and REFUSES: this
    kind speaks the node-agent wire, so there is nothing to build for a
    declaration on the local entry. Refusing by name beats the bare
    AttributeError a missing method would raise inside a world snapshot."""
    with pytest.raises(ValueError, match="sglang-omni"):
        _omni().build_client({"url": "http://127.0.0.1:8008"}, object(), None)


# --- warming (GF4): the rule, ahead of the reconcile wiring ----------------


def _loaded_intent(updated_ts):
    return {"state": "loaded", "model": None, "engine": "sglang-omni",
            "actor": "operator", "updated_ts": updated_ts}


def test_sglang_omni_is_warming_while_a_recent_load_is_still_booting():
    """GF4: a cold start takes ~3.5-4.5 minutes, and the whole time the
    engine observes exactly like one that died. Within the window, a
    loaded-intent + down observation is a boot in flight."""
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    intent = _loaded_intent((now - timedelta(seconds=300)).isoformat())

    assert _omni().warming(intent, _obs("down", None), now) is True


def test_sglang_omni_is_not_warming_once_the_window_lapses():
    """The condition SparkObserver's boot_in_flight exists to enforce: a
    stale "we asked it to load" must not shield a model that died hours
    later from restore forever (the 26-hour hipfire failure)."""
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    intent = _loaded_intent((now - timedelta(seconds=601)).isoformat())

    assert _omni().warming(intent, _obs("down", None), now) is False


def test_sglang_omni_is_not_warming_without_a_loaded_intent():
    from datetime import UTC, datetime

    now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

    assert _omni().warming(None, _obs("down", None), now) is False
    assert _omni().warming({**_loaded_intent(now.isoformat()), "state": "unloaded"},
                           _obs("down", None), now) is False


def test_sglang_omni_is_not_warming_once_it_is_healthy():
    from datetime import UTC, datetime

    now = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    intent = _loaded_intent(now.isoformat())

    assert _omni().warming(intent, _obs("idle", 0), now) is False
    assert _omni().warming(intent, _obs("busy", 1), now) is False


# ===========================================================================
# Fix round 1 — review findings 2 and 3.
# ===========================================================================


def test_sglang_omni_malformed_status_body_observes_unknown():
    """Finding 3, at the consumer: a 200 the deck cannot read is "we failed
    to look", never a crash. The client refuses the body (EngineError) and
    observe's existing narrow catch heals it — the same one arm, so there is
    no second definition of "malformed" to drift."""
    class _MalformedOmni:
        def status(self):
            raise EngineError("sglang-omni engine 'song-r' status: "
                              "malformed body (missing 'healthy')")

    obs = _observe(_MalformedOmni())

    assert obs == _omni().unknown()


def test_known_kinds_declare_where_each_kind_may_RUN():
    """Finding 2: `remote_capable` alone described only ONE direction, so
    nothing refused a remote-only kind on the LOCAL entry. Both directions
    are declared now, and every kind must be runnable SOMEWHERE (a kind
    with both False could be declared nowhere and would be dead weight in
    the picker).

    INST I1 Task 1: the key set grew by four (max_gpus/instance/
    instance_env/instance_policy, the per-kind instance-capability
    descriptors) — asserted here as a superset rather than splitting this
    test, since the two halves (run-location vs. instance-capability) are
    proven together against every kind anyway."""
    for kind, spec in KNOWN_KINDS.items():
        assert set(spec) == {"connection", "remote_capable", "local_capable",
                             "max_gpus", "instance", "instance_env", "instance_policy"}, kind
        assert isinstance(spec["local_capable"], bool), kind
        assert spec["remote_capable"] or spec["local_capable"], kind


def test_sglang_omni_is_the_only_remote_only_kind():
    assert [k for k, s in KNOWN_KINDS.items() if not s["local_capable"]] == [
        "sglang-omni"]


def test_validate_engines_refuses_a_remote_only_kind_on_a_local_declaration():
    """Guard at the BOUNDARY (finding 2): before this, a local sglang-omni
    declaration was accepted here and only refused later, when
    `build_client` was asked for a client that cannot exist — which surfaced
    as a 422 on /api/state (and a tick-error every ~2 s) rather than a 422
    on the declaration that caused it."""
    with pytest.raises(ValueError, match="sglang-omni") as exc:
        validate_engines([_omni_entry()])

    assert "remote-only" in str(exc.value)
    # ...and it says where it CAN be declared, so the refusal is actionable.
    assert "node-agent" in str(exc.value)


@pytest.mark.parametrize("kind, connection", [
    ("lemonade", {"url": "http://u", "metrics_url": "http://m",
                  "container": "c"}),
    ("comfyui", {"url": "http://u"}),
    ("hipfire", {"container": "c"}),
])
def test_validate_engines_still_accepts_every_local_kind_locally(kind, connection):
    """The other half of finding 2's fix: the new gate refuses ONE kind, it
    does not make local declarations stricter in general."""
    validate_engines([_entry(kind=kind, connection=connection)])


def test_validate_engines_local_gate_is_the_mirror_of_the_remote_one():
    """The two gates are symmetric and independent: the same entry is
    refused locally and accepted remotely, and a lemonade entry is the
    reverse. Proven together so a future edit cannot silently collapse them
    into one flag."""
    validate_engines([_omni_entry()], remote=True)
    with pytest.raises(ValueError, match="remote-only"):
        validate_engines([_omni_entry()], remote=False)
    validate_engines([_entry()], remote=False)
    with pytest.raises(ValueError, match="not remote_capable"):
        validate_engines([_entry()], remote=True)


# ===========================================================================
# INST I1 Task 1 — instance capability descriptors, gpu_indices, managed/
# port/env
#
# Fixture rule (design §8 + 2026-08-12 topology review): resource "gguf-a"
# (never a kind name), GPUs 2/3/4 (never 0/1).
# ===========================================================================

from app.engine_kinds import (KNOWN_KINDS, validate_engines, gpu_indices_of,
                              instance_connection, instance_policy, instance_env_schema,
                              max_gpus, kind_instantiable, INSTANCE_INTERNAL_PORT,
                              INSTANCE_CONTAINER_PREFIX)

_LEM = {"url": "http://gguf-a:8080", "metrics_url": "http://gguf-a:8001/metrics",
        "container": "deck-gguf-a"}
_POL = {"priority": 0, "pinned": False, "idle_ttl": 0}


def _decl(**over):
    base = {"resource": "gguf-a", "kind": "lemonade", "connection": dict(_LEM),
            "gpu_indices": [3], "container_consent": True, "policy_defaults": dict(_POL)}
    base.update(over)
    return base


def test_every_kind_declares_the_four_instance_capability_keys():
    for kind, spec in KNOWN_KINDS.items():
        assert {"max_gpus", "instance", "instance_env", "instance_policy"} <= set(spec), kind
        assert isinstance(spec["instance"], bool)
        assert spec["max_gpus"] is None or (isinstance(spec["max_gpus"], int) and spec["max_gpus"] >= 1)
        assert set(spec["instance_policy"]) == {"priority", "pinned", "idle_ttl"}


def test_instance_capability_per_kind_matches_the_design():
    # design §3: hipfire 1 (single-card by necessity), comfyui 1, llama-server N, sglang-omni not instantiable
    assert (max_gpus("hipfire"), kind_instantiable("hipfire")) == (1, True)
    assert (max_gpus("comfyui"), kind_instantiable("comfyui")) == (1, True)
    assert (max_gpus("lemonade"), kind_instantiable("lemonade")) == (None, True)
    assert kind_instantiable("sglang-omni") is False


def test_gpu_indices_of_reads_either_spelling():
    assert gpu_indices_of({"gpu_index": 2}) == [2]
    assert gpu_indices_of({"gpu_indices": [3, 4]}) == [3, 4]


def test_validate_engines_accepts_gpu_indices_list_and_refuses_both_spellings():
    validate_engines([_decl()])
    validate_engines([_decl(gpu_indices=[3, 4])])
    import pytest
    with pytest.raises(ValueError, match="exactly one of gpu_index or gpu_indices"):
        validate_engines([_decl(gpu_index=3)])          # both present (gpu_indices from _decl)
    with pytest.raises(ValueError, match="exactly one of gpu_index or gpu_indices"):
        d = _decl(); del d["gpu_indices"]; validate_engines([d])


def test_validate_engines_gpu_indices_shape():
    import pytest
    for bad, why in (([], "non-empty"), ([3, 3], "unique"), ([4, 3], "sorted"),
                     ([True], "non-negative integer"), ([-1], "non-negative integer"),
                     ("3", "list")):
        with pytest.raises(ValueError, match=why):
            validate_engines([_decl(gpu_indices=bad)])


def test_validate_engines_enforces_max_gpus_per_kind():
    import pytest
    with pytest.raises(ValueError, match="kind 'hipfire' allows at most 1 GPU"):
        validate_engines([_decl(kind="hipfire", connection={"container": "deck-agent"},
                                gpu_indices=[3, 4])])


def test_validate_engines_managed_requires_port_and_env_and_refuses_them_otherwise():
    import pytest
    validate_engines([_decl(managed=True, port=11500, env={})])
    with pytest.raises(ValueError, match="managed entry requires port"):
        validate_engines([_decl(managed=True, env={})])
    with pytest.raises(ValueError, match="port must be an int in 1024-65535"):
        validate_engines([_decl(managed=True, port=80, env={})])
    with pytest.raises(ValueError, match="port/env are only valid on a managed entry"):
        validate_engines([_decl(port=11500)])
    with pytest.raises(ValueError, match="managed must be a boolean"):
        validate_engines([_decl(managed="yes")])


def test_validate_engines_env_follows_the_kinds_instance_env_schema():
    import pytest
    hip = {"resource": "agent", "kind": "hipfire", "connection": {"container": "deck-agent"},
           "gpu_indices": [2], "container_consent": True, "policy_defaults": dict(_POL),
           "managed": True, "port": 11501}
    validate_engines([{**hip, "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}}])
    with pytest.raises(ValueError, match="env.HIPFIRE_MODEL is required for kind 'hipfire'"):
        validate_engines([{**hip, "env": {}}])
    with pytest.raises(ValueError, match="env has unknown field"):
        validate_engines([{**hip, "env": {"HIPFIRE_MODEL": "x", "NOPE": "1"}}])
    with pytest.raises(ValueError, match="env.HIPFIRE_MODEL must be a non-empty string"):
        validate_engines([{**hip, "env": {"HIPFIRE_MODEL": ""}}])


def test_hipfire_connection_accepts_optional_gateway_host():
    validate_engines([{"resource": "agent", "kind": "hipfire", "gpu_indices": [2],
                       "connection": {"container": "deck-agent", "gateway_host": "agent"},
                       "container_consent": True, "policy_defaults": dict(_POL)}])


def test_instance_connection_and_ports_per_kind():
    assert instance_connection("hipfire", "agent") == {
        "container": "deck-agent", "gateway_host": "agent"}
    assert instance_connection("lemonade", "gguf-a") == {
        "url": "http://gguf-a:8080", "metrics_url": "http://gguf-a:8001/metrics",
        "container": "deck-gguf-a"}
    assert instance_connection("comfyui", "img") == {
        "url": "http://img:8188", "container": "deck-img"}
    assert INSTANCE_CONTAINER_PREFIX == "deck-"
    assert INSTANCE_INTERNAL_PORT == {"hipfire": 11435, "lemonade": 8080, "comfyui": 8188}
    # every instance connection validates under its own kind's schema
    for kind, res in (("hipfire", "agent"), ("lemonade", "gguf-a"), ("comfyui", "img")):
        validate_engines([{"resource": res, "kind": kind, "gpu_indices": [4],
                           "connection": instance_connection(kind, res),
                           "container_consent": True, "policy_defaults": instance_policy(kind),
                           "managed": True, "port": 11502,
                           "env": {k: "v" for k, req in instance_env_schema(kind).items() if req}}])


def test_instance_policy_returns_a_copy():
    a = instance_policy("lemonade"); a["priority"] = 999
    assert instance_policy("lemonade")["priority"] != 999


def test_no_reader_of_gpu_index_outside_the_accessor():
    """Single seam (D-I1-2): app/ reads a declaration's GPU claim ONLY via
    gpu_indices_of(). The derived tenant field `obs["gpu_index"]` written by
    state.py and read by arbiter's tenant_gpus() is the ONE other spelling
    and is listed here explicitly."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    allowed = {"engine_kinds.py", "state.py", "arbiter.py", "settings.py", "node_store.py"}
    hits = []
    for path in root.rglob("*.py"):
        if path.name in allowed:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'\["gpu_index"\]|\.get\("gpu_index"\)', line):
                hits.append(f"{path.relative_to(root)}:{i}")
    assert hits == [], hits
