"""Tests for app.sets — config sets: schema, slug store, diff planner, apply engine.

A ConfigSet is user-authored JSON (the one place Model Deck uses pydantic to
validate it). ``plan_apply`` is a PURE diff of a set against a world snapshot,
emitting only the steps that change reality, in a fixed safety order. ``apply``
is the imperative shell: it snapshots pre-apply reality as the ``_previous``
revert set FIRST, executes the plan serialized under a module lock, and halts
on the first failing step with an exact report. Clients are recording fakes.
"""

import threading

import pytest
from pydantic import ValidationError

from app.arbiter import HealSuppressor
from app.engines import BusyError, EngineError, GuardError
from app.events import tail_events
from app.policy import DEFAULT_POLICIES, PolicyStore
from app.sets import (
    PREVIOUS_NAME,
    RESERVED_SLUG,
    ConfigSet,
    SetStore,
    apply,
    apply_in_progress,
    plan_apply,
    slugify,
)


# ===========================================================================
# Fakes + builders
# ===========================================================================


def make_world(
    *,
    lemonade=("unloaded", None),
    comfy=("idle", 0),
    hipfire="running",
    default_route=None,
):
    lem_state, lem_model = lemonade
    comfy_state, comfy_queue = comfy
    return {
        "gpus": [],
        "tenants": {
            "lemonade": {
                "state": lem_state,
                "model": lem_model,
                "footprint": None,
                "idle_s": None,
            },
            "comfyui": {"state": comfy_state, "queue": comfy_queue, "idle_s": None},
            "hipfire": {"state": hipfire, "model": None, "footprint": 0},
        },
        "externals": [],
        "default_route": default_route,
    }


class RecLemonade:
    def __init__(self):
        self.calls = []
        self.fail = {}

    def load(self, model):
        self.calls.append(("load", model))
        if "load" in self.fail:
            raise self.fail["load"]

    def unload(self, model):
        self.calls.append(("unload", model))
        if "unload" in self.fail:
            raise self.fail["unload"]


class RecComfy:
    def __init__(self):
        self.calls = []
        self.fail = None

    def free(self):
        self.calls.append("free")
        if self.fail:
            raise self.fail


class RecHipfire:
    def __init__(self):
        self.calls = []
        self.fail = {}
        self.busy_checks = []  # action strings passed to ensure_not_busy
        self.park_forces = []  # force flag per park call

    def ensure_not_busy(self, action):
        self.busy_checks.append(action)
        spec = self.fail.get("ensure_not_busy")
        if isinstance(spec, list):  # successive per-call behaviors; None = pass
            exc = spec.pop(0) if spec else None
        else:
            exc = spec
        if exc is not None:
            raise exc

    def park(self, force=False):
        self.calls.append("park")
        self.park_forces.append(force)
        if "park" in self.fail:
            raise self.fail["park"]

    def resume(self):
        self.calls.append("resume")
        if "resume" in self.fail:
            raise self.fail["resume"]


class RecHostAgent:
    def __init__(self):
        self.calls = []
        self.fail = None

    def activate(self, model_id):
        self.calls.append(model_id)
        if self.fail:
            raise self.fail
        return {"activated": model_id}

    def lifecycle(self):
        # Idle by default — every existing apply() test that doesn't
        # explicitly care about the host-agent guard must see "not busy".
        return {"active": False, "operation": None, "target": None}


class _BusyHostAgent:
    """Mirrors test_arbiter.py's I2 busy-lifecycle fixture (``_BusyHostAgent``
    there). Kept as a local copy — importing a 3-line fake across test
    modules would be more awkward than just mirroring it."""

    def lifecycle(self):
        return {"active": True, "operation": "model_activation", "target": "qwen3-30b"}

    def activate(self, model_id):  # pragma: no cover - must never run
        raise AssertionError("hostagent.activate must not run past a busy veto")


class RecPolicyStore:
    def __init__(self, current=None):
        self.calls = []
        self.fail = None
        self._current = (
            current
            if current is not None
            else {tenant: dict(pol) for tenant, pol in DEFAULT_POLICIES.items()}
        )

    def get(self):
        return {tenant: dict(pol) for tenant, pol in self._current.items()}

    def put(self, policies):
        self.calls.append(policies)
        if self.fail:
            raise self.fail


def run_apply(cfgset, world, tmp_path, **overrides):
    """Invoke apply with recording fakes; returns (report, clients-dict)."""
    clients = {
        "lemonade": RecLemonade(),
        "comfy": RecComfy(),
        "hipfire": RecHipfire(),
        "hostagent": RecHostAgent(),
        "policy_store": RecPolicyStore(),
        "store": SetStore(tmp_path / "sets"),
        "events_path": tmp_path / "events.jsonl",
    }
    clients.update(overrides)
    report = apply(cfgset, world=world, **clients)
    return report, clients


# ===========================================================================
# ConfigSet schema (pydantic)
# ===========================================================================


def test_minimal_configset_defaults():
    cfg = ConfigSet(name="Fresh")
    assert cfg.name == "Fresh"
    assert cfg.notes == ""
    assert cfg.durable is None
    assert cfg.ephemeral is None
    assert cfg.policy_overrides is None


def test_name_required_nonempty():
    with pytest.raises(ValidationError):
        ConfigSet(name="")


def test_name_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        ConfigSet(name="   ")


def test_name_is_trimmed():
    assert ConfigSet(name="  Image session  ").name == "Image session"


def test_comfyui_reserve_gb_defaults_to_24():
    cfg = ConfigSet(name="x", ephemeral={"comfyui": {"state": "free"}})
    assert cfg.ephemeral.comfyui.reserve_gb == 24


def test_ephemeral_rejects_bad_lemonade_state():
    with pytest.raises(ValidationError):
        ConfigSet(name="x", ephemeral={"lemonade": {"state": "sleeping"}})


def test_durable_activate_model_id_optional():
    cfg = ConfigSet(name="x", durable={"default_route_model": "extra.m.gguf"})
    assert cfg.durable.default_route_model == "extra.m.gguf"
    assert cfg.durable.activate_model_id is None


def test_configset_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        ConfigSet(name="x", bogus=1)


# ===========================================================================
# slugify
# ===========================================================================


def test_slugify_spaces_to_dashes():
    assert slugify("Image session") == "image-session"


def test_slugify_collapses_punct_runs_and_trims():
    assert slugify("  Hello,  World!! ") == "hello-world"


def test_slugify_keeps_digits():
    assert slugify("GPT4 Turbo v2") == "gpt4-turbo-v2"


def test_slugify_slashes_and_symbols():
    assert slugify("Model A/B (fast)") == "model-a-b-fast"


def test_slugify_previous_display_name_collapses_to_previous():
    assert slugify(PREVIOUS_NAME) == "previous"


def test_slugify_all_punct_is_empty():
    assert slugify("!!! ???") == ""


# ===========================================================================
# SetStore CRUD
# ===========================================================================


def test_save_returns_slug_and_writes_file(tmp_path):
    store = SetStore(tmp_path / "sets")
    slug = store.save(ConfigSet(name="Image Session"))
    assert slug == "image-session"
    assert (tmp_path / "sets" / "image-session.json").is_file()


def test_save_get_roundtrip(tmp_path):
    store = SetStore(tmp_path / "sets")
    cfg = ConfigSet(
        name="Chat",
        notes="for talking",
        durable={"default_route_model": "extra.m.gguf", "activate_model_id": "cat-1"},
        ephemeral={"comfyui": {"state": "free", "reserve_gb": 12}},
        policy_overrides={"lemonade": {"priority": 5, "pinned": False, "idle_ttl": 10}},
    )
    slug = store.save(cfg)
    got = store.get(slug)
    assert got == cfg


def test_get_missing_returns_none(tmp_path):
    store = SetStore(tmp_path / "sets")
    assert store.get("nope") is None


def test_list_sorted_by_name(tmp_path):
    store = SetStore(tmp_path / "sets")
    store.save(ConfigSet(name="Zulu"))
    store.save(ConfigSet(name="Alpha"))
    store.save(ConfigSet(name="Mike"))
    assert [c.name for c in store.list()] == ["Alpha", "Mike", "Zulu"]


def test_list_empty_when_dir_absent(tmp_path):
    assert SetStore(tmp_path / "never").list() == []


def test_delete_removes_file(tmp_path):
    store = SetStore(tmp_path / "sets")
    slug = store.save(ConfigSet(name="Temp"))
    store.delete(slug)
    assert store.get(slug) is None
    assert [c.name for c in store.list()] == []


def test_delete_missing_is_noop(tmp_path):
    SetStore(tmp_path / "sets").delete("ghost")  # must not raise


def test_save_write_is_atomic_no_temp_files_left(tmp_path):
    store = SetStore(tmp_path / "sets")
    store.save(ConfigSet(name="Atomic"))
    assert list((tmp_path / "sets").glob("*.tmp")) == []


def test_save_parent_dir_created(tmp_path):
    store = SetStore(tmp_path / "deep" / "nested" / "sets")
    store.save(ConfigSet(name="Deep"))
    assert (tmp_path / "deep" / "nested" / "sets" / "deep.json").is_file()


# --- reserved slug ---


@pytest.mark.parametrize("name", ["previous", "Previous", "· previous", "_previous"])
def test_save_rejects_reserved_slug(tmp_path, name):
    store = SetStore(tmp_path / "sets")
    with pytest.raises(ValueError):
        store.save(ConfigSet(name=name))


def test_save_rejects_empty_slug(tmp_path):
    store = SetStore(tmp_path / "sets")
    with pytest.raises(ValueError):
        store.save(ConfigSet(name="!!! ???"))


def test_save_previous_writes_reserved_file(tmp_path):
    store = SetStore(tmp_path / "sets")
    slug = store.save_previous(ConfigSet(name=PREVIOUS_NAME))
    assert slug == RESERVED_SLUG
    assert (tmp_path / "sets" / f"{RESERVED_SLUG}.json").is_file()
    assert store.get(RESERVED_SLUG).name == PREVIOUS_NAME


# ===========================================================================
# plan_apply — pure diff
# ===========================================================================


def test_empty_plan_when_reality_matches():
    world = make_world(
        lemonade=("loaded", "extra.d.gguf"),
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.d.gguf",
    )
    cfg = ConfigSet(
        name="steady",
        durable={"default_route_model": "extra.d.gguf", "activate_model_id": "cat-1"},
        ephemeral={
            "lemonade": {"state": "loaded"},
            "comfyui": {"state": "leave"},
            "hipfire": {"state": "running"},
        },
    )
    assert plan_apply(cfg, world) == []


def test_omitted_subsections_touch_nothing():
    world = make_world(lemonade=("loaded", "extra.x.gguf"), hipfire="parked")
    cfg = ConfigSet(name="empty-eph", ephemeral={})
    assert plan_apply(cfg, world) == []


def test_cheap_path_durable_unchanged_no_activate():
    world = make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg = ConfigSet(
        name="cheap",
        durable={"default_route_model": "extra.d.gguf", "activate_model_id": "cat-1"},
        ephemeral={"lemonade": {"state": "loaded"}},
    )
    plan = plan_apply(cfg, world)
    assert plan == [{"step": "load_lemonade", "model": "extra.d.gguf"}]
    assert not any(s["step"] == "activate" for s in plan)


def test_load_uses_world_default_route_when_no_durable():
    world = make_world(lemonade=("unloaded", None), default_route="extra.world.gguf")
    cfg = ConfigSet(name="noload-durable", ephemeral={"lemonade": {"state": "loaded"}})
    assert plan_apply(cfg, world) == [
        {"step": "load_lemonade", "model": "extra.world.gguf"}
    ]


def test_no_model_to_load_warn():
    world = make_world(lemonade=("unloaded", None), default_route=None)
    cfg = ConfigSet(name="nomodel", ephemeral={"lemonade": {"state": "loaded"}})
    assert plan_apply(cfg, world) == [{"step": "warn", "reason": "no-model-to-load"}]


def test_unload_uses_worlds_loaded_model():
    world = make_world(lemonade=("loaded", "extra.live.gguf"))
    cfg = ConfigSet(name="unload", ephemeral={"lemonade": {"state": "unloaded"}})
    assert plan_apply(cfg, world) == [
        {"step": "unload_lemonade", "model": "extra.live.gguf"}
    ]


def test_comfy_free_only_when_queue_zero():
    world = make_world(comfy=("idle", 0))
    cfg = ConfigSet(name="free", ephemeral={"comfyui": {"state": "free"}})
    assert plan_apply(cfg, world) == [{"step": "free_comfyui"}]


def test_comfy_busy_warns_not_frees():
    world = make_world(comfy=("busy", 3))
    cfg = ConfigSet(name="free", ephemeral={"comfyui": {"state": "free"}})
    assert plan_apply(cfg, world) == [
        {"step": "warn", "reason": "comfyui-busy-skipped"}
    ]


def test_comfy_unknown_queue_is_not_freed():
    world = make_world(comfy=("unknown", None))
    cfg = ConfigSet(name="free", ephemeral={"comfyui": {"state": "free"}})
    assert plan_apply(cfg, world) == [
        {"step": "warn", "reason": "comfyui-busy-skipped"}
    ]


def test_hipfire_park_when_running():
    world = make_world(hipfire="running")
    cfg = ConfigSet(name="park", ephemeral={"hipfire": {"state": "parked"}})
    assert plan_apply(cfg, world) == [{"step": "park_hipfire"}]


def test_hipfire_park_when_loading():
    world = make_world(hipfire="loading")
    cfg = ConfigSet(name="park", ephemeral={"hipfire": {"state": "parked"}})
    assert plan_apply(cfg, world) == [{"step": "park_hipfire"}]


def test_hipfire_resume_when_parked():
    world = make_world(hipfire="parked")
    cfg = ConfigSet(name="resume", ephemeral={"hipfire": {"state": "running"}})
    assert plan_apply(cfg, world) == [{"step": "resume_hipfire"}]


def test_durable_change_with_id_emits_activate():
    world = make_world(default_route="extra.old.gguf")
    cfg = ConfigSet(
        name="switch",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-9"},
    )
    assert plan_apply(cfg, world) == [{"step": "activate", "model_id": "cat-9"}]


def test_durable_revert_unavailable_warn():
    world = make_world(default_route="extra.old.gguf")
    cfg = ConfigSet(
        name="revert",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": None},
    )
    assert plan_apply(cfg, world) == [
        {"step": "warn", "reason": "durable-revert-unavailable"}
    ]


def test_policy_patch_always_emitted():
    world = make_world()
    overrides = {"lemonade": {"priority": 1, "pinned": False, "idle_ttl": 5}}
    cfg = ConfigSet(name="pol", policy_overrides=overrides)
    assert plan_apply(cfg, world) == [{"step": "policy_patch", "policies": overrides}]


def test_full_ordering_park_activate_load_policy():
    # evictions (free) -> park -> activate -> load -> policy
    world = make_world(
        lemonade=("unloaded", None),
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.old.gguf",
    )
    overrides = {"comfyui": {"priority": 9, "pinned": True, "idle_ttl": 0}}
    cfg = ConfigSet(
        name="big",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
        ephemeral={
            "lemonade": {"state": "loaded"},
            "comfyui": {"state": "free"},
            "hipfire": {"state": "parked"},
        },
        policy_overrides=overrides,
    )
    assert plan_apply(cfg, world) == [
        {"step": "free_comfyui"},
        {"step": "park_hipfire"},
        {"step": "activate", "model_id": "cat-7"},
        {"step": "load_lemonade", "model": "extra.new.gguf"},
        {"step": "policy_patch", "policies": overrides},
    ]


def test_full_ordering_unload_activate_resume():
    # evictions (unload, free) -> activate -> resume -> policy
    world = make_world(
        lemonade=("loaded", "extra.live.gguf"),
        comfy=("idle", 0),
        hipfire="parked",
        default_route="extra.old.gguf",
    )
    overrides = {"hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0}}
    cfg = ConfigSet(
        name="big2",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-3"},
        ephemeral={
            "lemonade": {"state": "unloaded"},
            "comfyui": {"state": "free"},
            "hipfire": {"state": "running"},
        },
        policy_overrides=overrides,
    )
    assert plan_apply(cfg, world) == [
        {"step": "unload_lemonade", "model": "extra.live.gguf"},
        {"step": "free_comfyui"},
        {"step": "activate", "model_id": "cat-3"},
        {"step": "resume_hipfire"},
        {"step": "policy_patch", "policies": overrides},
    ]


# ===========================================================================
# apply — imperative shell
# ===========================================================================


def test_apply_executes_steps_in_order(tmp_path):
    world = make_world(
        lemonade=("unloaded", None),
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.old.gguf",
    )
    overrides = {"lemonade": {"priority": 1, "pinned": False, "idle_ttl": 5}}
    cfg = ConfigSet(
        name="do-it",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
        ephemeral={
            "lemonade": {"state": "loaded"},
            "comfyui": {"state": "free"},
            "hipfire": {"state": "parked"},
        },
        policy_overrides=overrides,
    )
    report, clients = run_apply(cfg, world, tmp_path)

    assert report["failed"] is None
    assert report["error"] is None
    assert report["warnings"] == []
    assert [s["step"] for s in report["completed"]] == [
        "free_comfyui",
        "park_hipfire",
        "activate",
        "load_lemonade",
        "policy_patch",
    ]
    assert clients["comfy"].calls == ["free"]
    assert clients["hipfire"].calls == ["park"]
    assert clients["hostagent"].calls == ["cat-7"]
    assert clients["lemonade"].calls == [("load", "extra.new.gguf")]
    assert clients["policy_store"].calls == [overrides]


def test_apply_warn_recorded_not_executed(tmp_path):
    world = make_world(comfy=("busy", 2))
    cfg = ConfigSet(name="skip", ephemeral={"comfyui": {"state": "free"}})
    report, clients = run_apply(cfg, world, tmp_path)

    assert report["warnings"] == ["comfyui-busy-skipped"]
    assert report["completed"] == []
    assert report["failed"] is None
    assert clients["comfy"].calls == []


def test_apply_halts_at_failing_step_with_exact_report(tmp_path):
    world = make_world(
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.old.gguf",
    )
    cfg = ConfigSet(
        name="halt",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
        ephemeral={"comfyui": {"state": "free"}, "hipfire": {"state": "parked"}},
    )
    hipfire = RecHipfire()
    hipfire.fail = {"park": GuardError("default routes to hipfire")}
    report, clients = run_apply(cfg, world, tmp_path, hipfire=hipfire)

    assert report["completed"] == [{"step": "free_comfyui"}]
    assert report["failed"] == {"step": "park_hipfire"}
    assert report["error"] == "default routes to hipfire"
    assert report["warnings"] == []
    # comfy ran, hipfire.park attempted, activate NEVER reached
    assert clients["comfy"].calls == ["free"]
    assert clients["hipfire"].calls == ["park"]
    assert clients["hostagent"].calls == []


def test_apply_vetoes_before_any_mutation_when_hipfire_busy(tmp_path):
    """A plan that would park hipfire or flip the durable route is refused
    up front while a hipfire conversation is live: GuardError propagates
    (-> 409 at the route), nothing executes, no _previous snapshot is
    written (there is nothing to revert)."""
    world = make_world(
        comfy=("idle", 0), hipfire="running", default_route="extra.old.gguf"
    )
    cfg = ConfigSet(
        name="veto",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
        ephemeral={"comfyui": {"state": "free"}, "hipfire": {"state": "parked"}},
    )
    hipfire = RecHipfire()
    hipfire.fail = {
        "ensure_not_busy": GuardError("hipfire request in flight (queue_depth=1)")
    }
    store = SetStore(tmp_path / "sets")
    comfy = RecComfy()
    hostagent = RecHostAgent()

    with pytest.raises(GuardError, match="in flight"):
        run_apply(
            cfg, world, tmp_path,
            hipfire=hipfire, store=store, comfy=comfy, hostagent=hostagent,
        )

    assert hipfire.busy_checks == ["apply set 'veto'"]
    assert hipfire.calls == []
    assert comfy.calls == []
    assert hostagent.calls == []
    assert store.get("_previous") is None


def test_apply_force_skips_veto_and_threads_into_park(tmp_path):
    world = make_world(hipfire="running")
    cfg = ConfigSet(name="forced", ephemeral={"hipfire": {"state": "parked"}})
    hipfire = RecHipfire()
    hipfire.fail = {"ensure_not_busy": GuardError("busy")}

    report, _ = run_apply(cfg, world, tmp_path, hipfire=hipfire, force=True)

    assert report["failed"] is None
    assert hipfire.busy_checks == []  # veto skipped entirely
    assert hipfire.calls == ["park"]
    assert hipfire.park_forces == [True]


def test_apply_without_hipfire_steps_never_busy_checks(tmp_path):
    world = make_world(lemonade=("loaded", "extra.m.gguf"), hipfire="running")
    cfg = ConfigSet(name="lem-only", ephemeral={"lemonade": {"state": "unloaded"}})
    hipfire = RecHipfire()

    report, clients = run_apply(cfg, world, tmp_path, hipfire=hipfire)

    assert report["failed"] is None
    assert hipfire.busy_checks == []
    assert clients["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_apply_activate_step_rechecks_busy_and_halts(tmp_path):
    """The pre-veto passes, then a request lands on hipfire before the
    activate step runs (TOCTOU): the in-step recheck halts the apply with
    the ordinary failed-step report and the host agent is never called."""
    world = make_world(default_route="extra.old.gguf")
    cfg = ConfigSet(
        name="flip",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
    )
    hipfire = RecHipfire()
    hipfire.fail = {
        "ensure_not_busy": [
            None,
            GuardError("hipfire request in flight (queue_depth=1)"),
        ]
    }

    report, clients = run_apply(cfg, world, tmp_path, hipfire=hipfire)

    assert report["failed"] == {"step": "activate", "model_id": "cat-7"}
    assert "in flight" in report["error"]
    assert clients["hostagent"].calls == []
    assert hipfire.busy_checks == ["apply set 'flip'", "activate 'cat-7'"]


@pytest.mark.parametrize(
    "exc",
    [GuardError("g"), EngineError("e"), BusyError("b"), ValueError("v")],
)
def test_apply_halts_on_each_expected_exception(tmp_path, exc):
    world = make_world(default_route="extra.old.gguf")
    cfg = ConfigSet(
        name="boom",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-7"},
    )
    hostagent = RecHostAgent()
    hostagent.fail = exc
    report, _ = run_apply(cfg, world, tmp_path, hostagent=hostagent)

    assert report["failed"] == {"step": "activate", "model_id": "cat-7"}
    assert report["error"] == str(exc)


# ===========================================================================
# apply — pre-veto while the ODS host agent is mid-lifecycle-operation
# ===========================================================================


def test_apply_vetoed_while_host_agent_busy(tmp_path):
    """A plan whose steps touch the host-agent-guarded surface (here:
    load_lemonade) is refused up front, before any step executes, while the
    host agent owns the box — mirrors the hipfire pre-veto above."""
    world = make_world(lemonade=("unloaded", None), default_route="extra.m.gguf")
    cfg = ConfigSet(name="load-it", ephemeral={"lemonade": {"state": "loaded"}})

    with pytest.raises(BusyError, match="host agent is busy"):
        run_apply(cfg, world, tmp_path, hostagent=_BusyHostAgent())

    events = tail_events(tmp_path / "events.jsonl")
    assert any(
        e["kind"] == "apply-vetoed" and e["detail"].get("reason") == "host-agent-busy"
        for e in events
    )


def test_apply_force_bypasses_host_agent_veto(tmp_path):
    """force=True skips the host-agent probe entirely — the agent stays
    busy the whole time (proving it was never consulted), and the plan
    still completes."""
    world = make_world(lemonade=("unloaded", None), default_route="extra.m.gguf")
    cfg = ConfigSet(name="load-it", ephemeral={"lemonade": {"state": "loaded"}})

    report, clients = run_apply(
        cfg, world, tmp_path, hostagent=_BusyHostAgent(), force=True
    )

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("load", "extra.m.gguf")]


def test_apply_not_vetoed_when_host_agent_idle(tmp_path):
    world = make_world(lemonade=("unloaded", None), default_route="extra.m.gguf")
    cfg = ConfigSet(name="load-it", ephemeral={"lemonade": {"state": "loaded"}})

    report, clients = run_apply(cfg, world, tmp_path)  # default RecHostAgent is idle

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("load", "extra.m.gguf")]


def test_apply_not_vetoed_when_plan_has_no_guarded_steps(tmp_path):
    """comfyui-only plans never touch the host-agent-guarded step set; a busy
    agent must not block them, mirroring comfyui/free being unguarded at the
    control route."""
    world = make_world(comfy=("idle", 0))
    cfg = ConfigSet(name="free-it", ephemeral={"comfyui": {"state": "free"}})

    report, clients = run_apply(cfg, world, tmp_path, hostagent=_BusyHostAgent())

    assert report["failed"] is None
    assert clients["comfy"].calls == ["free"]


def test_apply_tolerates_hostagent_none(tmp_path):
    """hostagent defaults to None; a plan with guarded steps must not crash
    when no host-agent client is wired up (same tolerance heal_suppressor
    already gets)."""
    world = make_world(lemonade=("unloaded", None), default_route="extra.m.gguf")
    cfg = ConfigSet(name="load-it", ephemeral={"lemonade": {"state": "loaded"}})

    report, clients = run_apply(cfg, world, tmp_path, hostagent=None)

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("load", "extra.m.gguf")]


def test_apply_logs_start_and_end_ok(tmp_path):
    world = make_world()
    cfg = ConfigSet(name="Logged", ephemeral={})
    _, clients = run_apply(cfg, world, tmp_path)

    events = tail_events(clients["events_path"])
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "apply-start"
    assert events[0]["detail"] == {"name": "Logged"}
    assert kinds[-1] == "apply-end"
    assert events[-1]["detail"]["outcome"] == "ok"


def test_apply_logs_end_failed_on_halt(tmp_path):
    world = make_world(default_route="extra.old.gguf")
    cfg = ConfigSet(
        name="Failer",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "c"},
    )
    hostagent = RecHostAgent()
    hostagent.fail = EngineError("boom")
    _, clients = run_apply(cfg, world, tmp_path, hostagent=hostagent)

    events = tail_events(clients["events_path"])
    end = events[-1]
    assert end["kind"] == "apply-end"
    assert end["detail"]["outcome"] == "failed"
    assert end["detail"]["step"] == "activate"


def test_apply_logs_each_real_step_by_name(tmp_path):
    world = make_world(comfy=("idle", 0), hipfire="running")
    cfg = ConfigSet(
        name="Two",
        ephemeral={"comfyui": {"state": "free"}, "hipfire": {"state": "parked"}},
    )
    _, clients = run_apply(cfg, world, tmp_path)
    kinds = [e["kind"] for e in tail_events(clients["events_path"])]
    assert "free_comfyui" in kinds
    assert "park_hipfire" in kinds


# --- _previous snapshot ---


def test_previous_captures_pre_apply_reality(tmp_path):
    world = make_world(
        lemonade=("loaded", "extra.live.gguf"),
        comfy=("busy", 4),
        hipfire="running",
        default_route="extra.d.gguf",
    )
    # A no-op-ish set (leave everything) so apply changes nothing but still snapshots.
    cfg = ConfigSet(name="noop", ephemeral={"comfyui": {"state": "leave"}})
    _, clients = run_apply(cfg, world, tmp_path)

    prev = clients["store"].get(RESERVED_SLUG)
    assert prev.name == PREVIOUS_NAME
    assert prev.ephemeral.lemonade.state == "loaded"
    assert prev.ephemeral.comfyui.state == "leave"
    assert prev.ephemeral.hipfire.state == "running"
    assert prev.durable.default_route_model == "extra.d.gguf"
    assert prev.durable.activate_model_id is None
    assert prev.notes


def test_previous_durable_none_when_no_default_route(tmp_path):
    world = make_world(
        lemonade=("unloaded", None), hipfire="parked", default_route=None
    )
    cfg = ConfigSet(name="noop", ephemeral={})
    _, clients = run_apply(cfg, world, tmp_path)

    prev = clients["store"].get(RESERVED_SLUG)
    assert prev.durable is None
    assert prev.ephemeral.lemonade.state == "unloaded"
    assert prev.ephemeral.hipfire.state == "parked"


def test_previous_hipfire_loading_snapshots_as_running(tmp_path):
    world = make_world(hipfire="loading")
    cfg = ConfigSet(name="noop", ephemeral={})
    _, clients = run_apply(cfg, world, tmp_path)
    assert clients["store"].get(RESERVED_SLUG).ephemeral.hipfire.state == "running"


def test_previous_captured_before_any_step_runs(tmp_path):
    # apply halts on the very first step, yet _previous must already be on disk.
    world = make_world(
        lemonade=("loaded", "extra.live.gguf"), default_route="extra.d.gguf"
    )
    cfg = ConfigSet(name="halt-first", ephemeral={"lemonade": {"state": "unloaded"}})
    lemonade = RecLemonade()
    lemonade.fail = {"unload": EngineError("nope")}
    report, clients = run_apply(cfg, world, tmp_path, lemonade=lemonade)

    assert report["failed"] == {"step": "unload_lemonade", "model": "extra.live.gguf"}
    prev = clients["store"].get(RESERVED_SLUG)
    assert prev is not None
    assert prev.ephemeral.lemonade.state == "loaded"


# --- serialization under the module lock ---


def test_apply_is_serialized_second_waits_for_first(tmp_path):
    """A slow apply holds the module lock; a second apply must block until it
    releases. Proven by lock state + thread liveness, not by sleeps."""
    entered = threading.Event()
    release = threading.Event()

    class BlockingLemonade:
        def __init__(self):
            self.calls = []

        def load(self, model):
            self.calls.append(("load", model))
            entered.set()
            release.wait(timeout=5)

        def unload(self, model):  # pragma: no cover - unused here
            self.calls.append(("unload", model))

    world = make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg1 = ConfigSet(name="slow", ephemeral={"lemonade": {"state": "loaded"}})
    cfg2 = ConfigSet(name="fast", ephemeral={"lemonade": {"state": "loaded"}})

    store = SetStore(tmp_path / "sets")
    events_path = tmp_path / "events.jsonl"
    slow_lem = BlockingLemonade()
    fast_lem = RecLemonade()

    def base_clients(lem):
        return {
            "lemonade": lem,
            "comfy": RecComfy(),
            "hipfire": RecHipfire(),
            "hostagent": RecHostAgent(),
            "policy_store": RecPolicyStore(),
            "store": store,
            "events_path": events_path,
        }

    results = {}

    t1 = threading.Thread(
        target=lambda: results.__setitem__(
            "t1", apply(cfg1, world=world, **base_clients(slow_lem))
        )
    )
    t1.start()
    assert entered.wait(timeout=5), "first apply never reached its blocking step"

    import app.sets as sets_mod

    assert sets_mod._apply_lock.locked()

    t2 = threading.Thread(
        target=lambda: results.__setitem__(
            "t2", apply(cfg2, world=world, **base_clients(fast_lem))
        )
    )
    t2.start()

    # While t1 holds the lock, t2 cannot proceed: it stays alive and never calls.
    t2.join(timeout=0.5)
    assert t2.is_alive()
    assert fast_lem.calls == []

    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive()
    assert not t2.is_alive()
    assert fast_lem.calls == [("load", "extra.d.gguf")]
    assert results["t1"]["failed"] is None
    assert results["t2"]["failed"] is None


# ===========================================================================
# I4 — park AFTER activate when the durable change moves the route off hipfire
# ===========================================================================


def test_park_after_activate_when_route_moves_off_hipfire():
    """When the current default route targets hipfire and the set both parks
    hipfire and activates a different model, park is emitted AFTER activate so
    the GPU isn't yanked out from under the still-default hipfire model."""
    world = make_world(
        lemonade=("unloaded", None),
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.hip-model",
    )
    world["tenants"]["hipfire"]["model"] = "extra.hip-model"  # route IS hipfire
    cfg = ConfigSet(
        name="off-hipfire",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-9"},
        ephemeral={
            "lemonade": {"state": "loaded"},
            "comfyui": {"state": "free"},
            "hipfire": {"state": "parked"},
        },
    )
    assert plan_apply(cfg, world) == [
        {"step": "free_comfyui"},
        {"step": "activate", "model_id": "cat-9"},
        {"step": "park_hipfire"},
        {"step": "load_lemonade", "model": "extra.new.gguf"},
    ]


def test_park_before_activate_when_route_not_on_hipfire():
    """Complement: route NOT on hipfire (hipfire.model != default_route) keeps
    the normal park-BEFORE-activate order, even with both steps present."""
    world = make_world(
        lemonade=("unloaded", None),
        comfy=("idle", 0),
        hipfire="running",
        default_route="extra.old.gguf",
    )
    world["tenants"]["hipfire"]["model"] = "extra.hip-model"  # != default_route
    cfg = ConfigSet(
        name="normal",
        durable={"default_route_model": "extra.new.gguf", "activate_model_id": "cat-9"},
        ephemeral={"hipfire": {"state": "parked"}},
    )
    assert plan_apply(cfg, world) == [
        {"step": "park_hipfire"},
        {"step": "activate", "model_id": "cat-9"},
    ]


# ===========================================================================
# I5b — policy_patch merges field-partial per-tenant overrides
# ===========================================================================


def test_policy_patch_merges_partial_override(tmp_path):
    """A partial override ({"comfyui": {"priority": 90}}) merges onto the
    tenant's current stored values before put()."""
    world = make_world()
    cfg = ConfigSet(name="pol", policy_overrides={"comfyui": {"priority": 90}})
    report, clients = run_apply(cfg, world, tmp_path)

    assert report["failed"] is None
    assert clients["policy_store"].calls == [
        {"comfyui": {"priority": 90, "pinned": False, "idle_ttl": 300}}
    ]


def test_policy_patch_full_record_override_still_works(tmp_path):
    """A full 3-field override survives the merge unchanged."""
    world = make_world()
    full = {"lemonade": {"priority": 7, "pinned": True, "idle_ttl": 42}}
    cfg = ConfigSet(name="pol", policy_overrides=full)
    report, clients = run_apply(cfg, world, tmp_path)

    assert report["failed"] is None
    assert clients["policy_store"].calls == [full]


def test_policy_patch_partial_merge_persists_via_real_store(tmp_path):
    """End-to-end with the real PolicyStore: a partial override lands with the
    other two fields intact and leaves untouched tenants alone."""
    world = make_world()
    store = PolicyStore(tmp_path / "policy.json")
    cfg = ConfigSet(name="pol", policy_overrides={"comfyui": {"priority": 90}})
    run_apply(cfg, world, tmp_path, policy_store=store)

    assert store.get()["comfyui"] == {"priority": 90, "pinned": False, "idle_ttl": 300}
    assert store.get()["lemonade"] == DEFAULT_POLICIES["lemonade"]


def test_policy_patch_unknown_tenant_still_fails(tmp_path):
    """An unknown tenant merges onto {} and is rejected by put's validation."""
    world = make_world()
    store = PolicyStore(tmp_path / "policy.json")
    cfg = ConfigSet(name="pol", policy_overrides={"nosuch": {"priority": 1}})
    report, _ = run_apply(cfg, world, tmp_path, policy_store=store)

    assert report["failed"] == {"step": "policy_patch", "policies": {"nosuch": {"priority": 1}}}
    assert "nosuch" in report["error"]


# ===========================================================================
# I1 — apply_in_progress() peeks the module lock without acquiring it
# ===========================================================================


def test_apply_in_progress_reflects_lock_state():
    import app.sets as sets_mod

    assert apply_in_progress() is False
    with sets_mod._apply_lock:
        assert apply_in_progress() is True
    assert apply_in_progress() is False


# ===========================================================================
# C2 — set-apply unload arms / load clears the heal suppressor
# ===========================================================================


def test_apply_unload_step_engages_heal_suppressor(tmp_path):
    world = make_world(lemonade=("loaded", "extra.live.gguf"))
    cfg = ConfigSet(name="unload", ephemeral={"lemonade": {"state": "unloaded"}})
    suppressor = HealSuppressor(window_s=600, clock=lambda: 0.0)
    run_apply(cfg, world, tmp_path, heal_suppressor=suppressor)

    assert suppressor.suppressed() is True


def test_apply_load_step_clears_heal_suppressor(tmp_path):
    world = make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg = ConfigSet(name="load", ephemeral={"lemonade": {"state": "loaded"}})
    suppressor = HealSuppressor(window_s=600, clock=lambda: 0.0)
    suppressor.note_deck_unload()
    assert suppressor.suppressed() is True

    run_apply(cfg, world, tmp_path, heal_suppressor=suppressor)

    assert suppressor.suppressed() is False


def test_apply_tolerates_no_heal_suppressor(tmp_path):
    """heal_suppressor defaults to None (unit tests without the arbiter) and
    apply must run the unload step without error."""
    world = make_world(lemonade=("loaded", "extra.live.gguf"))
    cfg = ConfigSet(name="unload", ephemeral={"lemonade": {"state": "unloaded"}})
    report, clients = run_apply(cfg, world, tmp_path)  # no heal_suppressor

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("unload", "extra.live.gguf")]


# ===========================================================================
# last_used observation on the set-apply load step (I5)
# ===========================================================================


class _RecCatalog:
    def __init__(self):
        self.noted = []

    def note_used_gguf(self, filename):
        self.noted.append(filename)


def test_apply_load_step_notes_last_used(tmp_path):
    """A set-apply load is a real observation of the model being used; without
    it the LRU eviction order treats a model the operator loads via a set as
    never used."""
    world = make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg = ConfigSet(name="chat", ephemeral={"lemonade": {"state": "loaded"}})
    catalog = _RecCatalog()

    report, clients = run_apply(cfg, world, tmp_path, catalog=catalog)

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("load", "extra.d.gguf")]
    assert catalog.noted == ["d.gguf"]          # bare name, "extra." stripped


def test_apply_load_step_without_catalog_still_works(tmp_path):
    """catalog is optional — unit tests (and any caller without the deck) must
    keep working."""
    world = make_world(lemonade=("unloaded", None), default_route="extra.d.gguf")
    cfg = ConfigSet(name="chat", ephemeral={"lemonade": {"state": "loaded"}})

    report, clients = run_apply(cfg, world, tmp_path)

    assert report["failed"] is None
    assert clients["lemonade"].calls == [("load", "extra.d.gguf")]
