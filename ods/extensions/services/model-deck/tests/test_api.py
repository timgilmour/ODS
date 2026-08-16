"""Tests for the Model Deck HTTP API — app.routers.* (no auth: the admin
gate was deliberately removed 2026-07-22; every endpoint is open).

TestClient against app.main.create_app(), with individual app.state.deck
entries swapped for recording fakes AFTER construction — no env vars beyond
MODEL_DECK_NO_WATCHER=1 (so no background thread starts). No real sockets
are ever touched.

Fakes track only *mutating* calls in `.calls` (load/unload/free/park/resume/
activate/policy put) — read-only calls a World snapshot legitimately makes
(status/queue_len/route_table) are NOT mutations and are deliberately left
untracked, so "assert fakes record zero calls" (preview) means exactly what
it says: no action was executed, even though preview must still read live
state to diff against.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.engine_kinds import KNOWN_KINDS
from app.engines import EngineError, GuardError
from app.intent import IntentStore
from app.main import create_app
from app.policy import PolicyStore
from app.sets import PREVIOUS_NAME, RESERVED_SLUG, ConfigSet, SetStore

# E1 Task 8: Ephemeral is now {resources: {...}} keyed by resource, not the
# old fixed lemonade/comfyui/hipfire sub-sections. Mirrors tests/test_sets.py's
# `_eph` — kept as a local copy rather than imported, same call test_sets.py's
# own docstring makes for `_BusyHostAgent`: a 3-kwarg translator is more
# awkward to share across test modules than to just mirror.
_STATE_TO_DESIRED = {
    "loaded": "loaded", "unloaded": "unloaded",        # lemonade
    "free": "freed", "leave": None,                     # comfyui ("leave" -> omit)
    "running": "loaded", "parked": "parked",             # hipfire
}


def _eph(*, lemonade=None, comfyui=None, hipfire=None):
    resources = {}
    for resource, spec in (("lemonade", lemonade), ("comfyui", comfyui), ("hipfire", hipfire)):
        if spec is None:
            continue
        spec = dict(spec)
        state = spec.pop("state")
        spec.pop("reserve_gb", None)
        desired = _STATE_TO_DESIRED[state]
        if desired is None:
            continue
        resources[resource] = {"desired": desired, **spec}
    return {"resources": resources}


# ===========================================================================
# Fakes
# ===========================================================================


class FakeLemonade:
    def __init__(self, loaded=None, raise_on_load=None, raise_on_unload=None):
        self.calls = []  # mutating only: ("load", model) / ("unload", model)
        self.fail = None
        # Fails load() only (``fail`` fails both) — lets a test prove that a
        # load which errored records no intent while an unload still works.
        self.load_raises = None
        self._loaded = loaded
        # Per-method injection, distinct from `fail`/`load_raises` above:
        # lets a single test fail exactly one of load()/unload() without
        # touching the other's fixture wiring.
        self.raise_on_load = raise_on_load
        self.raise_on_unload = raise_on_unload

    def load_in_flight(self):
        return False

    def status(self):
        return {"loaded": self._loaded}

    def activity(self):
        return None

    def load(self, model):
        self.calls.append(("load", model))
        if self.load_raises:
            raise self.load_raises
        if self.raise_on_load:
            raise self.raise_on_load
        if self.fail:
            raise self.fail
        self._loaded = model

    def unload(self, model):
        self.calls.append(("unload", model))
        if self.raise_on_unload:
            raise self.raise_on_unload
        if self.fail:
            raise self.fail
        self._loaded = None


class FakeComfy:
    def __init__(self, queue=0):
        self.calls = []  # mutating only: "free"
        self.fail = None
        self._queue = queue

    def queue_len(self):
        return self._queue

    def free(self):
        self.calls.append("free")
        if self.fail:
            raise self.fail


class FakeHipfire:
    def __init__(self, state="running", busy_error=None):
        self.calls = []  # mutating only: "park" / "resume"
        self.fail = None
        self.busy_error = busy_error  # raised by ensure_not_busy / unforced park
        self.busy_checks = []
        self.park_forces = []
        self._state = state

    def status(self):
        return self._state

    def stats(self):
        return {"queue_depth": 0, "requests_served": 0}

    def ensure_not_busy(self, action):
        self.busy_checks.append(action)
        if self.busy_error:
            raise self.busy_error

    def park(self, force=False):
        self.calls.append("park")
        self.park_forces.append(force)
        if self.fail:
            raise self.fail
        if not force and self.busy_error:
            raise self.busy_error
        self._state = "parked"

    def resume(self):
        self.calls.append("resume")
        if self.fail:
            raise self.fail
        self._state = "running"


class FakeLiteLLM:
    def __init__(self, default=None, hipfire=None):
        self._routes = {}
        if default is not None:
            self._routes["default"] = default
        if hipfire is not None:
            self._routes["hipfire"] = hipfire
        # Keyed by RESOLVED model id (never the route alias), for model_info().
        self.max_input_tokens = {}

    def route_table(self):
        return dict(self._routes)

    def model_info(self):
        """Mirrors the real /model/info shape (see app/engines/litellm.py):
        each entry's `model_name` is a route ALIAS ("default", "hipfire",
        ...), and `litellm_params.model` is the "openai/"-prefixed
        RESOLVED model id that alias actually points at — the two are
        deliberately different so a join keyed by alias fails loudly
        instead of an accidental match hiding the bug."""
        entries = []
        for alias, resolved in self._routes.items():
            entry = {
                "model_name": alias,
                "litellm_params": {"model": f"openai/{resolved}"},
            }
            tokens = self.max_input_tokens.get(resolved)
            if tokens is not None:
                entry["model_info"] = {"max_input_tokens": tokens}
            entries.append(entry)
        return entries


class FakeHostAgent:
    def __init__(self):
        self.calls = []  # mutating only: ("activate", model_id)
        self.fail = None

    def activate(self, model_id):
        self.calls.append(("activate", model_id))
        if self.fail:
            raise self.fail
        return {"activated": model_id}

    def lifecycle(self):
        # Idle by default — matches a box with no in-flight host-agent op.
        return {"active": False, "operation": None, "target": None}


class _BusyHostAgent:
    """Mirrors test_arbiter.py's I2 busy-lifecycle fixture (``_BusyHostAgent``
    there). Kept as a local copy here — importing a 3-line fake across test
    modules would be more awkward than just mirroring it."""

    def lifecycle(self):
        return {"active": True, "operation": "model_activation", "target": "qwen3-30b"}


class FakeRegistry:
    def __init__(self, footprints=None, models=None):
        self._footprints = footprints or {}
        self._models = models or []

    def footprint(self, key):
        if key not in self._footprints:
            raise FileNotFoundError(key)
        return self._footprints[key]

    def scan(self):
        return list(self._models)


class FakeReadGpus:
    """Recording stand-in for app.gpu.read_gpus; ignores the roots passed
    in and returns a fixed two-GPU list, but records every call."""

    def __init__(self, gpus=None):
        self.calls = []
        self._gpus = (
            gpus
            if gpus is not None
            else [
                {"index": 0, "vram_total": 40_000_000_000, "vram_used": 5_000_000_000, "pids": {}},
                {"index": 1, "vram_total": 24_000_000_000, "vram_used": 1_000_000_000, "pids": {}},
            ]
        )

    def __call__(self, drm_root, kfd_root):
        self.calls.append((drm_root, kfd_root))
        return self._gpus


# ===========================================================================
# App builder
# ===========================================================================


# The coexistence-triple declaration (E1 Task 3): a fresh data dir has no
# presence proof, so app.node_store.seed_engines_if_missing stamps `engines:
# []` — World.snapshot would then see NO declared resources at all, and
# every downstream consumer that still assumes the pre-E1 fixed triple
# (app.observe.observe_local, app.storage.unit_in_use, ...) would either
# emit nothing or KeyError. This file's tests are about THOSE consumers
# behaving as if lemonade/comfyui/hipfire are declared and running the
# fakes below — the fixture's job, not each individual test's — so
# make_app seeds this declaration unconditionally, post-construction (same
# posture as policy_store/set_store/intent_store being swapped below).
_ENGINES = [
    {"resource": "hipfire", "kind": "hipfire",
     "connection": {"container": "ods-hipfire"}, "gpu_index": 0,
     "policy_defaults": {"priority": 100, "pinned": True, "idle_ttl": 0}},
    {"resource": "lemonade", "kind": "lemonade",
     "connection": {"url": "http://llama-server:8080",
                    "metrics_url": "http://llama-server:8001/metrics",
                    "container": "ods-llama-server"},
     "gpu_index": 1,
     "policy_defaults": {"priority": 50, "pinned": False, "idle_ttl": 900}},
    {"resource": "comfyui", "kind": "comfyui",
     "connection": {"url": "http://comfyui:8188"}, "gpu_index": 1,
     "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
]

# Declared policy defaults for test assertions (replaces deleted _TEST_POLICY_DEFAULTS)
_TEST_POLICY_DEFAULTS = {
    "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
    "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
    "comfyui": {"priority": 40, "pinned": False, "idle_ttl": 300},
}


class FakeLocalClients:
    """Stand-in for app.local_clients.LocalClients: resolves each resource
    to the deck's CURRENT `lemonade`/`comfy`/`hipfire` fake, read LIVE off
    `deck` on every call — not captured at construction time — so a test
    that reassigns e.g. ``deck["lemonade"] = FakeLemonade(...)`` mid-test
    (many below do) is still what any later World.snapshot call observes."""

    _DECK_KEY = {"lemonade": "lemonade", "comfyui": "comfy", "hipfire": "hipfire"}

    def __init__(self, deck: dict) -> None:
        self._deck = deck

    def client_for(self, resource: str):
        return self._deck.get(self._DECK_KEY.get(resource, resource))

    def retire_absent(self, keep_resources) -> None:
        pass  # nothing built to retire — every lookup is live, see client_for

    def fake(self, resource: str) -> "FakeLemonade":
        """T10: wire (and return) a fresh FakeLemonade for `resource`,
        stored wherever `client_for` resolves it (the same deck-key
        convention `_declare_local`'s docstring documents for a resource
        outside the legacy trio's `_DECK_KEY` map). FakeLemonade specifically
        — not a per-kind dispatch — because every use of this accessor to
        date only needs the shared `.calls` list a bookkeeping-only
        assertion (e.g. forget) reads; nothing here depends on lemonade's
        load/unload semantics."""
        instance = FakeLemonade()
        self._deck[resource] = instance
        return instance


def make_app(tmp_path, monkeypatch):
    """create_app() with MODEL_DECK_NO_WATCHER=1 and every engine client /
    read_gpus swapped for a fake; policy_store/set_store point at tmp_path
    (real, not faked — their own test files already cover their behavior).
    No auth setup: the admin gate was deliberately removed 2026-07-22."""
    monkeypatch.setenv("MODEL_DECK_NO_WATCHER", "1")

    app = create_app()
    deck = app.state.deck

    def _get_declared_policy_defaults():
        """Provide declared policy defaults LIVE off node_store's local
        `engines[]` (mirrors app.main._build_deck's own same-named helper)
        — not a frozen snapshot of the module-level `_ENGINES` fixture, so
        a test that edits the declaration after construction (`_declare_local`,
        or — T10 — the engines CRUD routes themselves) sees policy defaults
        that track the edit, the same as production. `_ENGINES` is still
        what `make_app` SEEDS node_store with below, so every test that never
        calls `_declare_local` sees byte-identical behavior to before this
        fixture fix (disclosed, T10: the static version predates any test
        combining `_declare_local` with a policy assertion, so this gap was
        never exercised until T10's forget-deletes-the-policy-row test)."""
        local = deck["node_store"].get("local")
        engines = local.get("engines", []) if local is not None else []
        declared = {}
        for engine in engines:
            if "resource" in engine and "policy_defaults" in engine:
                declared[engine["resource"]] = engine["policy_defaults"]
        return declared

    deck.update(
        {
            "lemonade": FakeLemonade(),
            "comfy": FakeComfy(),
            "hipfire": FakeHipfire(),
            "hostagent": FakeHostAgent(),
            "litellm": FakeLiteLLM(default="extra.model.gguf"),
            "registry": FakeRegistry(),
            "read_gpus": FakeReadGpus(),
            "policy_store": PolicyStore(tmp_path / "policy.json",
                                        declared_defaults=_get_declared_policy_defaults),
            "set_store": SetStore(tmp_path / "sets"),
            # Real store (like policy_store), pointed at tmp_path: every
            # deliberate action now writes intent, and the default deck's
            # store points at the container's /data, which does not exist
            # under test.
            "intent_store": IntentStore(tmp_path / "intent.json"),
            "events_path": tmp_path / "events.jsonl",
        }
    )
    # E1 Task 3: declare the coexistence triple + wire World.snapshot's
    # clients onto the fakes above (see _ENGINES/FakeLocalClients docstrings).
    deck["node_store"].update("local", {"engines": _ENGINES})
    deck["local_clients"] = FakeLocalClients(deck)
    return app, deck


class FakeNodeClients:
    """Stand-in for app.node_clients.NodeClients: a plain dict lookup, so a
    test wires exactly the fake client(s) it means to and nothing rebinds
    underneath it."""

    def __init__(self, clients=None):
        self._clients = dict(clients or {})

    def set(self, node_id, client):
        self._clients[node_id] = client

    def client_for(self, node_id):
        return self._clients.get(node_id)

    def retire_absent(self, keep_ids) -> None:
        """No-op: this fake is a plain dict, not the real NodeClients'
        lazy build/retire cache, so there is nothing here to go stale or
        need closing. NodeObservers.snapshot() calls this unconditionally
        (see app/node_clients.py's retire_absent), so the fake must answer
        to the same interface."""


def wire_swap_node(deck, node_id, client, label=None):
    """Registry row (control:"swap", prereqs present) + a fake client bound
    for it. Replaces the deck's client map with a FakeNodeClients (once) and
    rebuilds node_observers over it. Labels ≠ ids on purpose
    ([[defaults-that-hide-bugs]]).

    NOTE: each node's SparkObserver caches for SPARK_OBSERVE_TTL_S; a test
    that mutates the fake's status mid-test must call
    deck["node_observers"].invalidate(node_id) before re-reading state."""
    from app.node_clients import NodeObservers

    store = deck["node_store"]
    if store.get(node_id) is None:
        store.add({"id": node_id, "label": label or f"{node_id.title()} Box",
                   "agent_kind": "node-agent",
                   "address": f"http://{node_id}:7720",
                   "serving_address": f"http://{node_id}:8000",
                   "control": "swap"}, credential=f"key-{node_id}")
    if not isinstance(deck["node_clients"], FakeNodeClients):
        deck["node_clients"] = FakeNodeClients()
        deck["node_observers"] = NodeObservers(store, deck["node_clients"])
    deck["node_clients"].set(node_id, client)


# E1 Task 7: a second lemonade-kind resource's fixture entry, module-level so
# later tasks (10+) can use it verbatim without redefining the shape. Mirrors
# tests/test_engine_kinds.py's `_entry()` — resource deliberately NOT an
# engine name, GPU not 0/1 ([[defaults-that-hide-bugs]]) — but pinned to GPU
# 2 specifically (not that file's 3) so a fixture pairing it with a sibling
# "gguf-b" entry lands the two on DIFFERENT GPUs, same as
# test_engine_kinds.py's own two-resource fixture.
_GGUF_A_ENTRY = {
    "resource": "gguf-a", "kind": "lemonade",
    "connection": {"url": "http://gguf-a:8080",
                   "metrics_url": "http://gguf-a:8001/metrics",
                   "container": "ods-gguf-a"},
    "gpu_index": 2,
    "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60},
}


def _declare_local(deck, engines):
    """Replace the local node's engines[] declaration (E1 Task 7+): write
    `engines` onto node_store and rely on every consumer that reads it fresh
    per call to pick it up. `deck["local_clients"]` (FakeLocalClients above)
    reads live off `deck` on every `client_for` call — nothing there to
    rebuild — but this still goes through node_store, not a deck dict write,
    because that is what the real dispatch path
    (app.routers.control._declared_kind / LocalClients.client_for) reads
    from. A test that needs a client for a resource this declares (anything
    other than the seeded lemonade/comfyui/hipfire trio) must still wire one
    itself, e.g. ``deck["gguf-a"] = FakeLemonade()`` — see FakeLocalClients'
    docstring for why that deck-key convention resolves any resource name,
    not just the legacy three."""
    deck["node_store"].update("local", {"engines": engines})


class FakeSpark:
    """The fake swap-node client wire_swap_node binds, for both the node
    registry/adopt suite here and tests/test_serving_api.py's per-node
    serving suite. Lived in tests/test_spark_api.py until the /api/spark/*
    alias (and that file with it) was removed."""

    def __init__(self):
        self.calls = []  # mutating only: ("swap", profile, force)
        self.status_calls = 0
        self.fail = None
        self.settings_sent = None  # (profile, document), last put_settings call
        # Reload re-fetches the profile's compose before shipping (final
        # branch review: a stale service name in the identity map would
        # introduce an imageless service AFTER teardown killed everything),
        # so every reload test needs real compose text behind get_compose.
        self.compose = {}          # {profile: text}; default = the fixture
        self._status = {
            "profiles": [
                {"name": "laguna", "engine": "vllm", "health_url": None,
                 "container": None},
                {"name": "mm27b", "engine": "vllm", "health_url": None,
                 "container": None},
            ],
            "swap_status": None,
            "serving": {"model": "aeon", "endpoint_ok": True,
                        "container_status": None},
        }

    def status(self):
        self.status_calls += 1
        return self._status

    def swap(self, profile, force=False):
        self.calls.append(("swap", profile, force))
        if self.fail:
            raise self.fail
        return {"id": "u1", "profile": profile}

    def put_settings(self, profile, document):
        self.settings_sent = (profile, document)

    def get_compose(self, profile):
        return self.compose.get(profile, HERETIC_COMPOSE)


# ===========================================================================


def test_gets_are_open_without_auth(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    assert client.get("/api/state").status_code == 200
    assert client.get("/api/events").status_code == 200
    assert client.get("/api/sets").status_code == 200
    assert client.get("/api/policy").status_code == 200


# ===========================================================================
# /api/state, /api/events
# ===========================================================================


def test_api_state_shape(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["registry"] = FakeRegistry(models=[{"file": "m.gguf", "size": 1, "footprint": 2}])
    resp = TestClient(app).get("/api/state")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"node", "world", "policy", "models", "lifecycle",
                                "provenance", "nodes"}
    assert body["policy"] == _TEST_POLICY_DEFAULTS
    assert body["models"] == [{"file": "m.gguf", "size": 1, "footprint": 2}]
    assert body["world"]["default_route"] == "extra.model.gguf"
    assert body["world"]["tenants"]["hipfire"]["state"] == "running"
    assert body["world"]["placement"]["hipfire"] == 0


def test_state_provenance_block_carries_an_updates_count(tmp_path, monkeypatch):
    app, _deck = make_app(tmp_path, monkeypatch)
    block = TestClient(app).get("/api/state").json()["provenance"]
    assert set(block) == {"drift", "gaps", "updates"}
    assert isinstance(block["updates"], int)


def test_api_state_calls_read_gpus_fresh_each_request(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    client.get("/api/state")
    client.get("/api/state")

    assert len(deck["read_gpus"].calls) == 2


def test_api_events_returns_tail(tmp_path, monkeypatch):
    from app.events import log_event

    app, deck = make_app(tmp_path, monkeypatch)
    log_event(deck["events_path"], "test-event", {"a": 1})

    resp = TestClient(app).get("/api/events?n=100")

    assert resp.status_code == 200
    events = resp.json()["events"]
    assert events[-1]["kind"] == "test-event"
    assert events[-1]["detail"] == {"a": 1}


# ===========================================================================
# Control: lemonade / comfyui / hipfire
# ===========================================================================


def test_lemonade_load(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.new.gguf")]


def test_lemonade_load_bare_name_is_extra_prefixed(tmp_path, monkeypatch):
    """The Deck select carries bare GGUF filenames; the load route prefixes
    them with 'extra.' before handing them to Lemonade (I2)."""
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "Qwen3.5-27B.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.Qwen3.5-27B.gguf")]


def test_lemonade_load_clears_heal_suppressor(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["heal_suppressor"].note_deck_unload()
    assert deck["heal_suppressor"].suppressed() is True
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert deck["heal_suppressor"].suppressed() is False


def test_lemonade_load_suppresses_healing_while_load_is_in_flight(tmp_path, monkeypatch):
    # While the blocking engine load runs, lemonade still reports "unloaded"
    # and the target GPU's VRAM is already climbing, so an un-suppressed
    # watcher tick infers a pending default-route load and stomps the manual
    # one. The route must arm suppression BEFORE the engine call; the
    # clear-after-success behavior stays.
    app, deck = make_app(tmp_path, monkeypatch)
    fake = deck["lemonade"]
    seen = {}
    original_load = fake.load

    def observing_load(model):
        seen["suppressed_during_load"] = deck["heal_suppressor"].suppressed()
        return original_load(model)

    fake.load = observing_load
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert seen["suppressed_during_load"] is True
    assert deck["heal_suppressor"].suppressed() is False


def test_lemonade_unload_engages_heal_suppressor(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    assert deck["heal_suppressor"].suppressed() is False
    resp = TestClient(app).post(
        "/api/tenants/lemonade/unload", json={"model": "extra.m.gguf"}
    )
    assert resp.status_code == 200
    assert deck["heal_suppressor"].suppressed() is True


def test_lemonade_unload_explicit_model(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post(
        "/api/tenants/lemonade/unload", json={"model": "extra.m.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_omitted_model_uses_currently_loaded(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={})
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_lemonade_unload_no_model_loaded_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={})
    assert resp.status_code == 409
    assert deck["lemonade"].calls == []


def test_comfy_free_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = GuardError("ComfyUI queue is not empty")
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "ComfyUI queue is not empty"


def test_comfy_free_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"].fail = EngineError("comfyui unreachable")
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "comfyui unreachable"


def test_hipfire_park_guard_error_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = GuardError("litellm default route targets hipfire")
    resp = TestClient(app).post("/api/tenants/hipfire/park")
    assert resp.status_code == 409


def test_hipfire_resume_success(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/tenants/hipfire/resume")
    assert resp.status_code == 200
    assert deck["hipfire"].calls == ["resume"]


def test_hipfire_resume_engine_error_502(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"].fail = EngineError("dockerctl unreachable")
    resp = TestClient(app).post("/api/tenants/hipfire/resume")
    assert resp.status_code == 502


# ===========================================================================
# Control routes refuse while the host agent is mid-lifecycle-operation
# ===========================================================================


def test_lemonade_load_409_while_host_agent_busy(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 409
    assert "host agent is busy" in resp.json()["detail"]
    assert deck["lemonade"].calls == []


def test_lemonade_load_force_bypasses_host_agent_guard(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post(
        "/api/tenants/lemonade/load?force=true", json={"model": "extra.new.gguf"}
    )
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("load", "extra.new.gguf")]


def test_lemonade_unload_409_while_host_agent_busy(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/lemonade/unload", json={})
    assert resp.status_code == 409
    assert "host agent is busy" in resp.json()["detail"]
    assert deck["lemonade"].calls == []


def test_lemonade_unload_force_bypasses_host_agent_guard(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/lemonade/unload?force=true", json={})
    assert resp.status_code == 200
    assert deck["lemonade"].calls == [("unload", "extra.m.gguf")]


def test_hipfire_park_409_while_host_agent_busy(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/hipfire/park")
    assert resp.status_code == 409
    assert "host agent is busy" in resp.json()["detail"]
    assert deck["hipfire"].calls == []


def test_hipfire_park_force_bypasses_host_agent_guard(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/hipfire/park?force=true")
    assert resp.status_code == 200
    assert deck["hipfire"].calls == ["park"]


def test_hipfire_resume_409_while_host_agent_busy(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/hipfire/resume")
    assert resp.status_code == 409
    assert "host agent is busy" in resp.json()["detail"]
    assert deck["hipfire"].calls == []


def test_hipfire_resume_force_bypasses_host_agent_guard(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/hipfire/resume?force=true")
    assert resp.status_code == 200
    assert deck["hipfire"].calls == ["resume"]


def test_comfyui_free_unguarded_while_host_agent_busy(tmp_path, monkeypatch):
    """comfyui/free is deliberately NOT guarded — freeing VRAM helps an
    in-flight host-agent activation, it never fights it."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()
    resp = TestClient(app).post("/api/tenants/comfyui/free")
    assert resp.status_code == 200
    assert deck["comfy"].calls == ["free"]


# ===========================================================================
# Generic /{resource}/{verb} dispatch (E1 Task 7)
# ===========================================================================


def test_unknown_resource_404s(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/tenants/nope/load", json={"model": "m"})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_unsupported_verb_405s_with_kind_named(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [{"resource": "img", "kind": "comfyui",
                           "connection": {"url": "http://img:8188"},
                           "gpu_index": 2,
                           "policy_defaults": {"priority": 1, "pinned": False,
                                               "idle_ttl": 0}}])
    r = TestClient(app).post("/api/tenants/img/load", json={"model": "m"})
    assert r.status_code == 405
    assert "comfyui" in r.json()["detail"]


def test_load_body_validation_error_never_echoes_payload_and_uses_redacting_handler(
    tmp_path, monkeypatch
):
    """E1 final-review item 1: `LoadBody`/`UnloadBody` are constructed
    INSIDE control.dispatch(), not via FastAPI's automatic body binding — a
    pydantic v2 ValidationError from a malformed body subclasses ValueError,
    so it used to land on the bare ValueError handler (app/main.py:551)
    instead of the REDACTING RequestValidationError one, echoing the raw
    request body (`input`) straight into the 422. Mirrors
    tests/test_sets.py's test_create_set_validation_error_never_echoes_the_
    payload — same proof (a distinctive marker must never appear anywhere
    in the full response text), same fix idiom
    (app.routers.sets.create_set's own ValidationError -> RequestValidation
    Error re-raise), applied to control.py's hand-rolled body construction.
    Fixture resource is gguf-a (lemonade-kind, declared away from live
    topology, not the seeded "lemonade") — the seam being proven is general
    to the dispatch mechanism, not specific to any one resource name."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    sentinel = "SENTINEL-SECRET"

    resp = TestClient(app).post(
        "/api/tenants/gguf-a/load", json={"model": {sentinel: 1}}
    )

    assert resp.status_code == 422
    assert sentinel not in resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, list)          # RequestValidationError shape,
    assert all("input" not in err for err in detail)  # never the bare-ValueError one


def test_load_on_a_second_declared_lemonade_resource_dispatches_to_its_own_client(
    tmp_path, monkeypatch
):
    """The generalization's point: TWO lemonade-kind resources declared at
    once each dispatch to THEIR OWN client, resolved per-request through
    deck["local_clients"] off the resource in the URL — never a single
    shared boot-time alias. gguf-b's load must reach gguf-b's fake and
    record local/gguf-b; gguf-a's fake (and its own key) must stay
    untouched.

    The recorded `engine` field (E1 final-review item 2) is gguf_b's
    DECLARED KIND ("lemonade"), threaded through from control.dispatch()'s
    own `_declared_kind` lookup — never a literal hardcoded in the load
    handler, which a resource named anything but "lemonade" would silently
    mis-satisfy only by coincidence (human_verbs() being kind-disjoint is
    what made the old hardcoded literal happen to be right for every
    resource that could ever reach it, this one included)."""
    app, deck = make_app(tmp_path, monkeypatch)
    gguf_b = {**_GGUF_A_ENTRY, "resource": "gguf-b", "gpu_index": 3,
             "connection": {"url": "http://gguf-b:8080",
                            "metrics_url": "http://gguf-b:8001/metrics",
                            "container": "ods-gguf-b"}}
    _declare_local(deck, [_GGUF_A_ENTRY, gguf_b])
    deck["gguf-a"] = FakeLemonade()
    deck["gguf-b"] = FakeLemonade()

    resp = TestClient(app).post(
        "/api/tenants/gguf-b/load", json={"model": "extra.new.gguf"}
    )

    assert resp.status_code == 200
    assert deck["gguf-b"].calls == [("load", "extra.new.gguf")]
    assert deck["gguf-a"].calls == []
    record = deck["intent_store"].get()["local/gguf-b"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.new.gguf"
    assert record["engine"] == gguf_b["kind"]
    assert "local/gguf-a" not in deck["intent_store"].get()


# ===========================================================================
# Sets CRUD
# ===========================================================================


def test_sets_create_get_roundtrip(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/api/sets", json={"name": "Chat mode", "notes": "n"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"slug": "chat-mode"}

    got = client.get("/api/sets/chat-mode")
    assert got.status_code == 200
    assert got.json()["name"] == "Chat mode"
    assert got.json()["notes"] == "n"


def test_sets_create_bad_payload_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets", json={"name": ""})
    assert resp.status_code == 422


def test_sets_create_duplicate_without_overwrite_409(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    body = {"name": "Chat mode"}
    client.post("/api/sets", json=body)

    resp = client.post("/api/sets", json=body)
    assert resp.status_code == 409


def test_sets_create_duplicate_with_overwrite_true_succeeds(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Chat mode", "notes": "v1"})

    resp = client.post(
        "/api/sets?overwrite=true", json={"name": "Chat mode", "notes": "v2"}
    )
    assert resp.status_code == 200

    got = client.get("/api/sets/chat-mode")
    assert got.json()["notes"] == "v2"


def test_sets_create_on_corrupt_existing_file_without_overwrite_422(tmp_path, monkeypatch):
    """A corrupt file at the target slug isn't 'absent' — create's own
    existence precheck (store.get(slug)) surfaces the SAME named ValueError
    -> 422, naming the slug that's blocking the save."""
    app, _ = make_app(tmp_path, monkeypatch)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / "chat-mode.json").write_text('{"name": "bad", "unknown_field": 1}')

    resp = TestClient(app).post("/api/sets", json={"name": "Chat mode"})

    assert resp.status_code == 422
    assert "stored set 'chat-mode'" in resp.json()["detail"]


def test_sets_create_on_corrupt_existing_file_with_overwrite_recovers(tmp_path, monkeypatch):
    """Recovery-by-overwrite: ?overwrite=true skips the existence precheck
    entirely (short-circuits on `not overwrite`), so it never calls the
    corrupt slug's get() and just writes over the bad file."""
    app, _ = make_app(tmp_path, monkeypatch)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / "chat-mode.json").write_text('{"name": "bad", "unknown_field": 1}')

    resp = TestClient(app).post(
        "/api/sets?overwrite=true", json={"name": "Chat mode", "notes": "recovered"}
    )

    assert resp.status_code == 200
    got = TestClient(app).get("/api/sets/chat-mode")
    assert got.status_code == 200
    assert got.json()["notes"] == "recovered"


def test_sets_get_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/sets/nope")
    assert resp.status_code == 404


def test_sets_delete_missing_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).delete("/api/sets/nope")
    assert resp.status_code == 404


def test_sets_delete_reserved_403(tmp_path, monkeypatch):
    """A HEALTHY (parseable) _previous keeps the 403 — direction 1."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save_previous(ConfigSet(name=PREVIOUS_NAME))

    resp = TestClient(app).delete(f"/api/sets/{RESERVED_SLUG}")

    assert resp.status_code == 403
    assert deck["set_store"].get(RESERVED_SLUG) is not None


def test_sets_delete_reserved_corrupt_succeeds(tmp_path, monkeypatch):
    """[c44] A CORRUPT _previous is exactly the file the recovery path
    exists to remove — the reserved-slug 403 must not block it, or the
    rollback scenario the isolation fix targets has no way out through the
    API at all. Direction 2 of the reserved-slug guard."""
    app, _ = make_app(tmp_path, monkeypatch)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / f"{RESERVED_SLUG}.json").write_text('{"name": "bad", "unknown_field": 1}')

    resp = TestClient(app).delete(f"/api/sets/{RESERVED_SLUG}")

    assert resp.status_code == 200
    assert not (sets_dir / f"{RESERVED_SLUG}.json").exists()


def test_sets_delete_reserved_absent_still_403(tmp_path, monkeypatch):
    """No file at all is neither 'corrupt' nor a green light — the guard's
    default (protect the slot) holds when there's nothing to prove
    corruption from."""
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).delete(f"/api/sets/{RESERVED_SLUG}")
    assert resp.status_code == 403


def test_sets_delete_success(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post("/api/sets", json={"name": "Temp"})

    resp = client.delete("/api/sets/temp")

    assert resp.status_code == 200
    assert client.get("/api/sets/temp").status_code == 404


def test_sets_list_separates_previous_and_filters_it_out_of_sets(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="Chat"))
    deck["set_store"].save_previous(ConfigSet(name=PREVIOUS_NAME))

    resp = TestClient(app).get("/api/sets")

    body = resp.json()
    assert [s["name"] for s in body["sets"]] == ["Chat"]
    assert body["previous"]["name"] == PREVIOUS_NAME


def test_sets_list_previous_null_when_absent(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/sets")
    assert resp.json() == {"sets": [], "previous": None, "unreadable": []}


# --- per-file isolation [c44] ---


def test_sets_list_surfaces_unreadable_slugs_without_downing_the_route(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["set_store"].save(ConfigSet(name="Chat"))
    (tmp_path / "sets" / "bad.json").write_text('{"name": "bad", "unknown_field": 1}')

    resp = TestClient(app).get("/api/sets")

    assert resp.status_code == 200
    body = resp.json()
    assert [s["name"] for s in body["sets"]] == ["Chat"]
    assert body["unreadable"] == ["bad"]


def test_sets_get_of_corrupt_set_is_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / "bad.json").write_text('{"name": "bad", "unknown_field": 1}')
    resp = TestClient(app).get("/api/sets/bad")
    assert resp.status_code == 422
    # The raw pydantic message names neither the slug nor the filename — a
    # status-code-only assertion here would pass even pre-fix, since a bare
    # ValidationError escaping unwrapped ALSO 422s via the app-wide
    # handler. Pin the actual contract: get()'s named ValueError.
    assert "stored set 'bad'" in resp.json()["detail"]


def test_delete_route_removes_a_corrupt_set(tmp_path, monkeypatch):
    """The recovery path: DELETE must work precisely when the file can't
    parse — that is the only API path to remove it."""
    app, _ = make_app(tmp_path, monkeypatch)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    (sets_dir / "bad.json").write_text('{"name": "bad", "unknown_field": 1}')

    resp = TestClient(app).delete("/api/sets/bad")

    assert resp.status_code == 200
    assert not (sets_dir / "bad.json").exists()


# ===========================================================================
# Preview — pure, no execution
# ===========================================================================


def test_preview_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/preview")
    assert resp.status_code == 404


def test_preview_no_exec_and_estimate_arithmetic(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["comfy"] = FakeComfy(queue=0)
    deck["hipfire"] = FakeHipfire(state="parked")
    deck["litellm"] = FakeLiteLLM(default="extra.old.gguf")
    client = TestClient(app)

    client.post(
        "/api/sets",
        json={
            "name": "Full Switch",
            "durable": {"default_route_model": "extra.new.gguf", "activate_model_id": "cat-1"},
            "ephemeral": _eph(
                lemonade={"state": "loaded"},
                comfyui={"state": "free"},
                hipfire={"state": "running"},
            ),
            "policy_overrides": {"lemonade": {"priority": 10, "pinned": False, "idle_ttl": 60}},
        },
       
    )

    resp = client.post("/api/sets/full-switch/preview")

    assert resp.status_code == 200
    body = resp.json()
    step_names = [step["step"] for step in body["steps"]]
    assert step_names == [
        "free",
        "activate",
        "resume",
        "load",
        "policy_patch",
    ]
    # 5 (free, default) + 120 (activate) + 180 (resume)
    # + 5 (load, default) + 5 (policy_patch, default) == 315
    assert body["estimate_s"] == 315

    # Preview must not execute anything.
    assert deck["lemonade"].calls == []
    assert deck["comfy"].calls == []
    assert deck["hipfire"].calls == []
    assert deck["hostagent"].calls == []


def test_preview_warn_step_costs_zero(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["comfy"] = FakeComfy(queue=3)  # busy -> comfyui "free" desired emits a warn
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Free Comfy", "ephemeral": _eph(comfyui={"state": "free"})},
    )

    resp = client.post("/api/sets/free-comfy/preview")

    body = resp.json()
    assert body["steps"] == [
        {"step": "warn", "reason": "busy-skipped", "resource": "comfyui"}
    ]
    assert body["estimate_s"] == 0


# ===========================================================================
# Apply — executes, fresh snapshot per call
# ===========================================================================


def test_apply_unknown_slug_404(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).post("/api/sets/nope/apply")
    assert resp.status_code == 404


def test_apply_executes_and_uses_a_fresh_snapshot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Load It", "ephemeral": _eph(lemonade={"state": "loaded"})},
       
    )

    client.get("/api/state")  # already calls read_gpus once
    calls_before = len(deck["read_gpus"].calls)

    resp = client.post("/api/sets/load-it/apply")

    assert resp.status_code == 200
    body = resp.json()
    assert body["completed"] == [
        {"step": "load", "resource": "lemonade", "model": "extra.model.gguf"}
    ]
    assert body["failed"] is None
    assert deck["lemonade"].calls == [("load", "extra.model.gguf")]

    # The apply endpoint itself must call read_gpus exactly once more — a
    # fresh snapshot, never one reused from /api/state.
    assert len(deck["read_gpus"].calls) == calls_before + 1


def test_apply_captures_previous_snapshot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Noop", "ephemeral": _eph(hipfire={"state": "running"})},
       
    )

    resp = client.post("/api/sets/noop/apply")

    assert resp.status_code == 200
    previous = deck["set_store"].get(RESERVED_SLUG)
    assert previous is not None
    assert previous.name == PREVIOUS_NAME


def test_apply_hipfire_busy_veto_409_and_no_mutation(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        state="running",
        busy_error=GuardError("hipfire request in flight (queue_depth=1)"),
    )
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Park It", "ephemeral": _eph(hipfire={"state": "parked"})},
       
    )

    resp = client.post("/api/sets/park-it/apply")

    assert resp.status_code == 409
    assert "in flight" in resp.json()["detail"]
    assert deck["hipfire"].calls == []
    assert deck["set_store"].get(RESERVED_SLUG) is None
    # A vetoed apply never reached _record_goal_intents either — nothing
    # ran, so there is nothing to declare (mirrors the control-route
    # sibling invariant, test_guard_refused_unload_records_nothing /
    # test_failed_park_records_no_intent above: "a refused action never
    # happened").
    assert deck["intent_store"].get() == {}


def test_apply_force_skips_hipfire_busy_veto(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        state="running",
        busy_error=GuardError("hipfire request in flight (queue_depth=1)"),
    )
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Park It", "ephemeral": _eph(hipfire={"state": "parked"})},
       
    )

    resp = client.post("/api/sets/park-it/apply?force=true")

    assert resp.status_code == 200
    assert resp.json()["failed"] is None
    assert deck["hipfire"].calls == ["park"]
    assert deck["hipfire"].park_forces == [True]


def test_apply_host_agent_busy_veto_409_and_no_mutation(tmp_path, monkeypatch):
    """HTTP-level proof of the routers/sets.py wiring: hostagent=deck["hostagent"]
    passthrough + BusyError -> 409 app handler. Mirrors the hipfire-busy-veto
    precedent above but for the new host-agent guard (plan contains a
    "load" step, a guarded verb)."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["hostagent"] = _BusyHostAgent()
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Load It", "ephemeral": _eph(lemonade={"state": "loaded"})},
    )

    resp = client.post("/api/sets/load-it/apply")

    assert resp.status_code == 409
    assert "host agent is busy" in resp.json()["detail"]
    assert deck["lemonade"].calls == []
    assert deck["set_store"].get(RESERVED_SLUG) is None
    # Same invariant as the hipfire-busy veto above: a refused apply never
    # reached _record_goal_intents.
    assert deck["intent_store"].get() == {}


def test_apply_force_skips_host_agent_busy_veto(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["hostagent"] = _BusyHostAgent()
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Load It", "ephemeral": _eph(lemonade={"state": "loaded"})},
    )

    resp = client.post("/api/sets/load-it/apply?force=true")

    assert resp.status_code == 200
    assert resp.json()["failed"] is None
    assert deck["lemonade"].calls == [("load", "extra.model.gguf")]


def test_hipfire_park_busy_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(
        busy_error=GuardError(
            "hipfire served a request 12s ago (activity window 600s; "
            "pass force=true to override)"
        )
    )

    resp = TestClient(app).post("/api/tenants/hipfire/park")

    assert resp.status_code == 409
    assert "activity window" in resp.json()["detail"]
    assert deck["hipfire"]._state == "running"


def test_hipfire_park_force_query_param(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(busy_error=GuardError("busy"))

    resp = TestClient(app).post("/api/tenants/hipfire/park?force=true")

    assert resp.status_code == 200
    assert deck["hipfire"].park_forces == [True]
    assert deck["hipfire"]._state == "parked"


# ===========================================================================
# Policy
# ===========================================================================


def test_policy_get_default(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/policy")
    assert resp.status_code == 200
    assert resp.json() == _TEST_POLICY_DEFAULTS


def test_policy_put_roundtrip_partial_by_tenant(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.put(
        "/api/policy",
        json={"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}},
       
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["lemonade"] == {"priority": 5, "pinned": True, "idle_ttl": 0}
    assert body["comfyui"] == _TEST_POLICY_DEFAULTS["comfyui"]  # untouched

    again = client.get("/api/policy")
    assert again.json() == body


def test_policy_put_rejects_undeclared_resource_and_reserved_key(tmp_path, monkeypatch):
    """E1 Task 4: PUT is refused for undeclared resources (declaration-driven).
    The reserved ``_auto`` config key is still rejected."""
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    # Undeclared resource: rejected with 422
    resp = client.put(
        "/api/policy",
        json={"sparky-vllm": {"priority": 5, "pinned": True, "idle_ttl": 0}},
    )
    assert resp.status_code == 422

    # Reserved key: still rejected
    reserved = client.put(
        "/api/policy",
        json={"_auto": {"priority": 5, "pinned": True, "idle_ttl": 0}},
    )
    assert reserved.status_code == 422


def test_policy_put_bad_field_type_422(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).put(
        "/api/policy",
        json={"lemonade": {"priority": "high", "pinned": True, "idle_ttl": 0}},

    )
    assert resp.status_code == 422


def test_policy_put_undeclared_resource_rejected(tmp_path, monkeypatch):
    """E1 Task 4: PUT for undeclared resources is rejected with 422."""
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.put(
        "/api/policy",
        json={"sparky-vllm": {"priority": 5, "pinned": True, "idle_ttl": 0}},
    )
    assert resp.status_code == 422
    assert "unknown resource" in resp.json()["detail"]


def test_policy_put_declared_resources_accepted(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.put(
        "/api/policy",
        json={"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}},
    )
    assert resp.status_code == 200
    assert resp.json()["lemonade"]["priority"] == 5


# ===========================================================================
# Open access — no admin token required (auth deliberately removed)
# ===========================================================================


def test_mutating_endpoints_open_without_any_auth(tmp_path, monkeypatch):
    """Ops-first mode: every endpoint works with zero auth headers. The
    admin-token/proxy-key gate was removed 2026-07-22 (Tim: config and
    operations now, security later)."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.post("/api/tenants/comfyui/free")
    assert resp.status_code == 200
    assert deck["comfy"].calls == ["free"]

    resp = client.post("/api/sets", json={"name": "Open"})
    assert resp.status_code == 200

    resp = client.put(
        "/api/policy",
        json={
            "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
            "comfyui": {"priority": 60, "pinned": False, "idle_ttl": 300},
            "hipfire": {"priority": 90, "pinned": True, "idle_ttl": 0},
        },
    )
    assert resp.status_code == 200


# ===========================================================================
# Intent recording — every deliberate action leaves a last-known-good record
# ===========================================================================


def test_lemonade_load_records_loaded_intent(tmp_path, monkeypatch):
    """Every deliberate action must leave a record, or the reconciler has
    nothing to restore to after a reboot. `engine` is asserted against the
    "lemonade" resource's DECLARED kind (E1 final-review item 2), not a bare
    literal — see test_load_on_a_second_declared_lemonade_resource_
    dispatches_to_its_own_client for the sibling case where resource name
    and kind actually diverge."""
    app, deck = make_app(tmp_path, monkeypatch)

    TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.qwen.gguf"}
    )

    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.qwen.gguf"
    expected_kind = next(e["kind"] for e in _ENGINES if e["resource"] == "lemonade")
    assert record["engine"] == expected_kind


def test_lemonade_load_records_the_prefixed_name_actually_loaded(tmp_path, monkeypatch):
    """Intent must match what the engine was told, not what the caller typed
    — otherwise the recorded model can never equal the observed one."""
    app, deck = make_app(tmp_path, monkeypatch)

    TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "qwen.gguf"}
    )

    assert deck["intent_store"].get()["local/lemonade"]["model"] == "extra.qwen.gguf"


def test_lemonade_unload_records_unloaded_intent(tmp_path, monkeypatch):
    """A park is intent, not the absence of it — this is what stops the
    reconciler reloading something you deliberately took down."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf")

    TestClient(app).post("/api/tenants/lemonade/unload", json={})

    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "unloaded"
    assert record["model"] is None


def test_unload_records_intent_even_when_engine_fails(tmp_path, monkeypatch):
    """Intent is the operator's statement, recorded before actuation: an
    unload whose engine call then dies must leave intent=unloaded (deriving
    the inert 'unexpected', never a restore). 2026-08-06 design ruling."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded="extra.m.gguf",
                                    raise_on_unload=EngineError("engine boom"))
    client = TestClient(app)

    resp = client.post("/api/tenants/lemonade/unload", json={"model": None})

    assert resp.status_code == 502
    assert deck["intent_store"].get()["local/lemonade"]["state"] == "unloaded"


def test_load_records_intent_even_when_engine_fails(tmp_path, monkeypatch):
    """A failed deliberate load leaves intent=loaded so the reconciler
    retries it under the failure budget — bounded auto-retry, by design."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(raise_on_load=EngineError("engine boom"))
    client = TestClient(app)

    resp = client.post("/api/tenants/lemonade/load", json={"model": "m.gguf"})

    assert resp.status_code == 502
    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.m.gguf"


def test_guard_refused_unload_records_nothing(tmp_path, monkeypatch):
    """Guards run BEFORE the record: a 409 (nothing loaded) must not write
    intent — a refused action never happened."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    client = TestClient(app)

    resp = client.post("/api/tenants/lemonade/unload", json={"model": None})

    assert resp.status_code == 409
    assert "local/lemonade" not in deck["intent_store"].get()


def test_hipfire_park_records_unloaded_intent(tmp_path, monkeypatch):
    """`engine` is asserted against the "hipfire" resource's DECLARED kind
    (E1 final-review item 2), not a bare literal."""
    app, deck = make_app(tmp_path, monkeypatch)

    TestClient(app).post("/api/tenants/hipfire/park")

    record = deck["intent_store"].get()["local/hipfire"]
    assert record["state"] == "unloaded"
    expected_kind = next(e["kind"] for e in _ENGINES if e["resource"] == "hipfire")
    assert record["engine"] == expected_kind


def test_hipfire_resume_records_loaded_intent(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)

    TestClient(app).post("/api/tenants/hipfire/resume")

    record = deck["intent_store"].get()["local/hipfire"]
    assert record["state"] == "loaded"
    # hipfire is single-model and the Deck does not choose that model (it
    # comes from the litellm route table), so intent records None: "loaded,
    # no opinion which model". A name we cannot observe would derive as
    # permanent drift.
    assert record["model"] is None


def test_failed_load_still_records_intent(tmp_path, monkeypatch):
    """2026-08-06 reversal (see test_load_records_intent_even_when_engine_fails
    above): lemonade's record moved before the engine call, so a load that
    raises still leaves intent=loaded — exercised here via the fake's other
    failure-injection path (`load_raises`, distinct from `raise_on_load`)."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"].load_raises = EngineError("boom")

    TestClient(app).post(
        "/api/tenants/lemonade/load", json={"model": "extra.qwen.gguf"}
    )

    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.qwen.gguf"


def test_failed_park_records_no_intent(tmp_path, monkeypatch):
    """Same invariant on the other side: a park that the route guard refused
    never happened, so it must not become the state we restore to."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hipfire"] = FakeHipfire(state="running")
    deck["hipfire"].fail = GuardError("hipfire serves the default route")

    resp = TestClient(app).post("/api/tenants/hipfire/park")

    assert resp.status_code == 409
    assert deck["intent_store"].get() == {}


def test_guarded_action_records_no_intent(tmp_path, monkeypatch):
    """A 409 from the host-agent busy guard means no engine call was made at
    all — there is nothing to record."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["hostagent"] = _BusyHostAgent()

    resp = TestClient(app).post("/api/tenants/hipfire/park")

    assert resp.status_code == 409
    assert deck["intent_store"].get() == {}


def test_comfyui_free_records_no_intent(tmp_path, monkeypatch):
    """ComfyUI's /free drops cached VRAM; the server stays up and keeps
    reporting itself loaded. Recording 'unloaded' would derive as a
    permanent 'unexpected' — an alarm that is always on is an alarm nobody
    reads. ComfyUI gets intent only once something can actually park it."""
    app, deck = make_app(tmp_path, monkeypatch)

    resp = TestClient(app).post("/api/tenants/comfyui/free")

    assert resp.status_code == 200
    assert deck["intent_store"].get() == {}


def test_apply_records_the_declared_goals_via_the_router(tmp_path, monkeypatch):
    """A set apply is as deliberate as a button press — the same actions
    through a different door must leave the same record. Recording is now
    GOAL-derived (app.sets._record_goal_intents, Task 5), not per-completed
    -step: here both declared goals also happen to be the steps that ran, so
    the outcome matches the pre-Task-5 one; the sibling test below
    (test_apply_records_the_declared_goal_even_when_the_step_fails) is the
    case where the two diverge."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["hipfire"] = FakeHipfire(state="parked")
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={
            "name": "Chat",
            "ephemeral": _eph(
                lemonade={"state": "loaded"},
                hipfire={"state": "running"},
            ),
        },
    )

    resp = client.post("/api/sets/chat/apply")

    assert resp.status_code == 200
    intents = deck["intent_store"].get()
    assert intents["local/lemonade"]["state"] == "loaded"
    assert intents["local/lemonade"]["model"] == "extra.model.gguf"
    assert intents["local/hipfire"]["state"] == "loaded"


def test_apply_records_the_declared_goal_even_when_the_step_fails(tmp_path, monkeypatch):
    """2026-08-10 reversal of this test's own prior assertion (max-review
    #2/c40, Task 5's design ruling): intent is now recorded from the
    DECLARED goal, before the step runs — not from the completed-step list
    — so a load that then fails still leaves intent=loaded, matching the
    single-tenant control route's test_failed_load_still_records_intent
    above. Recording nothing here (the old behavior) would make a crashed
    apply invisible to the reconciler's restore-on-failure budget."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["lemonade"] = FakeLemonade(loaded=None)
    deck["lemonade"].fail = EngineError("boom")
    client = TestClient(app)
    client.post(
        "/api/sets",
        json={"name": "Chat", "ephemeral": _eph(lemonade={"state": "loaded"})},
    )

    resp = client.post("/api/sets/chat/apply")

    assert resp.status_code == 200
    assert resp.json()["failed"] == {
        "step": "load", "resource": "lemonade", "model": "extra.model.gguf"
    }
    record = deck["intent_store"].get()["local/lemonade"]
    assert record["state"] == "loaded"
    assert record["model"] == "extra.model.gguf"


# ===========================================================================
# /api/state lifecycle block, /api/lifecycle/*
# ===========================================================================


def test_state_includes_lifecycle_block(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="unloaded", model=None, engine="hipfire")
    deck["intent_store"] = store
    deck["hipfire"] = FakeHipfire(state="parked")

    body = TestClient(app).get("/api/state").json()

    assert body["lifecycle"]["local/hipfire"]["status"] == "parked"
    assert body["lifecycle"]["local/hipfire"]["intent"]["state"] == "unloaded"
    assert body["lifecycle"]["local/hipfire"]["observed"]["loaded"] is False
    assert "last_healthy_ts" in body["lifecycle"]["local/hipfire"]


def test_lifecycle_block_distinguishes_parked_from_down(tmp_path, monkeypatch):
    """The whole point, expressed at the API boundary: the SAME observation
    (hipfire stopped) reads 'parked' above and 'down' here. Only intent
    differs."""
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model=None, engine="hipfire")
    deck["intent_store"] = store
    deck["hipfire"] = FakeHipfire(state="parked")

    body = TestClient(app).get("/api/state").json()

    assert body["lifecycle"]["local/hipfire"]["status"] == "down"


def test_lifecycle_block_omits_spark_when_none_configured(tmp_path, monkeypatch):
    """An undeclared resource must not appear as a phantom failure."""
    app, deck = make_app(tmp_path, monkeypatch)

    body = TestClient(app).get("/api/state").json()

    assert "sparky/slot0" not in body["lifecycle"]


def test_lifecycle_view_carries_every_swap_nodes_slot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    wire_swap_node(deck, "boxa", FakeSpark(), label="Box Alpha")
    wire_swap_node(deck, "boxb", FakeSpark(), label="Box Beta")
    client = TestClient(app)
    lifecycle = client.get("/api/state").json()["lifecycle"]
    assert "boxa/slot0" in lifecycle
    assert "boxb/slot0" in lifecycle
    assert "sparky/slot0" not in lifecycle


def test_control_none_node_emits_no_slot(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["node_store"].add(
        {"id": "watcher", "label": "Watch Only", "agent_kind": "node-agent",
         "address": "http://watcher:7720",
         "serving_address": "http://watcher:8000"}, credential="key-watcher")
    client = TestClient(app)
    lifecycle = client.get("/api/state").json()["lifecycle"]
    assert "watcher/slot0" not in lifecycle


def test_clear_quarantine_releases_the_key(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")
    store.note_failure("local/hipfire")
    store.note_failure("local/hipfire")
    deck["intent_store"] = store
    assert store.get()["local/hipfire"]["quarantined"] is True

    resp = TestClient(app).post("/api/lifecycle/quarantine/local/hipfire/clear")

    assert resp.status_code == 200
    assert store.get()["local/hipfire"]["quarantined"] is False


def test_clear_quarantine_unknown_key_is_404(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["intent_store"] = IntentStore(tmp_path / "intent.json")

    resp = TestClient(app).post("/api/lifecycle/quarantine/nope/nothing/clear")

    assert resp.status_code == 404


def test_adopt_records_observed_state_as_intent(tmp_path, monkeypatch):
    """Bootstrap adoption: turn an unmanaged resource into a managed one
    WITHOUT touching it — adopting must never restart anything."""
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    deck["intent_store"] = store
    deck["hipfire"] = FakeHipfire(state="running")

    resp = TestClient(app).post("/api/lifecycle/adopt/local/hipfire")

    assert resp.status_code == 200
    record = store.get()["local/hipfire"]
    assert record["state"] == "loaded"
    assert record["engine"] == "hipfire"
    assert record["failures"] == 0


def test_adopt_a_stopped_resource_records_a_park(tmp_path, monkeypatch):
    """Adoption records what IS, not what we wish: adopting a stopped engine
    means 'this is deliberately off', which is exactly what stops the
    reconciler from starting it."""
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    deck["intent_store"] = store
    deck["hipfire"] = FakeHipfire(state="parked")

    TestClient(app).post("/api/lifecycle/adopt/local/hipfire")

    assert store.get()["local/hipfire"]["state"] == "unloaded"


def test_adopt_does_not_actuate(tmp_path, monkeypatch):
    """'Start managing this' must be a safe button. If adopt could restart
    a serving model, nobody would ever press it."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["intent_store"] = IntentStore(tmp_path / "intent.json")
    deck["hipfire"] = FakeHipfire(state="running")
    deck["lemonade"] = FakeLemonade(loaded="qwen")

    TestClient(app).post("/api/lifecycle/adopt/local/hipfire")

    assert deck["hipfire"].calls == []
    assert deck["lemonade"].calls == []


def test_adopt_unknown_key_is_404(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    deck["intent_store"] = IntentStore(tmp_path / "intent.json")

    resp = TestClient(app).post("/api/lifecycle/adopt/nope/nothing")

    assert resp.status_code == 404


class _UnreachableHipfire(FakeHipfire):
    """status() raising is what app.state degrades to tenant 'unknown', which
    app.observe maps to unreachable (we failed to look, not looked and saw
    nothing)."""

    def status(self):
        raise EngineError("connection refused")


def test_adopt_an_unreachable_resource_is_refused(tmp_path, monkeypatch):
    """Failing to look is not evidence of anything, and recording 'unloaded'
    from it would manufacture a park nobody asked for."""
    app, deck = make_app(tmp_path, monkeypatch)
    store = IntentStore(tmp_path / "intent.json")
    deck["intent_store"] = store
    deck["hipfire"] = _UnreachableHipfire()

    resp = TestClient(app).post("/api/lifecycle/adopt/local/hipfire")

    assert resp.status_code == 409
    assert store.get() == {}


def test_adopt_a_swap_nodes_slot(tmp_path, monkeypatch):
    """N1 T12: adopt's engine_for lookup now threads the live control:"swap"
    id-set through, so a non-legacy swap node's slot key (never the frozen
    LEGACY_SPARK_SEED_ID "sparky") resolves to the "spark" engine too — the
    gap the old static SPARK_SLOT_KEY row in observe.py's engine table could
    never have exercised, since it only ever matched the one hardcoded id."""
    app, deck = make_app(tmp_path, monkeypatch)
    wire_swap_node(deck, "boxa", FakeSpark(), label="Box Alpha")

    resp = TestClient(app).post("/api/lifecycle/adopt/boxa/slot0")

    assert resp.status_code == 200
    record = deck["intent_store"].get()["boxa/slot0"]
    assert record["state"] == "loaded"
    assert record["engine"] == "spark"


def test_adopt_a_non_legacy_local_resource(tmp_path, monkeypatch):
    """E1 T13 residue fix: the sibling of test_adopt_a_swap_nodes_slot above,
    for the LOCAL half of engine_for's same resource==kind-name-coincidence
    bug — the old static _LOCAL_ENGINE_BY_KEY table only ever matched a
    resource literally named lemonade/comfyui/hipfire, so gguf-a (declared
    kind "lemonade", not named "lemonade") 404'd with "no engine owns
    'local/gguf-a'" even though it was declared, reachable, and observed."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    deck["gguf-a"] = FakeLemonade(loaded="a.gguf")
    deck["intent_store"] = IntentStore(tmp_path / "intent.json")

    resp = TestClient(app).post("/api/lifecycle/adopt/local/gguf-a")

    assert resp.status_code == 200
    record = deck["intent_store"].get()["local/gguf-a"]
    assert record["state"] == "loaded" and record["engine"] == "lemonade"


def test_get_auto_defaults_to_enabled(tmp_path, monkeypatch):
    """Automation is on by default — its absence is what let hipfire stay
    dead for 26 hours."""
    app, _ = make_app(tmp_path, monkeypatch)

    resp = TestClient(app).get("/api/lifecycle/auto")

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True


def test_auto_can_be_turned_off_and_back_on(tmp_path, monkeypatch):
    """THE point of this route: shipping auto-actuation with no brake means
    the only way to stop the reconciler is hand-editing policy.json and
    restarting the container."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    assert client.post("/api/lifecycle/auto", json={"enabled": False}).status_code == 200
    assert client.get("/api/lifecycle/auto").json()["enabled"] is False
    assert deck["policy_store"].auto_enabled() is False

    assert client.post("/api/lifecycle/auto", json={"enabled": True}).status_code == 200
    assert deck["policy_store"].auto_enabled() is True


def test_turning_auto_off_preserves_tenant_policies(tmp_path, monkeypatch):
    """set_auto() seeds defaults on a fresh file; going through the route must
    not cost the deck its tenant policies."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    client.post("/api/lifecycle/auto", json={"enabled": False})

    policies = deck["policy_store"].get()
    assert "hipfire" in policies and "lemonade" in policies and "comfyui" in policies
    assert "_auto" not in policies


def test_auto_rejects_a_non_boolean(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)

    resp = TestClient(app).post("/api/lifecycle/auto", json={"enabled": "yes"})

    assert resp.status_code == 422


def test_state_reports_node_identity(tmp_path, monkeypatch):
    """The board needs a name for this box; /state is where it comes from."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    body = client.get("/api/state").json()

    assert body["node"] == {"id": "local", "label": "local"}


def test_state_node_label_comes_from_the_registry(tmp_path, monkeypatch):
    """Label provenance: seeded from MODEL_DECK_NODE_LABEL on first boot,
    then owned by the registry (app/node_store.py's seed_if_missing) -- a
    changed env var after that seed already exists has no effect. That
    second half is the valuable assertion: it pins seed-once at the API
    surface, not just at the store level (see tests/test_nodes_wiring.py's
    test_registry_wins_over_changed_env_after_seed for the same pin on the
    spark entry)."""
    monkeypatch.setenv("MODEL_DECK_NODE_LABEL", "autarch")
    app1, _ = make_app(tmp_path, monkeypatch)
    assert TestClient(app1).get("/api/state").json()["node"]["label"] == "autarch"

    # Same data dir (set once, for this whole test, by the autouse
    # _model_deck_default_data_dir fixture) -- env now points elsewhere, but
    # nodes.json already exists, so the registry entry wins.
    monkeypatch.setenv("MODEL_DECK_NODE_LABEL", "elsewhere")
    app2, _ = make_app(tmp_path, monkeypatch)
    assert TestClient(app2).get("/api/state").json()["node"]["label"] == "autarch"


def test_node_label_empty_env_var_falls_through_to_default(monkeypatch):
    """Empty env var must not override the Python default. When compose.yaml
    passes MODEL_DECK_NODE_LABEL= (empty), Settings must still yield "local"."""
    from app.settings import Settings

    monkeypatch.setenv("MODEL_DECK_NODE_LABEL", "")
    assert Settings().node_label == "local"


# ===========================================================================
# /api/facts, /api/facts/drift, /api/facts/declared/{key}
# ===========================================================================


def test_facts_endpoint_resolves_declared_over_derived(tmp_path, monkeypatch):
    from app.characteristics import CharacteristicsStore
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("model/m", {
        "label": {"value": "auto", "source": "config.json", "derived_ts": "t"}})
    declared = DeclaredStore(tmp_path / "d.json")
    declared.put("model/m", {"label": "human"})
    deck["characteristics_store"] = characteristics
    deck["declared_store"] = declared

    body = TestClient(app).get("/api/facts").json()

    assert body["model/m"]["label"]["value"] == "human"
    assert body["model/m"]["label"]["origin"] == "declared"
    assert body["model/m"]["label"]["shadowed_value"] == "auto"


def test_put_declared_rejects_a_derivable_field(tmp_path, monkeypatch):
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["declared_store"] = DeclaredStore(tmp_path / "d.json")

    resp = TestClient(app).put("/api/facts/declared/model/m",
                               json={"max_model_len": 262144})

    assert resp.status_code == 422


def test_put_declared_accepts_tools_verified(tmp_path, monkeypatch):
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/model/m",
                               json={"tools_verified": True})

    assert resp.status_code == 200
    assert store.entry("model/m")["tools_verified"] is True


def test_put_declared_rejects_empty_key(tmp_path, monkeypatch):
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/", json={"label": "x"})

    assert resp.status_code == 422
    assert store.get() == {}


def test_put_declared_rejects_trailing_slash_key(tmp_path, monkeypatch):
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/model/m/", json={"label": "x"})

    assert resp.status_code == 422
    assert store.get() == {}


def test_put_declared_rejects_unprefixed_key(tmp_path, monkeypatch):
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/junk/x", json={"label": "x"})

    assert resp.status_code == 422
    assert store.get() == {}


def test_put_declared_rejects_empty_fields_body(tmp_path, monkeypatch):
    """An empty body materializes an empty entry today — reject it before it
    ever reaches the store."""
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/model/m", json={})

    assert resp.status_code == 422
    assert store.get() == {}


def test_put_declared_accepts_engine_key_with_interior_slash(tmp_path, monkeypatch):
    """Engine keys legitimately contain a second slash ('engine/sparky/vllm')
    — only an empty remainder or a trailing slash is rejected."""
    from app.declared import DeclaredStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = DeclaredStore(tmp_path / "d.json")
    deck["declared_store"] = store

    resp = TestClient(app).put("/api/facts/declared/engine/sparky/vllm",
                               json={"label": "x"})

    assert resp.status_code == 200
    assert store.entry("engine/sparky/vllm")["label"] == "x"


def test_drift_endpoint_reports_context_mismatch(tmp_path, monkeypatch):
    """The gateway route ALIAS ("default") is never a checkpoint id — only
    the route's RESOLVED model (litellm_params.model, "m" here once the
    "openai/" prefix is stripped) can join against a facts key ("model/m").
    A join keyed by alias would silently find nothing."""
    from app.characteristics import CharacteristicsStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("model/m", {
        "max_model_len_live": {"value": 262144, "source": "/v1/models", "derived_ts": "t"}})
    deck["characteristics_store"] = characteristics
    deck["litellm"] = FakeLiteLLM(default="m")   # alias "default" resolves to model id "m"
    deck["litellm"].max_input_tokens = {"m": 131072}   # keyed by RESOLVED id, not alias

    body = TestClient(app).get("/api/facts/drift").json()

    assert body["model/m"][0]["field"] == "max_input_tokens"


def test_drift_endpoint_reports_live_context_exceeding_checkpoint_max(tmp_path, monkeypatch):
    """max_model_len needs no gateway I/O: it's a live-vs-checkpoint check
    within the SAME key's own already-derived facts (max_model_len_live
    from /v1/models, max_position_embeddings from config.json)."""
    from app.characteristics import CharacteristicsStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("model/m", {
        "max_position_embeddings": {"value": 131072, "source": "config.json", "derived_ts": "t"},
        "max_model_len_live": {"value": 262144, "source": "/v1/models", "derived_ts": "t"},
    })
    deck["characteristics_store"] = characteristics

    body = TestClient(app).get("/api/facts/drift").json()

    assert body["model/m"][0]["field"] == "max_model_len"
    assert body["model/m"][0]["severity"] == "mismatch"


def test_drift_endpoint_no_drift_when_live_context_within_checkpoint_max(tmp_path, monkeypatch):
    from app.characteristics import CharacteristicsStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("model/m", {
        "max_position_embeddings": {"value": 262144, "source": "config.json", "derived_ts": "t"},
        "max_model_len_live": {"value": 131072, "source": "/v1/models", "derived_ts": "t"},
    })
    deck["characteristics_store"] = characteristics

    assert TestClient(app).get("/api/facts/drift").json() == {}


def test_drift_endpoint_is_empty_when_nothing_disagrees(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)

    assert TestClient(app).get("/api/facts/drift").json() == {}


# ===========================================================================
# Settings API (Task 7)
# ===========================================================================


def test_put_and_get_settings_round_trip(tmp_path, monkeypatch):
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    client.put("/api/settings/engines/sparky/vllm",
               json={"namespace": "args", "values": {"generation-config": "auto"}})
    body = client.get("/api/settings/engines/sparky/vllm").json()

    assert body["args"]["generation-config"] == "auto"


def test_settings_responses_do_not_leak_the_internal_updated_ts_bookkeeping(tmp_path, monkeypatch):
    """updated_ts is app.routers._settings_drift's write-tracking clock, not
    a documented field of this response shape (review round finding: fix if
    cheap). Both GET and the PUT's own return value must omit it."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    put_body = client.put(
        "/api/settings/engines/sparky/vllm",
        json={"namespace": "args", "values": {"generation-config": "auto"}}).json()
    get_body = client.get("/api/settings/engines/sparky/vllm").json()

    assert "updated_ts" not in put_body
    assert "updated_ts" not in get_body


def test_put_settings_missing_body_fields_is_422_not_500(tmp_path, monkeypatch):
    """A malformed body (missing namespace/values) must fail the same way
    the container allowlist does — a ValueError the global handler turns
    into 422 — not subscript straight into a KeyError and 500."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")

    resp = TestClient(app).put("/api/settings/engines/sparky/vllm", json={"namespace": "args"})

    assert resp.status_code == 422


def test_get_settings_rejects_an_unknown_kind_422(tmp_path, monkeypatch):
    """Optional fix, final branch review 2026-08-07: PUT already 422s an
    unknown `kind` via SettingsStore.put's own ValueError; GET had no
    equivalent check and silently returned 200 {} for a typo'd kind,
    indistinguishable from a real, empty scope of a valid kind."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")

    resp = TestClient(app).get("/api/settings/bogus-kind/sparky/vllm")

    assert resp.status_code == 422


def test_put_settings_rejects_deck_managed_container_field(tmp_path, monkeypatch):
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")

    resp = TestClient(app).put("/api/settings/engines/sparky/vllm",
                               json={"namespace": "container",
                                     "values": {"volumes": ["/a:/b"]}})

    assert resp.status_code == 422


def test_effective_settings_include_argline_and_warnings(tmp_path, monkeypatch):
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"brand-new-flag": "1"})
    deck["settings_store"] = store

    body = TestClient(app).get(
        "/api/settings/effective/sparky/vllm/Qwen3.6-35B-A3B-heretic-NVFP4").json()

    assert "--brand-new-flag 1" in body["argline"]
    assert body["warnings"][0]["class"] == "unknown"


def test_effective_settings_resolves_five_layers_against_a_real_catalog(tmp_path, monkeypatch):
    """Review-round finding: the whole catalog-present path of /effective
    was untested — engine defaults extracted from a real option_catalog
    (including the `default not in (None, "None")` filter), checkpoint
    recommendations from recommended_sampling (raw int -> normalized str,
    proving BOTH app.argline.normalize_args_map wrappers in _resolve), the
    models/engine_models store layers, five-layer per-key precedence (a
    store layer beating a derived one), and a catalog-driven non-'unknown'
    warning actually firing.

    F2, CRITICAL (final branch review, 2026-08-07): the catalog is built by
    running the REAL app.harvest.parse_probe_output over a probe payload
    shaped exactly like PROBE_SOURCE's own output (extending
    tests/test_harvest.py's PROBE_OUTPUT) rather than hand-written option
    dicts. A hand-seeded `"default": 4096` (a Python int) is a shape
    parse_probe_output can never actually produce — the real probe always
    stores `repr(action.default)`, a STRING — and building the catalog
    that way is exactly what masked the F2 bug this test now also covers:
    a repr'd string default (`"'auto'"`) or a repr'd `False` reaching
    engine_defaults, and the resolved settings/argline, undecoded.
    """
    from app.characteristics import CharacteristicsStore
    from app.harvest import parse_probe_output
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)

    probe_output = json.dumps({"options": [
        # default=4096 (int): must survive the `default not in (None,
        # "None")` filter and normalize_args_map's int->str axis to
        # become an engine_defaults entry. repr'd exactly as the real
        # probe would: `repr(4096)`, not the bare int.
        {"flags": ["--max-model-len"], "type": "int", "choices": None,
         "default": repr(4096), "nargs": None, "cls": "_StoreAction",
         "help": "Model context length."},
        # default=None (repr'd "None"): must NOT become an
        # engine_defaults entry.
        {"flags": ["--quantization"], "type": "str",
         "choices": ["awq", "gptq"], "default": repr(None), "nargs": None,
         "cls": "_StoreAction", "help": "Quantization method."},
        # F2: a repr'd STRING default (`repr("auto") == "'auto'"`) — the
        # exact shape a real argparse `default="auto"` produces. Must
        # decode to the plain string 'auto', not the two-quote literal
        # text "'auto'", and must not warn as a type mismatch against its
        # own choices.
        {"flags": ["--tokenizer-mode"], "type": "str",
         "choices": ["auto", "slow", "mistral", "custom"],
         "default": repr("auto"), "nargs": None, "cls": "_StoreAction",
         "help": "Tokenizer mode."},
        # F2: a store_true option whose default is False (repr'd
        # "False"). An off-by-default flag's honest rendering is an
        # ABSENT flag, not a bare flag turned on and not the literal
        # string 'False' as a value.
        {"flags": ["--enable-log-requests"], "type": None, "choices": None,
         "default": repr(False), "nargs": 0, "cls": "_StoreTrueAction",
         "help": "Enable request logging."},
    ]})

    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {
        "option_catalog": parse_probe_output(probe_output, engine_version="test", now="t"),
    })
    characteristics.put_fields("model/TestModel-7B", {
        # Raw int, straight from generation_config.json shape — proves the
        # checkpoint_recommendations normalize_args_map wrapper.
        "recommended_sampling": {
            "value": {"top-k": 40},
            "source": "generation_config.json", "derived_ts": "t",
        },
    })
    deck["characteristics_store"] = characteristics

    store = SettingsStore(tmp_path / "s.json")
    # Engine layer: overrides the catalog default for the SAME key (proves
    # a store layer beats a derived layer), and sets an out-of-choices
    # value for a real catalog option (proves a catalog-driven "type"
    # warning, not just "unknown", can fire).
    store.put("engines", "sparky/vllm", "args",
             {"max-model-len": "8192", "quantization": "bogus-quant"})
    # models vs engine_models: engine_models is more specific and must win.
    store.put("models", "TestModel-7B", "args", {"model-layer-flag": "from-model"})
    store.put("engine_models", "sparky/vllm|TestModel-7B", "args",
             {"model-layer-flag": "from-engine-model"})
    deck["settings_store"] = store

    body = TestClient(app).get(
        "/api/settings/effective/sparky/vllm/TestModel-7B").json()
    resolved = body["resolved"]

    # Store layer (engine) beats derived layer (engine_defaults) for the
    # same key.
    assert resolved["max-model-len"]["value"] == "8192"
    assert resolved["max-model-len"]["origin"] == "declared"
    assert resolved["max-model-len"]["layer"] == "engine"
    assert "--max-model-len 8192" in body["argline"]

    # checkpoint_recommendations: raw int normalized to str.
    assert resolved["top-k"]["value"] == "40"
    assert isinstance(resolved["top-k"]["value"], str)
    assert resolved["top-k"]["origin"] == "derived"
    assert resolved["top-k"]["layer"] == "checkpoint_recommendations"

    # engine_models (most specific) beats models for the same key.
    assert resolved["model-layer-flag"]["value"] == "from-engine-model"
    assert resolved["model-layer-flag"]["layer"] == "engine_model"

    # A catalog-driven, non-'unknown' warning fires for the out-of-choices
    # quantization value.
    quant_warnings = [w for w in body["warnings"] if w["key"] == "quantization"]
    assert quant_warnings
    assert quant_warnings[0]["class"] == "type"
    # max-model-len IS in the catalog — must not also read as unknown.
    assert not any(w["key"] == "max-model-len" for w in body["warnings"])

    # F2: the repr'd string default decodes to the plain value.
    assert resolved["tokenizer-mode"]["value"] == "auto"
    assert resolved["tokenizer-mode"]["origin"] == "derived"
    assert resolved["tokenizer-mode"]["layer"] == "engine_defaults"
    # CHANGED, Task 7 (design decision 3): the argline is declared-only —
    # engine_defaults never renders as a flag, even decoded correctly, even
    # though 'resolved' above still carries it with full provenance. Was
    # `assert "--tokenizer-mode auto" in body["argline"]` before Task 7.
    assert "--tokenizer-mode auto" not in body["argline"]
    assert "'auto'" not in body["argline"]

    # F2: the store_true default of False is an absent flag, not a value.
    assert "enable-log-requests" not in resolved
    assert "--enable-log-requests" not in body["argline"]

    # F2: neither engine-default-derived key trips a warning of its own —
    # these are the engine's OWN harvested defaults.
    assert not any(w["key"] in ("tokenizer-mode", "enable-log-requests")
                   for w in body["warnings"])


def test_effective_argline_renders_declared_layers_only(tmp_path, monkeypatch):
    """Design decision 3 (Plan C2, Task 7): the argline — and, by extension,
    anything ever SHIPPED to an engine — renders DECLARED layers (engine/
    model/engine_model) only. A harvested engine_defaults value that would
    otherwise render as a flag stays out of the argline even though it
    fully resolves and keeps its provenance in 'resolved'."""
    from app.characteristics import CharacteristicsStore
    from app.harvest import parse_probe_output
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)

    probe_output = json.dumps({"options": [
        {"flags": ["--tokenizer-mode"], "type": "str",
         "choices": ["auto", "slow", "mistral", "custom"],
         "default": repr("auto"), "nargs": None, "cls": "_StoreAction",
         "help": "Tokenizer mode."},
    ]})
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {
        "option_catalog": parse_probe_output(probe_output, engine_version="test", now="t"),
    })
    deck["characteristics_store"] = characteristics

    store = SettingsStore(tmp_path / "s.json")
    store.put("engine_models", "sparky/vllm|TestModel-7B", "args",
              {"max-model-len": "131072"})
    deck["settings_store"] = store

    body = TestClient(app).get(
        "/api/settings/effective/sparky/vllm/TestModel-7B").json()

    # 'resolved' keeps full provenance for both layers.
    assert body["resolved"]["tokenizer-mode"]["origin"] == "derived"
    assert body["resolved"]["tokenizer-mode"]["layer"] == "engine_defaults"
    assert body["resolved"]["max-model-len"]["origin"] == "declared"

    # The argline carries the declared flag and NOT the derived one.
    assert "--max-model-len 131072" in body["argline"]
    assert "--tokenizer-mode" not in body["argline"]


def test_effective_layers_filter(tmp_path, monkeypatch):
    """?layers=engine_model narrows the structured 'resolved' view to just
    that layer's keys; the argline is unaffected (still declared-only,
    across every declared layer, regardless of the filter). An unknown
    layer name is a 422, matching this router's other ValueError -> 422
    save-time rejections."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"engine-flag": "1"})
    store.put("models", "TestModel-7B", "args", {"model-flag": "1"})
    store.put("engine_models", "sparky/vllm|TestModel-7B", "args",
              {"engine-model-flag": "1"})
    deck["settings_store"] = store
    client = TestClient(app)

    body = client.get("/api/settings/effective/sparky/vllm/TestModel-7B",
                      params={"layers": "engine_model"}).json()

    assert set(body["resolved"]) == {"engine-model-flag"}
    assert "--engine-flag 1" in body["argline"]
    assert "--model-flag 1" in body["argline"]
    assert "--engine-model-flag 1" in body["argline"]

    multi = client.get("/api/settings/effective/sparky/vllm/TestModel-7B",
                       params={"layers": "engine,model"}).json()
    assert set(multi["resolved"]) == {"engine-flag", "model-flag"}

    resp = client.get("/api/settings/effective/sparky/vllm/TestModel-7B",
                      params={"layers": "bogus"})
    assert resp.status_code == 422


def test_unset_round_trip_none_survives_to_the_ladder(tmp_path, monkeypatch):
    """PUT {'max-model-len': None} at the most specific layer unsets an
    inherited value all the way through: the store's normalize_args_map
    (verified 2026-08-07: None passes through both normalization axes
    unchanged), the ladder's None-pop (app.ladder L73-76), the structured
    'resolved' view, and the argline all agree the key is gone."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"max-model-len": "8192"})
    deck["settings_store"] = store
    client = TestClient(app)

    before = client.get("/api/settings/effective/sparky/vllm/TestModel-7B").json()
    assert "max-model-len" in before["resolved"]
    assert "--max-model-len 8192" in before["argline"]

    put_resp = client.put(
        "/api/settings/engine_models/sparky/vllm|TestModel-7B",
        json={"namespace": "args", "values": {"max-model-len": None}})
    assert put_resp.status_code == 200
    assert put_resp.json()["args"]["max-model-len"] is None

    after = client.get("/api/settings/effective/sparky/vllm/TestModel-7B").json()
    assert "max-model-len" not in after["resolved"]
    assert "--max-model-len" not in after["argline"]


def test_preview_parses_text_and_returns_warnings(tmp_path, monkeypatch):
    """The text field's live feedback — parse without saving."""
    app, deck = make_app(tmp_path, monkeypatch)

    body = TestClient(app).post("/api/settings/preview",
                                json={"argline": "--max-model-len 262144 --weird"}).json()

    assert body["parsed"]["max-model-len"] == "262144"
    assert body["parsed"]["weird"] is True
    assert any(w["key"] == "weird" for w in body["warnings"])


def test_preview_preserves_an_unknown_token(tmp_path, monkeypatch):
    """The unacceptable failure mode is dropping something a human typed."""
    app, deck = make_app(tmp_path, monkeypatch)

    body = TestClient(app).post("/api/settings/preview",
                                json={"argline": "--totally-made-up xyz"}).json()

    assert body["parsed"]["totally-made-up"] == "xyz"


def test_preview_renders_args_map(tmp_path, monkeypatch):
    """The render direction: args map -> argline in the response."""
    app, deck = make_app(tmp_path, monkeypatch)
    res = TestClient(app).post("/api/settings/preview",
                               json={"args": {"max-model-len": 131072}})
    assert res.status_code == 200
    body = res.json()
    assert body["argline"] == "--max-model-len 131072"   # match render_argline's real join
    assert body["parsed"] == {"max-model-len": "131072"} # normalize: int -> str


def test_preview_requires_exactly_one_direction(tmp_path, monkeypatch):
    """Must accept exactly one of argline or args, not both or neither."""
    app, deck = make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    # Neither argline nor args
    assert client.post("/api/settings/preview", json={}).status_code == 422

    # Both argline and args
    assert client.post("/api/settings/preview",
                       json={"argline": "--seed 1", "args": {"seed": "1"}}
                       ).status_code == 422


def test_preview_parses_also_returns_argline(tmp_path, monkeypatch):
    """Existing parse direction now also returns argline."""
    app, deck = make_app(tmp_path, monkeypatch)

    body = TestClient(app).post("/api/settings/preview",
                                json={"argline": "--max-model-len 262144"}).json()

    assert body["parsed"]["max-model-len"] == "262144"
    assert body["argline"] == "--max-model-len 262144"


def test_catalog_absent_is_null_not_an_error(tmp_path, monkeypatch):
    """An engine that has never run has no catalog. Supported state."""
    app, deck = make_app(tmp_path, monkeypatch)

    resp = TestClient(app).get("/api/settings/catalog/sparky/vllm")

    assert resp.status_code == 200
    assert resp.json() is None


# ===========================================================================
# Manual force-harvest (Task 3, C2/Phase 3): POST /api/settings/harvest/...
# and GET /api/settings/catalog/... gaining "harvested_ts". make_app()'s
# deck.update() (see its docstring) never touches characteristics_store,
# engine_exec, or configurable_engines, so every test below sets exactly the
# deck entries it needs — same pattern as the /effective real-catalog tests
# above.
# ===========================================================================


def test_get_catalog_carries_harvested_ts_from_the_derived_ts_field(tmp_path, monkeypatch):
    """Additive field on GET /settings/catalog: harvested_ts is the
    characteristics field's own derived_ts, not a new timestamp — the All-
    options screen's "as of" display reads this, not option_catalog's
    internals."""
    from app.characteristics import CharacteristicsStore
    from app.harvest import parse_probe_output
    from tests.test_harvest import PROBE_OUTPUT

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {
        "option_catalog": parse_probe_output(
            PROBE_OUTPUT, engine_version="0.26.0", now="2026-08-07T09:00:00+00:00"),
    })
    deck["characteristics_store"] = characteristics

    body = TestClient(app).get("/api/settings/catalog/sparky/vllm").json()

    assert body["harvested_ts"] == "2026-08-07T09:00:00+00:00"
    # ...and validate_settings' consumption (app/validate_settings.py:81,
    # .get("options") only) is untouched — the additive field sits alongside
    # the existing shape, not instead of it.
    assert "options" in body


def test_get_catalog_decodes_reprd_defaults_for_the_wire(tmp_path, monkeypatch):
    """F2 follow-up (task-9 review, Critical): app.harvest stores each
    option's default as ``repr(action.default)`` (module docstring,
    "Engine-default decoding") — the probe's raw truth, not a value fit to
    hand a consumer as-is. GET /settings/catalog now decodes it the same
    way _resolve's engine_defaults layer already does
    (_decode_harvested_default), so a string-typed option like --dtype
    reads back "auto", not the repr "'auto'" quotes and all, a real default
    like "131072" reads back as the JSON number 131072, and every
    nothing-to-prefill shape (the "None" repr, and _decode_harvested_
    default's drop shapes False/[]/{}) comes back as JSON null instead of
    surviving as text a UI would have to re-parse.

    Extended for F3 (Important, final branch review 2026-08-07) with the two
    container shapes that decode to something `decoded in ([], {})` cannot
    match: an empty set/tuple must reach the wire as null like every other
    nothing-to-prefill shape, and a non-empty one as a list."""
    from app.characteristics import CharacteristicsStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {
        "option_catalog": {
            "value": {
                "engine_version": "sha256:test",
                "options": {
                    "dtype": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "'auto'", "nargs": None, "repeatable": False,
                        "help": "Data type.", "widget": "text",
                    },
                    "max-model-len": {
                        "aliases": [], "type": "int", "choices": None,
                        "default": "None", "nargs": None, "repeatable": False,
                        "help": "Model context length.", "widget": "number",
                    },
                    "middleware": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "[]", "nargs": None, "repeatable": True,
                        "help": "ASGI middleware.", "widget": "list",
                    },
                    "block-size": {
                        "aliases": [], "type": "int", "choices": None,
                        "default": "131072", "nargs": None, "repeatable": False,
                        "help": "Block size.", "widget": "number",
                    },
                    "enable-prefix-caching": {
                        "aliases": [], "type": None, "choices": None,
                        "default": "True", "nargs": 0, "repeatable": False,
                        "help": "Enable prefix caching.", "widget": "toggle",
                    },
                    # F3, Important (final branch review 2026-08-07): the two
                    # container shapes literal_eval produces that
                    # _decode_harvested_default's `decoded in ([], {})` drop
                    # test cannot match. `set()` is not hypothetical — it is
                    # the live raw default of sparky/vllm's
                    # --cpu-offload-params and --offload-params (2 of 274).
                    "cpu-offload-params": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "set()", "nargs": None, "repeatable": True,
                        "help": "Params to offload to CPU.", "widget": "list",
                    },
                    "offload-tags": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "{'weights', 'kv'}", "nargs": None,
                        "repeatable": True, "help": "Offload tags.",
                        "widget": "list",
                    },
                    "trust-hosts": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "('localhost', '127.0.0.1')", "nargs": None,
                        "repeatable": True, "help": "Trusted hosts.",
                        "widget": "list",
                    },
                },
            },
            "source": "argparse introspection",
            "derived_ts": "2026-08-08T00:00:00+00:00",
        },
    })
    deck["characteristics_store"] = characteristics

    options = TestClient(app).get("/api/settings/catalog/sparky/vllm").json()["options"]

    assert options["dtype"]["default"] == "auto"            # "'auto'" -> "auto"
    assert options["max-model-len"]["default"] is None      # "None" repr -> null
    assert options["middleware"]["default"] is None         # "[]" repr -> null (drop shape)
    assert options["block-size"]["default"] == 131072        # "131072" -> real int
    assert options["enable-prefix-caching"]["default"] is True  # "True" -> real bool

    # An EMPTY set is the same "nothing to prefill" answer as [] and {} — not
    # the `[]` FastAPI would otherwise serialize it as, which reads to the UI
    # as a real (empty-list) default and violates CatalogOption.default's
    # documented null contract.
    assert options["cpu-offload-params"]["default"] is None
    # A non-empty set/tuple IS a real default, and becomes the one JSON shape
    # ArgValue has for many values: a list. Sets are sorted (no order of their
    # own — hash order would make the wire value unstable); a tuple's own
    # order is kept.
    assert options["offload-tags"]["default"] == ["kv", "weights"]
    assert options["trust-hosts"]["default"] == ["localhost", "127.0.0.1"]

    # Fields other than `default` pass through unmodified.
    assert options["dtype"]["type"] == "str"
    assert options["block-size"]["widget"] == "number"


def test_get_catalog_decode_does_not_touch_the_cached_repr(tmp_path, monkeypatch):
    """The decode in the finding above is a RESPONSE-shape transform only —
    the characteristics store's cached option_catalog keeps the raw repr,
    and _resolve/get_effective's engine_defaults layer (which reads that
    cache directly, not through get_catalog) is unaffected."""
    from app.characteristics import CharacteristicsStore

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {
        "option_catalog": {
            "value": {
                "engine_version": "sha256:test",
                "options": {
                    "dtype": {
                        "aliases": [], "type": "str", "choices": None,
                        "default": "'auto'", "nargs": None, "repeatable": False,
                        "help": "Data type.", "widget": "text",
                    },
                },
            },
            "source": "argparse introspection",
            "derived_ts": "2026-08-08T00:00:00+00:00",
        },
    })
    deck["characteristics_store"] = characteristics

    TestClient(app).get("/api/settings/catalog/sparky/vllm")

    cached = characteristics.entry("engine/sparky/vllm")["option_catalog"]["value"]
    assert cached["options"]["dtype"]["default"] == "'auto'"  # untouched raw repr

    # get_effective's engine_defaults layer decodes dtype to the real
    # "auto" string via _resolve's own _decode_harvested_default call —
    # same outcome, independent code path, per the ruling that _resolve is
    # not touched by this fix.
    resp = TestClient(app).get("/api/settings/effective/sparky/vllm/")
    assert resp.json()["resolved"]["dtype"]["value"] == "auto"


def test_harvest_now_rejects_an_unconfigurable_pair(tmp_path, monkeypatch):
    """A (node, engine) pair not in deck["configurable_engines"] is a
    ValueError -> 422, same family as this router's other save-time
    rejections (module docstring) — matched even with engine_exec wired,
    proving the pair check runs first."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["configurable_engines"] = [("sparky", "vllm")]
    deck["engine_exec"] = lambda node, engine, interpreter, source: ("v", "o")

    resp = TestClient(app).post("/api/settings/harvest/local/hipfire")

    assert resp.status_code == 422


def test_harvest_now_503s_when_no_engine_exec_is_wired(tmp_path, monkeypatch):
    """A known pair with no engine_exec configured (no spark on this box,
    see app.main._build_watcher) is a 503, not a 422 — the pair itself is
    valid, the wiring to actually harvest it just isn't there."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["configurable_engines"] = [("sparky", "vllm")]
    deck["engine_exec"] = None

    resp = TestClient(app).post("/api/settings/harvest/sparky/vllm")

    assert resp.status_code == 503


def test_harvest_now_forces_a_fresh_harvest_and_returns_the_outcome(tmp_path, monkeypatch):
    """Happy path, end to end through the router: a cached catalog whose
    engine_version equals engine_exec's peeked version would report
    "current" via the watcher's own (force=False) path — the manual route
    must force past that gate, write the fresh catalog, and echo
    harvest_catalog_pair's result dict back to the caller."""
    from app.characteristics import CharacteristicsStore
    from tests.test_harvest import PROBE_OUTPUT

    app, deck = make_app(tmp_path, monkeypatch)
    characteristics = CharacteristicsStore(tmp_path / "c.json")
    characteristics.put_fields("engine/sparky/vllm", {"option_catalog": {
        "value": {"engine_version": "0.26.0", "options": {}},
        "source": "argparse introspection", "derived_ts": "t0",
    }})
    deck["characteristics_store"] = characteristics
    deck["configurable_engines"] = [("sparky", "vllm")]

    class _FakeExec:
        version = "0.26.0"

        def __call__(self, node, engine, interpreter, source):
            return "0.26.0", PROBE_OUTPUT

    deck["engine_exec"] = _FakeExec()

    resp = TestClient(app).post("/api/settings/harvest/sparky/vllm")

    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "harvested"
    assert body["options"] > 0
    assert (characteristics.entry("engine/sparky/vllm")["option_catalog"]["value"]["engine_version"]
            == "0.26.0")


def test_settings_change_flags_drift_on_a_loaded_placement(tmp_path, monkeypatch):
    """``changed`` entries are namespace-qualified ("env:KEY", not a bare
    "KEY") — review-round RULING: an unqualified key would make
    args:max-model-len and env:MAX_MODEL_LEN indistinguishable and let
    same-named keys in different namespaces dedupe into one."""
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire")
    deck["intent_store"] = intent
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    client.put("/api/settings/engines/local/hipfire",
               json={"namespace": "env", "values": {"HIPFIRE_MAX_SEQ": "131072"}})

    entry = client.get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["settings_drift"]["changed"] == ["env:HIPFIRE_MAX_SEQ"]


def test_settings_drift_baseline_is_intent_updated_ts_not_last_healthy_ts(tmp_path, monkeypatch):
    """CRITICAL fix, review round 2026-08-07: app.arbiter.Watcher._reconcile_pass
    calls note_healthy(key) on every tick a placement is observed serving,
    and note_healthy unconditionally re-stamps last_healthy_ts to now — so
    comparing drift against last_healthy_ts made the flag self-erase within
    one arbiter tick of a placement actually serving (drift visible only
    while NOT serving — backwards). intent["updated_ts"] is the stable
    baseline: IntentStore.record() stamps it on every deliberate
    load/unload/park and note_healthy never touches it. This is the "after"
    side of the boundary: settings written strictly after the intent's
    updated_ts ARE drift, with a real (not None) baseline in play."""
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    intent = IntentStore(tmp_path / "intent.json")
    # A fixed PAST baseline — settings written "now" (real wall clock, via
    # the PUT below) are unambiguously after it.
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire",
                  now="2020-01-01T00:00:00+00:00")
    deck["intent_store"] = intent
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    client.put("/api/settings/engines/local/hipfire",
               json={"namespace": "env", "values": {"HIPFIRE_MAX_SEQ": "131072"}})

    entry = client.get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["settings_drift"]["changed"] == ["env:HIPFIRE_MAX_SEQ"]


def test_settings_drift_survives_repeated_note_healthy_reconcile_ticks(tmp_path, monkeypatch):
    """The most direct proof of the Finding-1 fix: simulates the exact
    sequence that self-erased the flag under the old (last_healthy_ts)
    comparison — a placement already healthy for a while (note_healthy
    called before the edit), a settings edit, then several MORE note_healthy
    calls (what app.arbiter.Watcher._reconcile_pass does on every tick it
    observes the placement serving, BEFORE plan_reconcile ever runs). Under
    the old baseline this would push last_healthy_ts past the settings
    write and erase the flag; intent["updated_ts"] is untouched by
    note_healthy, so drift must still be present after all of it."""
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire",
                  now="2020-01-01T00:00:00+00:00")
    intent.note_healthy("local/hipfire", now="2021-01-01T00:00:00+00:00")
    deck["intent_store"] = intent
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    client.put("/api/settings/engines/local/hipfire",
               json={"namespace": "env", "values": {"HIPFIRE_MAX_SEQ": "131072"}})

    # Several more simulated reconcile ticks observing it still serving.
    intent.note_healthy("local/hipfire")
    intent.note_healthy("local/hipfire")

    entry = client.get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["settings_drift"]["changed"] == ["env:HIPFIRE_MAX_SEQ"]


def test_settings_drift_is_none_when_settings_predate_the_intent_baseline(tmp_path, monkeypatch):
    """The other side of the boundary: settings written BEFORE intent's
    updated_ts are what the placement is already running with, not drift."""
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = SettingsStore(tmp_path / "s.json")
    # Written at real "now" — earlier than the future baseline below.
    store.put("engines", "local/hipfire", "env", {"HIPFIRE_MAX_SEQ": "131072"})
    deck["settings_store"] = store

    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire",
                  now="2030-01-01T00:00:00+00:00")
    deck["intent_store"] = intent

    entry = TestClient(app).get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["settings_drift"] is None


def test_settings_drift_is_none_with_no_intent_recorded(tmp_path, monkeypatch):
    """No intent at all means nothing is running deliberately, so there is
    nothing a settings write could be drift 'since' — None, not a guess."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "local/hipfire", "env", {"HIPFIRE_MAX_SEQ": "131072"})
    deck["settings_store"] = store

    entry = TestClient(app).get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["settings_drift"] is None


def test_settings_drift_never_triggers_a_restart(tmp_path, monkeypatch):
    """Distinct from placement drift, which DOES auto-restore. Conflating
    the two would restart a serving model because someone typed in a box.

    Review-round strengthening: the original version of this test hand-built
    statuses/intents and called plan_reconcile directly without ever writing
    a setting or building the lifecycle view — it would have passed
    unchanged even if build_lifecycle_view fed settings_drift straight into
    the reconciler. This version does a REAL settings write against a REAL,
    serving placement (drift genuinely fires through build_lifecycle_view),
    then builds the reconciler's own status input the way
    app.arbiter.Watcher._reconcile_pass builds it — derive_status over
    intent x observation, never from build_lifecycle_view — and proves that
    input carries no settings_drift key at all (the view and the
    reconciler's input are different objects) and that plan_reconcile still
    plans nothing.
    """
    from app.intent import IntentStore
    from app.lifecycle import derive_status
    from app.reconcile import plan_reconcile
    from app.routers import build_observations, build_world_snapshot
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    intent = IntentStore(tmp_path / "intent.json")
    intent.record("local/hipfire", state="loaded", model=None, engine="hipfire",
                  now="2020-01-01T00:00:00+00:00")
    deck["intent_store"] = intent
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    client.put("/api/settings/engines/local/hipfire",
               json={"namespace": "env", "values": {"HIPFIRE_MAX_SEQ": "131072"}})

    # Drift genuinely fires through the real lifecycle view (FakeHipfire
    # defaults to state="running" -> observed loaded, matching intent's
    # model=None -> status "serving").
    entry = client.get("/api/state").json()["lifecycle"]["local/hipfire"]
    assert entry["status"] == "serving"
    assert entry["settings_drift"] is not None

    # The reconciler's OWN inputs, built exactly the way
    # app.arbiter.Watcher._reconcile_pass builds them — never from
    # build_lifecycle_view, which is a read-only display derivation with no
    # relationship to what the reconciler consumes.
    world = build_world_snapshot(deck)
    observed = build_observations(deck, world)
    intents = intent.get()
    statuses = {key: derive_status(intents.get(key), obs) for key, obs in observed.items()}

    assert "settings_drift" not in statuses["local/hipfire"]
    assert plan_reconcile(statuses, intents, auto_enabled=True) == []


# ===========================================================================
# Settings drift speaks the spark vocabulary (Plan C2, Task 6) — unit tests
# of _settings_drift's identity_map translation, called directly (rather
# than through the API) so the profile/identity boundary is exercised
# precisely, the same way Task 5's adopt tests below exercise
# CharacteristicsStore directly. settings_data/intent are still built with
# the real stores, matching every other test in this module.
# ===========================================================================


def test_spark_drift_translates_profile_to_identity(tmp_path):
    """intent {engine: 'spark', model: 'heretic', updated_ts: T0};
    identity_map {'heretic': {'identity': 'Qwen3.6-35B-A3B-heretic-NVFP4', ...}};
    settings_data with engine_models 'sparky/vllm|Qwen3.6-35B-A3B-heretic-NVFP4'
    args updated_ts AFTER T0 -> _settings_drift(..., 'sparky/slot0', intent,
    identity_map=map) returns changed == ['args:max-model-len'] (qualified,
    per T7's C1 ruling). This is the 5th vocabulary-bug instance, caught at
    plan time: intent records the deck adapter ('spark') and the PROFILE
    ('heretic'), but the settings PUT lands under the real engine ('vllm')
    and the checkpoint identity — the exact D11 live-drill flow."""
    from app.intent import IntentStore
    from app.routers import _settings_drift
    from app.settings_store import SettingsStore

    intent_store = IntentStore(tmp_path / "intent.json")
    # Fixed PAST baseline (T0) — the settings PUT below lands at real "now",
    # unambiguously after it (same idiom as C1's baseline tests above).
    intent_store.record("sparky/slot0", state="loaded", model="heretic",
                        engine="spark", now="2020-01-01T00:00:00+00:00")
    intent = intent_store.get()["sparky/slot0"]

    identity_map = {"heretic": {"identity": "Qwen3.6-35B-A3B-heretic-NVFP4",
                                "service": "aeon-vllm",
                                "container_name": "aeon-vllm"}}

    settings = SettingsStore(tmp_path / "settings.json")
    settings.put("engine_models", "sparky/vllm|Qwen3.6-35B-A3B-heretic-NVFP4",
                "args", {"max-model-len": "131072"})

    result = _settings_drift(settings.get(), "sparky/slot0", intent,
                             identity_map=identity_map)

    assert result is not None
    assert result["changed"] == ["args:max-model-len"]


def test_spark_drift_without_map_entry_falls_back_to_old_scopes(tmp_path):
    """No identity_map (or profile absent from it) -> behaves exactly as
    C1 shipped (scopes from intent verbatim); returns None here. The write
    below lands under the real vllm/identity vocabulary, which is invisible
    to intent's verbatim scopes ('sparky/spark|heretic') — proving the
    translation is opt-in per call, not a silent behavior change."""
    from app.intent import IntentStore
    from app.routers import _settings_drift
    from app.settings_store import SettingsStore

    intent_store = IntentStore(tmp_path / "intent.json")
    intent_store.record("sparky/slot0", state="loaded", model="heretic",
                        engine="spark", now="2020-01-01T00:00:00+00:00")
    intent = intent_store.get()["sparky/slot0"]

    settings = SettingsStore(tmp_path / "settings.json")
    settings.put("engine_models", "sparky/vllm|Qwen3.6-35B-A3B-heretic-NVFP4",
                "args", {"max-model-len": "131072"})
    settings_data = settings.get()

    # No map at all — the C1 3-arg call site.
    assert _settings_drift(settings_data, "sparky/slot0", intent) is None

    # Map present but the profile never made it in (e.g. never adopted) —
    # same verbatim fallback as no map at all.
    other_map = {"other-profile": {"identity": "some-other-identity",
                                   "service": "x", "container_name": "x"}}
    assert _settings_drift(settings_data, "sparky/slot0", intent,
                           identity_map=other_map) is None


def test_local_keys_are_untouched_by_the_map(tmp_path):
    """A local/hipfire key with a populated identity_map still resolves
    scopes from intent verbatim, because the model this intent recorded
    (None) is not in the map — _settings_drift's own gate is now
    identity_map-presence + model membership (N1 T7), not a key comparison;
    the guarantee that a LOCAL key's call never even receives a populated
    map lives one level up, in build_lifecycle_view, which only resolves an
    identity_map for a swap node's own slot key (keyed by that node's id).
    This direct call bypasses that caller-side gate on purpose, to prove
    the map alone does not leak translation into an unrelated model."""
    from app.intent import IntentStore
    from app.routers import _settings_drift
    from app.settings_store import SettingsStore

    intent_store = IntentStore(tmp_path / "intent.json")
    intent_store.record("local/hipfire", state="loaded", model=None,
                        engine="hipfire", now="2020-01-01T00:00:00+00:00")
    intent = intent_store.get()["local/hipfire"]

    settings = SettingsStore(tmp_path / "settings.json")
    settings.put("engines", "local/hipfire", "env", {"HIPFIRE_MAX_SEQ": "131072"})

    identity_map = {"heretic": {"identity": "Qwen3.6-35B-A3B-heretic-NVFP4",
                                "service": "aeon-vllm",
                                "container_name": "aeon-vllm"}}

    result = _settings_drift(settings.get(), "local/hipfire", intent,
                             identity_map=identity_map)

    assert result is not None
    assert result["changed"] == ["env:HIPFIRE_MAX_SEQ"]


def test_engines_scope_translates_too(tmp_path):
    """A change under engines 'sparky/vllm' (no model half) also flags —
    the translation touches engine_key (used by the 'engines' scope on its
    own), not only the model half used by 'models'/'engine_models'."""
    from app.intent import IntentStore
    from app.routers import _settings_drift
    from app.settings_store import SettingsStore

    intent_store = IntentStore(tmp_path / "intent.json")
    intent_store.record("sparky/slot0", state="loaded", model="heretic",
                        engine="spark", now="2020-01-01T00:00:00+00:00")
    intent = intent_store.get()["sparky/slot0"]

    identity_map = {"heretic": {"identity": "Qwen3.6-35B-A3B-heretic-NVFP4",
                                "service": "aeon-vllm",
                                "container_name": "aeon-vllm"}}

    settings = SettingsStore(tmp_path / "settings.json")
    settings.put("engines", "sparky/vllm", "env",
                {"VLLM_USE_FLASHINFER_SAMPLER": "1"})

    result = _settings_drift(settings.get(), "sparky/slot0", intent,
                             identity_map=identity_map)

    assert result is not None
    assert result["changed"] == ["env:VLLM_USE_FLASHINFER_SAMPLER"]


# ===========================================================================
# Adopt sweep (Plan C2, Task 5) — compose import into settings scopes +
# the profile -> identity map.
# ===========================================================================

HERETIC_COMPOSE = (Path(__file__).parent / "fixtures" / "spark-profiles"
                   / "compose-heretic.yaml").read_text()
DS4_COMPOSE = (Path(__file__).parent / "fixtures" / "spark-profiles"
               / "compose-ds4.yaml").read_text()
MM27B_COMPOSE = (Path(__file__).parent / "fixtures" / "spark-profiles"
                 / "compose-mm27b.yaml").read_text()


class FakeSparkForAdopt:
    """Just enough of SparkClient for the adopt route: status() for the
    profile list (mirrors FakeSpark above) and get_compose()
    serving the two real fixture files straight off disk — real compose
    text, not a hand-shaped stand-in, is the whole point of these tests."""

    def __init__(self):
        self.compose = {"heretic": HERETIC_COMPOSE, "ds4": DS4_COMPOSE}

    def status(self):
        return {"profiles": [
            {"name": "heretic", "engine": "vllm", "health_url": None, "container": None},
            {"name": "ds4", "engine": "ds4",
             "health_url": "http://127.0.0.1:8000/metrics", "container": "spark-ds4"},
        ], "swap_status": None, "serving": None}

    def get_compose(self, profile):
        return self.compose[profile]


def _adopt_app(tmp_path, monkeypatch, spark="default"):
    """The adopt route needs
    settings_store AND characteristics_store pointed at tmp_path — the
    default deck's copies point at the container's /data, which does not
    exist under test.

    Node id is "boxa" (N1 generalized the adopt gate off the single spark
    node — [[defaults-that-hide-bugs]], a fixture id of "sparky" could not
    have caught a handler that still secretly resolved the old single
    pair). ``spark=None`` declares boxa control:"swap" (all three
    prerequisites present, so the registry gate passes) but wires an EMPTY
    FakeNodeClients — no client bound for boxa — so client_for(boxa)
    answers None and the route 503s the same "known node, not operable" way
    every other swap node does; a real NodeClients here would instead build
    a genuine networked SparkClient (every prerequisite IS present) and the
    sweep's first network call would 502, not 503."""
    from app.characteristics import CharacteristicsStore
    from app.node_clients import NodeObservers
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    if spark is None:
        deck["node_store"].add(
            {"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
             "address": "http://boxa:7720", "serving_address": "http://boxa:8000",
             "control": "swap"}, credential="key-boxa")
        deck["node_clients"] = FakeNodeClients()
        deck["node_observers"] = NodeObservers(deck["node_store"], deck["node_clients"])
    else:
        wire_swap_node(deck, "boxa",
                       FakeSparkForAdopt() if spark == "default" else spark,
                       label="Box Alpha")
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    deck["characteristics_store"] = CharacteristicsStore(tmp_path / "c.json")
    return app, deck


def test_adopt_imports_vllm_profiles_into_engine_models_scope(tmp_path, monkeypatch):
    """POST /api/settings/adopt/boxa/vllm ->
    engine_models scope 'boxa/vllm|Qwen3.6-35B-A3B-heretic-NVFP4' has the
    imported args (max-model-len 262144, _positional [serve, /model]) and
    the modelopt note; response lists it under 'adopted'."""
    from app.argline import POSITIONAL_KEY

    app, deck = _adopt_app(tmp_path, monkeypatch)
    key = "boxa/vllm|Qwen3.6-35B-A3B-heretic-NVFP4"

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    assert resp.json()["adopted"] == [key]

    scope = deck["settings_store"].scope("engine_models", key)
    assert scope["args"]["max-model-len"] == "262144"
    assert scope["args"][POSITIONAL_KEY] == ["serve", "/model"]
    assert "modelopt" in scope["notes"]["args"]


def test_adopt_skips_non_vllm_profiles(tmp_path, monkeypatch):
    """ds4 appears in 'skipped' with its engine named; no ds4 scope is
    written."""
    app, deck = _adopt_app(tmp_path, monkeypatch)

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    body = resp.json()
    assert {"profile": "ds4", "engine": "ds4"} in body["skipped"]

    all_engine_models = deck["settings_store"].get()["engine_models"]
    assert not any("ds4" in key for key in all_engine_models)


def test_adopt_never_clobbers_an_existing_scope(tmp_path, monkeypatch):
    """Pre-seed the heretic identity scope with one arg; adopt again ->
    response lists it under 'kept' and the pre-seeded value survives."""
    app, deck = _adopt_app(tmp_path, monkeypatch)
    key = "boxa/vllm|Qwen3.6-35B-A3B-heretic-NVFP4"
    deck["settings_store"].put("engine_models", key, "args", {"max-model-len": "999"})

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    body = resp.json()
    assert body["kept"] == [key]
    assert key not in body["adopted"]
    assert deck["settings_store"].scope("engine_models", key)["args"] == {
        "max-model-len": "999"
    }


def test_adopt_records_the_identity_map(tmp_path, monkeypatch):
    """characteristics entry engine/boxa/vllm gains profile_identities
    with heretic -> {identity, service='aeon-vllm',
    container_name='aeon-vllm'}, carrying value/source/derived_ts."""
    app, deck = _adopt_app(tmp_path, monkeypatch)

    TestClient(app).post("/api/settings/adopt/boxa/vllm")

    field = deck["characteristics_store"].entry("engine/boxa/vllm")["profile_identities"]
    assert set(field) == {"value", "source", "derived_ts"}
    assert field["value"]["heretic"] == {
        "identity": "Qwen3.6-35B-A3B-heretic-NVFP4",
        "service": "aeon-vllm",
        "container_name": "aeon-vllm",
    }


def test_adopt_on_unknown_node_or_engine_is_422(tmp_path, monkeypatch):
    """POST /api/settings/adopt/local/vllm and /boxa/spark -> 422: local is
    control:"none" (not a swap node) and spark is the wrong engine — only a
    swap node's vllm is adoptable (N1 generalized this off the single
    (the legacy sparky node's id, 'vllm') pair C2 had)."""
    app, deck = _adopt_app(tmp_path, monkeypatch)
    client = TestClient(app)

    assert client.post("/api/settings/adopt/local/vllm").status_code == 422
    assert client.post("/api/settings/adopt/boxa/spark").status_code == 422


def test_adopt_any_swap_node(tmp_path, monkeypatch):
    """(boxa, vllm) is adoptable once boxa is control:"swap" — the C2
    single-pair gate generalizes to the declaration, not a hardcoded id."""
    app, deck = _adopt_app(tmp_path, monkeypatch)
    key = "boxa/vllm|Qwen3.6-35B-A3B-heretic-NVFP4"

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    assert resp.json()["adopted"] == [key]


def test_adopt_control_none_node_422(tmp_path, monkeypatch):
    """A node with a serving_address but control:"none" is NOT adoptable —
    declared, never inferred."""
    app, deck = make_app(tmp_path, monkeypatch)
    deck["node_store"].add(
        {"id": "boxc", "label": "Box Gamma", "agent_kind": "node-agent",
         "address": "http://boxc:7720", "serving_address": "http://boxc:8000",
         "control": "none"}, credential="key-boxc")

    resp = TestClient(app).post("/api/settings/adopt/boxc/vllm")

    assert resp.status_code == 422


def test_adopt_with_no_spark_configured_is_503(tmp_path, monkeypatch):
    """A declared control:"swap" node with no operable client bound -> 503,
    the same "known node, not operable" answer every swap node gets."""
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=None)

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 503


class FakeSparkForAdoptWithFailure:
    """Two real vllm profiles (heretic, mm27b) plus ds4 — lets a test swap
    in a bad compose string or an exception for exactly ONE profile's
    get_compose(), to prove the sweep isolates a single bad profile rather
    than failing the whole request (review round 1 fix, Task 5)."""

    def __init__(self, compose_overrides=None, raise_for=None):
        self.compose = {"heretic": HERETIC_COMPOSE, "mm27b": MM27B_COMPOSE,
                        "ds4": DS4_COMPOSE}
        self.compose.update(compose_overrides or {})
        self._raise_for = raise_for or {}  # {profile: exception instance}

    def status(self):
        return {"profiles": [
            {"name": "heretic", "engine": "vllm", "health_url": None, "container": None},
            {"name": "mm27b", "engine": "vllm", "health_url": None, "container": None},
            {"name": "ds4", "engine": "ds4",
             "health_url": "http://127.0.0.1:8000/metrics", "container": "spark-ds4"},
        ], "swap_status": None, "serving": None}

    def get_compose(self, profile):
        if profile in self._raise_for:
            raise self._raise_for[profile]
        return self.compose[profile]


def test_adopt_isolates_a_malformed_profile_and_continues_the_sweep(tmp_path, monkeypatch):
    """One profile's compose is malformed YAML (import_compose raises
    ValueError) -> it lands in 'skipped' with a reason naming the error;
    the OTHER real vllm profile still adopts, and the identity map still
    gets written for it. Review round 1 fix: a per-profile failure must not
    fail the whole request after earlier profiles' writes already
    committed, and must not suppress the profile_identities write for the
    profiles that DID import cleanly."""
    fake = FakeSparkForAdoptWithFailure(compose_overrides={"heretic": "services: [unclosed"})
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)
    good_key = "boxa/vllm|Qwen3.6-27B-AEON-MM-MTP"

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    body = resp.json()
    assert good_key in body["adopted"]
    bad_skip = next(s for s in body["skipped"] if s["profile"] == "heretic")
    assert bad_skip["engine"] == "vllm"
    assert "ValueError" in bad_skip["reason"]

    # The good profile's scope was actually written, not just reported.
    assert deck["settings_store"].scope("engine_models", good_key)["args"]

    # The identity map covers the good profile despite the other's failure;
    # the failed one never got far enough to have an identity to record.
    field = deck["characteristics_store"].entry("engine/boxa/vllm")["profile_identities"]
    assert "mm27b" in field["value"]
    assert "heretic" not in field["value"]


def test_adopt_isolates_a_compose_fetch_failure_and_continues_the_sweep(tmp_path, monkeypatch):
    """get_compose() itself raises EngineError (a node-agent transport/HTTP
    failure) for one profile -> same isolation as a malformed-YAML import
    failure: that profile lands in 'skipped' with a reason, and the sweep
    still completes for the rest."""
    from app.engines import EngineError

    fake = FakeSparkForAdoptWithFailure(raise_for={"heretic": EngineError("node down")})
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)
    good_key = "boxa/vllm|Qwen3.6-27B-AEON-MM-MTP"

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    body = resp.json()
    assert good_key in body["adopted"]
    bad_skip = next(s for s in body["skipped"] if s["profile"] == "heretic")
    assert bad_skip["engine"] == "vllm"
    assert "EngineError" in bad_skip["reason"]
    assert "node down" in bad_skip["reason"]


def test_adopt_isolates_an_unsupported_environment_shape(tmp_path, monkeypatch):
    """A profile whose `environment:` is neither mapping nor list is a
    ValueError (app.compose_import._import_env), so it isolates into
    'skipped' exactly like malformed YAML. The LIST form is not this case —
    it imports (tests/test_compose_import.py) — and before that fix it
    raised AttributeError, which escapes adopt's (ValueError, EngineError)
    catch and 500s the sweep after earlier profiles' writes committed."""
    junk = HERETIC_COMPOSE.replace(
        '    environment:\n      VLLM_USE_FLASHINFER_SAMPLER: "1"\n',
        "    environment: VLLM_LOGGING_LEVEL=DEBUG\n")
    assert "environment: VLLM" in junk, "fixture's environment block moved"
    fake = FakeSparkForAdoptWithFailure(compose_overrides={"heretic": junk})
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)

    resp = TestClient(app).post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    body = resp.json()
    assert "boxa/vllm|Qwen3.6-27B-AEON-MM-MTP" in body["adopted"]
    bad_skip = next(s for s in body["skipped"] if s["profile"] == "heretic")
    assert "ValueError" in bad_skip["reason"]


# --- re-adopt must not evict a previously-adopted profile (final review) ---


def test_readopt_keeps_the_identity_of_a_transiently_failing_profile(
        tmp_path, monkeypatch):
    """A profile that adopted cleanly once and whose fetch fails on a LATER
    sweep keeps its identity-map entry.

    put_fields REPLACES the field it is given, and `identities` is built
    from the current sweep alone — so a rebuild-from-scratch dropped the
    failing profile out of the map, which silently reverts its drift
    translation to verbatim scopes (drift goes dark, the failure Decision 9
    exists to prevent) and 409s its reload.
    """
    from app.engines import EngineError
    from app.observe import slot_key
    from app.routers import _settings_drift

    fake = FakeSparkForAdoptWithFailure()
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)
    client = TestClient(app)
    client.post("/api/settings/adopt/boxa/vllm")
    first = deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]
    assert "heretic" in first

    fake._raise_for = {"heretic": EngineError("node down")}
    resp = client.post("/api/settings/adopt/boxa/vllm")

    assert resp.status_code == 200
    assert any(s["profile"] == "heretic" for s in resp.json()["skipped"])
    field = deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]
    assert field["heretic"] == first["heretic"]
    assert "mm27b" in field

    # The consequence, not just the storage: drift for the swap node's slot
    # is only visible THROUGH this map (app/routers/__init__.py's slot
    # translation), so an evicted entry is drift going dark.
    key = slot_key("boxa")
    deck["intent_store"].record(key, state="loaded", model="heretic",
                                engine="spark", now="2020-01-01T00:00:00+00:00")
    intent = deck["intent_store"].get()[key]
    settings_data = deck["settings_store"].get()
    assert _settings_drift(settings_data, key, intent,
                           identity_map=field) is not None
    assert _settings_drift(settings_data, key, intent,
                           identity_map={k: v for k, v in field.items()
                                         if k != "heretic"}) is None


def test_readopt_drops_a_profile_that_left_the_node(tmp_path, monkeypatch):
    """Merging must not make the map immortal: a profile absent from
    status() entirely is gone from the node, so its entry is dropped —
    only a profile that IS still there but failed this sweep is kept."""
    fake = FakeSparkForAdoptWithFailure()
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)
    client = TestClient(app)
    client.post("/api/settings/adopt/boxa/vllm")
    assert "heretic" in deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]

    fake.status = lambda: {"profiles": [
        {"name": "mm27b", "engine": "vllm", "health_url": None, "container": None},
    ], "swap_status": None, "serving": None}
    client.post("/api/settings/adopt/boxa/vllm")

    field = deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]
    assert set(field) == {"mm27b"}


def test_readopt_with_every_profile_failing_keeps_the_whole_map(
        tmp_path, monkeypatch):
    """The `if identities:` write guard is not enough on its own: with the
    merge in place a sweep where EVERY still-present profile fails must
    leave the previously-adopted map intact, not merely unwritten."""
    from app.engines import EngineError

    fake = FakeSparkForAdoptWithFailure()
    app, deck = _adopt_app(tmp_path, monkeypatch, spark=fake)
    client = TestClient(app)
    client.post("/api/settings/adopt/boxa/vllm")
    before = deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]

    fake._raise_for = {"heretic": EngineError("down"), "mm27b": EngineError("down")}
    client.post("/api/settings/adopt/boxa/vllm")

    after = deck["characteristics_store"].entry(
        "engine/boxa/vllm")["profile_identities"]["value"]
    assert after == before


# --- Journal-driven settings drift (Task 2) ---


def test_drift_entries_carry_old_and_new():
    """The new entries key carries exact old->new changes from the journal."""
    from app.routers import _settings_drift

    BASE = "2026-08-07T09:00:00+00:00"
    LATER = "2026-08-07T10:00:00+00:00"

    def _scope(journal_entries, args):
        return {"args": args, "updated_ts": {"args": LATER},
                "journal": {"args": journal_entries}}

    data = {"engines": {"local/lemonade": _scope(
        [{"key": "max-model-len", "old": "262144", "new": "131072", "ts": LATER}],
        {"max-model-len": "131072"})}}
    intent = {"engine": "lemonade", "model": None, "updated_ts": BASE}
    drift = _settings_drift(data, "local/lemonade", intent)
    assert drift["changed"] == ["args:max-model-len"]
    assert drift["entries"] == [{"key": "args:max-model-len",
                                "old": "262144", "new": "131072", "ts": LATER}]


def test_drift_net_zero_change_is_no_drift():
    """A change that folds to old == new is honest no-drift."""
    from app.routers import _settings_drift

    BASE = "2026-08-07T09:00:00+00:00"
    LATER = "2026-08-07T10:00:00+00:00"

    def _scope(journal_entries, args):
        return {"args": args, "updated_ts": {"args": LATER},
                "journal": {"args": journal_entries}}

    data = {"engines": {"local/lemonade": _scope(
        [{"key": "seed", "old": "1", "new": "2", "ts": LATER},
         {"key": "seed", "old": "2", "new": "1", "ts": LATER}],
        {"seed": "1"})}}
    intent = {"engine": "lemonade", "model": None, "updated_ts": BASE}
    assert _settings_drift(data, "local/lemonade", intent) is None


def test_drift_without_journal_keeps_c1_shape():
    """Without a journal, drift keeps C1 shape: changed list, no entries."""
    from app.routers import _settings_drift

    BASE = "2026-08-07T09:00:00+00:00"
    LATER = "2026-08-07T10:00:00+00:00"

    data = {"engines": {"local/lemonade": {
        "args": {"seed": "1", "top-k": "40"}, "updated_ts": {"args": LATER}}}}
    intent = {"engine": "lemonade", "model": None, "updated_ts": BASE}
    drift = _settings_drift(data, "local/lemonade", intent)
    assert sorted(drift["changed"]) == ["args:seed", "args:top-k"]
    assert drift["entries"] == []


def test_unbuilt_mech_error_renders_501_not_502(tmp_path, monkeypatch):
    """502 says "the engine is broken" and sends an operator debugging the
    wrong side of the wire; an unbuilt configure mech is the DECK declining
    — 501. Starlette resolves handlers by MRO, so the subclass handler must
    win over EngineError's 502 [T9 review m-item]."""
    from app.configure import UnbuiltMechError

    app, _ = make_app(tmp_path, monkeypatch)

    def _boom():
        raise UnbuiltMechError("configure mech 'api' is declared but not implemented")

    # Inserted at the FRONT: the SPA catch-all registered by create_app()
    # shadows any route added after it.
    from fastapi.routing import APIRoute
    app.router.routes.insert(
        0, APIRoute("/__test/unbuilt-mech", _boom, methods=["GET"]))

    client = TestClient(app)
    resp = client.get("/__test/unbuilt-mech")
    assert resp.status_code == 501
    assert "not implemented" in resp.json()["detail"]


# ===========================================================================
# E1 Task 10 — GET /api/engine-kinds, declared-engines CRUD, forget
# ===========================================================================


def test_engine_kinds_route_shape(tmp_path, monkeypatch):
    """The UI kind picker's source (spec §5): every KNOWN_KINDS entry,
    its connection schema (field -> required), and its human verb
    vocabulary — served by the backend, never baked into the UI."""
    app, _deck = make_app(tmp_path, monkeypatch)
    resp = TestClient(app).get("/api/engine-kinds")

    assert resp.status_code == 200
    body = resp.json()
    kinds = {k["kind"]: k for k in body["kinds"]}
    assert set(kinds) == set(KNOWN_KINDS)
    assert kinds["lemonade"]["connection"] == {
        "url": {"required": True},
        "metrics_url": {"required": True},
        "container": {"required": True},
    }
    assert kinds["comfyui"]["connection"] == {"url": {"required": True}}
    assert kinds["hipfire"]["connection"] == {"container": {"required": True}}
    assert set(kinds["lemonade"]["human_verbs"]) == {"load", "unload"}
    assert set(kinds["comfyui"]["human_verbs"]) == {"free"}
    assert set(kinds["hipfire"]["human_verbs"]) == {"park", "resume"}
    # E1 Task 5: remote_capable is now part of the served shape — the E1
    # triple all run as sibling containers on THIS box, so all three read
    # False.
    assert kinds["lemonade"]["remote_capable"] is False
    assert kinds["comfyui"]["remote_capable"] is False
    assert kinds["hipfire"]["remote_capable"] is False
    # sglang-omni Task 7 — the picker now serves FOUR kinds, and the UI
    # needed no change for it: ui/src/model/engineForm.ts builds the whole
    # form from this payload (connection + human_verbs), never a UI literal.
    assert kinds["sglang-omni"]["connection"] == {"url": {"required": True}}
    assert set(kinds["sglang-omni"]["human_verbs"]) == {"load", "unload"}
    assert kinds["sglang-omni"]["remote_capable"] is True


def test_engine_kinds_route_lists_four_kinds(tmp_path, monkeypatch):
    """The count itself, pinned separately from the per-kind shape above: a
    kind added to KNOWN_KINDS but missing from ENGINE_KINDS (or vice versa)
    would 500 this route rather than quietly serving three."""
    app, _deck = make_app(tmp_path, monkeypatch)

    body = TestClient(app).get("/api/engine-kinds").json()

    assert [k["kind"] for k in body["kinds"]] == sorted(
        ["comfyui", "hipfire", "lemonade", "sglang-omni"])


def test_engine_add_validates_and_lands(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])

    r = TestClient(app).post("/api/nodes/local/engines", json=_GGUF_A_ENTRY)

    assert r.status_code == 200
    assert deck["node_store"].get("local")["engines"] == [_GGUF_A_ENTRY]

    bad = dict(_GGUF_A_ENTRY, kind="vllm")
    r2 = TestClient(app).post("/api/nodes/local/engines", json=bad)
    assert r2.status_code == 422


def test_engine_add_422_surfaces_validate_engines_message_via_redacting_handler(
    tmp_path, monkeypatch
):
    """The 422 body must carry validate_engines' own one-line reason, and
    must go through the REDACTING RequestValidationError handler (a list
    of error objects, no `input` echoed) — not the bare app-wide ValueError
    handler (a plain string `detail`), per this router's module docstring."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])
    bad = dict(_GGUF_A_ENTRY, kind="vllm")

    r = TestClient(app).post("/api/nodes/local/engines", json=bad)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)
    assert "unknown kind" in detail[0]["msg"]
    assert all("input" not in err for err in detail)


def test_engine_add_duplicate_409(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])

    r = TestClient(app).post("/api/nodes/local/engines", json=_GGUF_A_ENTRY)

    assert r.status_code == 409
    assert deck["node_store"].get("local")["engines"] == [_GGUF_A_ENTRY]


def test_engine_update_replaces_full_entry(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    moved = dict(_GGUF_A_ENTRY, gpu_index=5,
                 policy_defaults={"priority": 1, "pinned": True, "idle_ttl": 0})

    r = TestClient(app).put("/api/nodes/local/engines/gguf-a", json=moved)

    assert r.status_code == 200
    assert deck["node_store"].get("local")["engines"] == [moved]


def test_engine_update_kind_change_forgets_the_intent_record(tmp_path, monkeypatch):
    """E1 final-review item 5 (controller ruling): a kind-changed
    declaration invalidates the OLD kind's intent record — app.state.py's
    observation half already resets its own per-resource memory on a kind
    mismatch (`_KIND_MEM_KEY`); the intent (goal) half must do the same, or
    a stale old-kind record can drive app.arbiter._restore through the
    WRONG adapter (`app.reconcile.plan_reconcile` copies `intent["engine"]`
    straight into the restore action, `app.arbiter`:1178 resolves the
    adapter off it) — contained today only by the failure budget /
    quarantine, not a correctness guarantee. Fixture is gguf-a (away from
    live topology), re-declared from lemonade-kind to hipfire-kind (a
    connection shape that satisfies hipfire's schema, not lemonade's, so
    this exercises a REAL kind change, not just a relabeling)."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    deck["intent_store"].record("local/gguf-a", state="loaded",
                                model="a.gguf", engine="lemonade")

    reincarnated = dict(_GGUF_A_ENTRY, kind="hipfire",
                        connection={"container": "ods-gguf-a"})
    r = TestClient(app).put("/api/nodes/local/engines/gguf-a", json=reincarnated)

    assert r.status_code == 200
    assert "local/gguf-a" not in deck["intent_store"].get()


def test_engine_update_same_kind_preserves_the_intent_record(tmp_path, monkeypatch):
    """The guard against over-forgetting (E1 final-review item 5's sibling
    case): an edit that does NOT change `kind` — here, just moving
    `gpu_index` and `policy_defaults`, the same edit
    test_engine_update_replaces_full_entry above makes — must leave the
    resource's intent record untouched. Only an actual kind change
    invalidates it."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    deck["intent_store"].record("local/gguf-a", state="loaded",
                                model="a.gguf", engine="lemonade")

    moved = dict(_GGUF_A_ENTRY, gpu_index=5,
                policy_defaults={"priority": 1, "pinned": True, "idle_ttl": 0})
    r = TestClient(app).put("/api/nodes/local/engines/gguf-a", json=moved)

    assert r.status_code == 200
    record = deck["intent_store"].get()["local/gguf-a"]
    assert record["state"] == "loaded"
    assert record["model"] == "a.gguf"


def test_engine_rename_refused(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    moved = dict(_GGUF_A_ENTRY, resource="gguf-z")

    r = TestClient(app).put("/api/nodes/local/engines/gguf-a", json=moved)

    assert r.status_code == 422
    # Refused before any write: the original entry is untouched.
    assert deck["node_store"].get("local")["engines"] == [_GGUF_A_ENTRY]


def test_engine_update_unknown_404(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])

    r = TestClient(app).put("/api/nodes/local/engines/gguf-a", json=_GGUF_A_ENTRY)

    assert r.status_code == 404


def test_engine_delete_unknown_404(tmp_path, monkeypatch):
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])

    r = TestClient(app).delete("/api/nodes/local/engines/gguf-a")

    assert r.status_code == 404


def test_engine_forget_is_bookkeeping_only(tmp_path, monkeypatch):
    """Spec §6.2: removal never touches the engine. The fake keeps
    serving; only the deck's records go."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    fake = deck["local_clients"].fake("gguf-a")     # extend the fake map
    deck["intent_store"].record("local/gguf-a", state="loaded",
                                model="m.gguf", engine="lemonade")

    r = TestClient(app).delete("/api/nodes/local/engines/gguf-a")

    assert r.status_code == 200
    assert "local/gguf-a" not in deck["intent_store"].get()
    assert fake.calls == []                          # nothing actuated
    assert deck["node_store"].get("local")["engines"] == []


def test_engine_forget_deletes_the_stored_policy_row(tmp_path, monkeypatch):
    """Controller ruling: forget deletes the resource's STORED policy row
    (spec §6.2 names policy rows deck bookkeeping), not just makes it
    invisible-until-redeclared — a later re-add of the same resource name
    must not silently inherit a stale override."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [_GGUF_A_ENTRY])
    deck["policy_store"].put(
        {"gguf-a": {"priority": 999, "pinned": True, "idle_ttl": 12345}})
    assert deck["policy_store"].get()["gguf-a"]["priority"] == 999

    r = TestClient(app).delete("/api/nodes/local/engines/gguf-a")
    assert r.status_code == 200

    # Re-add the same resource name: policy must come back at its
    # DECLARED default, not the override the deleted row would have
    # resurrected.
    TestClient(app).post("/api/nodes/local/engines", json=_GGUF_A_ENTRY)
    assert deck["policy_store"].get()["gguf-a"] == _GGUF_A_ENTRY["policy_defaults"]


def test_engine_forget_leaves_settings_scopes_and_events_untouched(tmp_path, monkeypatch):
    """Controller ruling: NOTHING besides the declaration entry, the intent
    record, and the policy row goes. Settings scopes and the event log
    survive a forget."""
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    _declare_local(deck, [_GGUF_A_ENTRY])
    client = TestClient(app)
    client.put("/api/settings/engines/local/lemonade",
               json={"namespace": "args", "values": {"seed": "1"}})
    events_before = len(client.get("/api/events").json()["events"])

    r = client.delete("/api/nodes/local/engines/gguf-a")

    assert r.status_code == 200
    assert deck["settings_store"].get()["engines"]["local/lemonade"]["args"] == {"seed": "1"}
    events_after = client.get("/api/events").json()["events"]
    assert len(events_after) == events_before + 1     # only the new engine-removed line
    assert events_after[-1]["kind"] == "engine-removed"


def test_state_tenants_reflect_a_live_engine_declaration_edit(tmp_path, monkeypatch):
    """T10's whole point, verified end to end at the route: a declaration
    edit made through the new CRUD API is picked up by /api/state's
    world.tenants on the VERY NEXT request — Task 3's live-read promise,
    exercised through this task's own write path rather than a direct
    node_store call."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])
    client = TestClient(app)

    r = client.post("/api/nodes/local/engines", json=_GGUF_A_ENTRY)
    assert r.status_code == 200
    deck["gguf-a"] = FakeLemonade(loaded="extra.m.gguf")

    body = client.get("/api/state").json()

    assert body["world"]["tenants"]["gguf-a"]["state"] == "loaded"
    assert body["world"]["tenants"]["gguf-a"]["engine"] == "lemonade"
    assert body["world"]["placement"]["gguf-a"] == _GGUF_A_ENTRY["gpu_index"]

    client.delete("/api/nodes/local/engines/gguf-a")
    body_after = client.get("/api/state").json()
    assert "gguf-a" not in body_after["world"]["tenants"]


def test_api_state_exact_shape_tests_already_match_the_approved_interface(tmp_path, monkeypatch):
    """T10 disclosure (design §4: 'the two exact-shape tests update to the
    approved interface, disclosed'): `test_api_state_shape`'s top-level key
    set and `test_state_provenance_block_carries_an_updates_count`'s
    provenance-block key set already pin exactly the current interface —
    verified by re-asserting the same envelope here rather than silently
    assuming Task 3's wiring covers it. No change to either test was
    needed; this is the verification, not a duplicate assertion of intent."""
    app, deck = make_app(tmp_path, monkeypatch)
    body = TestClient(app).get("/api/state").json()
    assert set(body.keys()) == {"node", "world", "policy", "models", "lifecycle",
                                "provenance", "nodes"}
    assert set(body["provenance"]) == {"drift", "gaps", "updates"}


# ===========================================================================
# E1 Task 5 — declared-engines CRUD routes gain {node_id} scope (the write
# gate's remote_capable / agent-operability prereqs, exercised at the HTTP
# layer). `/nodes/local/engines*` staying byte-identical is proven by every
# test ABOVE this section still passing unchanged — this section covers only
# what's NEW: a non-local node_id on the same routes.
#
# Fixture rule ([[defaults-that-hide-bugs]]): node id "nimbus" (never the
# live-seeded "sparky"), resource "gguf-r" (never "omni").
# ===========================================================================


def _add_remote_node(deck, node_id="nimbus", *, credential="key-nimbus"):
    """A bare node-agent registry row — address + (optionally) a stored
    credential, no swap/serving machinery — enough for the declared-engines
    CRUD routes below."""
    deck["node_store"].add(
        {"id": node_id, "label": f"{node_id.title()} Box", "agent_kind": "node-agent",
         "address": f"http://{node_id}:7720"},
        credential=credential)


_REMOTE_ENGINE_BODY = {
    "resource": "gguf-r", "kind": "lemonade",
    "connection": {"url": "http://gguf-r:8080",
                   "metrics_url": "http://gguf-r:8001/metrics",
                   "container": "ods-gguf-r"},
    "gpu_index": 4,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 30},
}


def test_get_nodes_surfaces_remote_engines_declaration(tmp_path, monkeypatch):
    """Remote-accept happy path, at the API layer: an empty engines[]
    declaration on a node-agent entry is accepted by the write gate and
    surfaces through GET /api/nodes exactly like the local entry's own
    engines[] already does. (Its POPULATED counterpart is
    test_declaring_sglang_omni_on_a_node_agent_entry_is_accepted below,
    which Task 7 made possible by giving the gate a kind to let through.)"""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)
    deck["node_store"].update("nimbus", {"engines": []})

    body = TestClient(app).get("/api/nodes").json()
    nimbus = next(n for n in body["nodes"] if n["id"] == "nimbus")
    assert nimbus["engines"] == []


def test_add_engine_unknown_node_404(tmp_path, monkeypatch):
    app, _deck = make_app(tmp_path, monkeypatch)
    r = TestClient(app).post("/api/nodes/ghost/engines", json=_GGUF_A_ENTRY)
    assert r.status_code == 404
    # Not just Starlette's route-not-found catch-all (which would ALSO read
    # 404 for a stray typo'd URL): our own "unknown node" refusal, naming it.
    assert "ghost" in r.json()["detail"]


def test_update_engine_unknown_node_404(tmp_path, monkeypatch):
    app, _deck = make_app(tmp_path, monkeypatch)
    r = TestClient(app).put("/api/nodes/ghost/engines/gguf-a", json=_GGUF_A_ENTRY)
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_forget_engine_unknown_node_404(tmp_path, monkeypatch):
    app, _deck = make_app(tmp_path, monkeypatch)
    r = TestClient(app).delete("/api/nodes/ghost/engines/gguf-a")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_add_engine_on_remote_node_kind_not_remote_capable_422_via_redacting_handler(
    tmp_path, monkeypatch
):
    """422 kind-not-remote-capable, naming the kind: the router's own
    shape pre-check passes `remote=True` whenever the target node's
    `agent_kind` is "node-agent", so this refusal goes through the SAME
    redacting RequestValidationError handler as every other engine-shape
    defect (list-shaped `detail`, no `input` echoed) — not NodeStore's own
    bare-ValueError 422 (a plain string `detail`)."""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)

    r = TestClient(app).post("/api/nodes/nimbus/engines", json=_REMOTE_ENGINE_BODY)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)                       # the redacting handler's shape
    assert all("input" not in err for err in detail)
    assert "lemonade" in detail[0]["msg"]
    assert "not remote_capable" in detail[0]["msg"]
    # Refused before any write: nothing landed on the entry.
    assert deck["node_store"].get("nimbus").get("engines", []) == []


def test_add_engine_on_local_node_kind_not_remote_capable_gate_does_not_apply(
    tmp_path, monkeypatch
):
    """Sanity check on the `remote` flag's wiring: the SAME body that 422s
    on a node-agent target (test above) still lands normally on `local` —
    `remote` is derived from the TARGET node's agent_kind, not hardcoded."""
    app, deck = make_app(tmp_path, monkeypatch)
    _declare_local(deck, [])

    r = TestClient(app).post("/api/nodes/local/engines", json=_REMOTE_ENGINE_BODY)

    assert r.status_code == 200
    assert deck["node_store"].get("local")["engines"] == [_REMOTE_ENGINE_BODY]


def test_forget_engine_on_a_node_agent_node_with_nothing_declared_still_404s(
        tmp_path, monkeypatch):
    """RE-EXPRESSED (sglang-omni Task 7, was
    test_forget_engine_on_node_agent_node_404s_today_tripwire).

    The original was a TRIPWIRE: it pinned "404 is the only reachable
    outcome for a node-agent target" precisely so it would FAIL the moment
    Task 7 flipped the first `remote_capable` kind, rather than letting the
    cross-node policy-row collision open silently. It has now fired — a
    declared remote resource IS possible — so the intent it was protecting
    moves to its successor below
    (test_forget_engine_on_a_node_agent_node_does_not_pop_a_same_named_local_policy_row),
    which proves the anti-collision PROPERTY directly instead of proving the
    line is unreachable.

    What survives here is the half that is still exactly true and was never
    the tripwire: forgetting a resource a node does not declare is a 404,
    not a silent success — the same refusal the local entry gives."""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)
    deck["node_store"].update("nimbus", {"engines": []})

    r = TestClient(app).delete("/api/nodes/nimbus/engines/gguf-r")

    assert r.status_code == 404


# The first genuinely remote-capable declaration (sglang-omni Task 7): a
# real one, accepted by the real write gate — no hand-built registry needed
# anymore for this shape. Fixture rule: resource "song-r" (never "omni"),
# GPU 4, idle_ttl 120 (never the declared default 900).
_REMOTE_OMNI_BODY = {
    "resource": "song-r", "kind": "sglang-omni",
    "connection": {"url": "http://127.0.0.1:8008"},
    "gpu_index": 4,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}


def test_declaring_sglang_omni_on_a_node_agent_entry_is_accepted(
        tmp_path, monkeypatch):
    """The gate T5 built, opened by T7's first remote_capable kind: the
    same route that 422s a lemonade declaration on this node accepts an
    sglang-omni one."""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)

    r = TestClient(app).post("/api/nodes/nimbus/engines", json=_REMOTE_OMNI_BODY)

    assert r.status_code == 200
    assert deck["node_store"].get("nimbus")["engines"] == [_REMOTE_OMNI_BODY]


def test_forget_engine_on_a_node_agent_node_does_not_pop_a_same_named_local_policy_row(
        tmp_path, monkeypatch):
    """SUCCESSOR to the 404 tripwire above (controller ruling R7): now that
    a remote engine can really be declared, prove the property the tripwire
    was standing guard over.

    PolicyStore is node-BLIND — its rows are keyed by bare resource
    (app/policy.py), its declared-defaults source reads only the LOCAL
    entry's engines[], and app.arbiter indexes it by bare resource — so an
    ungated `policy_store.forget(resource)` in `forget_engine` would pop the
    unrelated LOCAL row of the same name. Resource names are unique per
    NODE, never globally, so "song-r" on nimbus and "song-r" on this box are
    two different things; forgetting one must not silently un-pin the other.

    The stored override (pinned=True, idle_ttl 240 — both away from the
    declared defaults, [[defaults-that-hide-bugs]]) is what makes this
    detectable: without the guard the row is popped and `get()` falls back
    to the DECLARED defaults, which are deliberately different values."""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)
    deck["node_store"].update("nimbus", {"engines": [_REMOTE_OMNI_BODY]})
    # A LOCAL resource of the same name, with its own stored override.
    _declare_local(deck, [{**_GGUF_A_ENTRY, "resource": "song-r"}])
    override = {"priority": 33, "pinned": True, "idle_ttl": 240}
    deck["policy_store"].put({"song-r": override})

    r = TestClient(app).delete("/api/nodes/nimbus/engines/song-r")

    assert r.status_code == 200
    assert deck["node_store"].get("nimbus")["engines"] == []
    # The local policy row is untouched — not merely present, but still
    # carrying the OVERRIDE rather than the declaration's defaults.
    assert deck["policy_store"].get()["song-r"] == override


def test_forget_engine_on_a_node_agent_node_still_forgets_its_own_intent(
        tmp_path, monkeypatch):
    """The other half of forget-is-bookkeeping, and the reason R7's guard is
    scoped to POLICY alone: the intent record is node-keyed since Task 6, so
    it can be — and is — forgotten for the right node, with the local record
    of the same name left standing."""
    app, deck = make_app(tmp_path, monkeypatch)
    _add_remote_node(deck)
    deck["node_store"].update("nimbus", {"engines": [_REMOTE_OMNI_BODY]})
    deck["intent_store"].record("nimbus/song-r", state="loaded", model=None,
                                engine="sglang-omni")
    deck["intent_store"].record("local/song-r", state="loaded", model="a.gguf",
                                engine="lemonade")

    TestClient(app).delete("/api/nodes/nimbus/engines/song-r")

    intents = deck["intent_store"].get()
    assert "nimbus/song-r" not in intents
    assert intents["local/song-r"]["model"] == "a.gguf"


# ===========================================================================
# sglang-omni Task 6 — node-keyed remote engine observation, at the route.
#
# `/api/state` now walks EVERY registry entry's engines[], not just the local
# one. The registry state below is HAND-BUILT (conftest's HandBuiltRegistry)
# because the Task 5 write gate refuses a LEMONADE-kind declaration on a
# node-agent entry (still true after Task 7, which made only sglang-omni
# remote-capable) and these tests deliberately use one — the remote
# observation path must work for ANY kind. The gate is not weakened; the
# tests that want the REAL gate declare sglang-omni through it instead.
#
# Fixture rule ([[defaults-that-hide-bugs]]): node "nimbus" (never the
# live-seeded "sparky"), resource "gguf-r" (never "omni"), GPU 4.
# ===========================================================================


class _StubAgent:
    """NodeAgentClient-shaped stub: the GPU pool read, and close()."""

    def __init__(self, payload=None, raises=None) -> None:
        self._payload = payload
        self._raises = raises
        self.calls = 0          # probe pacing (Task 7) counts these

    def gpu(self) -> dict:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._payload

    def close(self) -> None:
        pass


def _wire_remote_engine(deck, hand_built_registry, *, agent, engine_client):
    """Swap in a registry carrying one remote engine, plus the two deck
    entries the remote half of the world reads through."""
    from app.node_clients import RemoteEngineClients

    entries = [
        {"id": "local", "label": "This Box", "agent_kind": "local",
         "control": "none", "engines": _ENGINES},
        {"id": "nimbus", "label": "Nimbus Box", "agent_kind": "node-agent",
         "address": "http://nimbus:7720", "control": "none",
         "engines": [dict(_REMOTE_ENGINE_BODY)]},
    ]
    store = hand_built_registry(entries, {"nimbus": "key-nimbus"})
    deck["node_store"] = store
    deck["node_agent_client_factory"] = lambda address, credential: agent
    deck["remote_engine_clients"] = RemoteEngineClients(
        store, lambda entry, credential, declared: engine_client)
    return store


def test_state_reports_a_remote_engine_with_its_node_id(
        tmp_path, monkeypatch, hand_built_registry):
    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry,
        agent=_StubAgent({"gpus": [{"index": 4, "memory_total_mb": 8,
                                    "memory_used_mb": 3}]}),
        engine_client=FakeLemonade(loaded="extra.r.gguf"))

    body = TestClient(app).get("/api/state").json()

    tenant = body["world"]["remote_tenants"]["nimbus/gguf-r"]
    assert tenant["node_id"] == "nimbus"
    assert tenant["resource"] == "gguf-r"
    assert tenant["state"] == "loaded"
    assert body["world"]["remote_gpus"]["nimbus"] == [
        {"index": 4, "total": 8 * 1024**2, "used": 3 * 1024**2,
         "free": 5 * 1024**2}]


def test_state_lifecycle_view_carries_the_node_keyed_remote_engine(
        tmp_path, monkeypatch, hand_built_registry):
    """The observation key is `<node>/<resource>` — the same shape intent,
    policy and the UI already join local placements on."""
    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry,
        agent=_StubAgent({"gpus": []}),
        engine_client=FakeLemonade(loaded="extra.r.gguf"))

    body = TestClient(app).get("/api/state").json()

    assert body["lifecycle"]["nimbus/gguf-r"]["observed"] == {
        "reachable": True, "loaded": True, "model": "extra.r.gguf",
        "transitioning": False}


def test_state_reports_a_remote_engine_as_unknown_when_its_agent_is_down(
        tmp_path, monkeypatch, hand_built_registry):
    """A powered-off box is a NORMAL state: the route still answers, and the
    engine reads as "we failed to look", never as "nothing is loaded"."""
    from app.engines.node_agent import NodeAgentUnreachable

    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry,
        agent=_StubAgent(raises=NodeAgentUnreachable("connection refused")),
        engine_client=FakeLemonade(loaded="extra.r.gguf"))

    r = TestClient(app).get("/api/state")

    assert r.status_code == 200
    body = r.json()
    assert body["world"]["remote_gpus"]["nimbus"] is None
    assert body["lifecycle"]["nimbus/gguf-r"]["observed"] == {
        "reachable": False, "loaded": False, "model": None,
        "transitioning": False}


def test_state_local_resources_keep_their_bare_resource_keys(
        tmp_path, monkeypatch, hand_built_registry):
    """The local half is UNTOUCHED by the remote walk: `world.tenants` stays
    keyed by bare resource and the remote engines live in their own half,
    so a remote engine's gpu_index can never be matched against the LOCAL
    GPU list by the arbiter's co-residency arithmetic."""
    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry, agent=_StubAgent({"gpus": []}),
        engine_client=FakeLemonade(loaded="extra.r.gguf"))

    body = TestClient(app).get("/api/state").json()

    assert "gguf-r" not in body["world"]["tenants"]
    assert set(body["world"]["tenants"]) == {"lemonade", "comfyui", "hipfire"}


def test_remote_engine_drift_is_not_translated_through_its_nodes_slot_map(
        tmp_path, monkeypatch, hand_built_registry):
    """A node can be a swap node AND declare its own engines[]. The
    profile->identity translation `_settings_drift` applies belongs to that
    node's SERVING SLOT only (resource "slot0"), never to a declared engine
    that merely happens to sit on the same box: translating there would
    diff a declared engine's settings against the vLLM/checkpoint scope of
    an unrelated swap profile.

    `build_lifecycle_view` gates the translation on map PRESENCE (that
    function's own docstring), so the map has to be keyed by the slot key
    it belongs to — keying it by node id alone silently handed it to every
    key on that node the moment a node could own more than one.
    """
    from app.intent import IntentStore
    from app.settings_store import SettingsStore

    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry, agent=_StubAgent({"gpus": []}),
        engine_client=FakeLemonade(loaded="heretic"))
    deck["node_store"].patch("nimbus", control="swap",
                             serving_address="http://nimbus:8000")
    deck["characteristics_store"].put_fields(
        "engine/nimbus/vllm",
        {"profile_identities": {
            "value": {"heretic": {"identity": "her-1", "service": "vllm",
                                  "container_name": "vllm-her"}},
            "source": "test", "derived_ts": "2026-08-16T00:00:00+00:00"}})
    intent = IntentStore(tmp_path / "intent.json")
    intent.record("nimbus/gguf-r", state="loaded", model="heretic",
                  engine="lemonade")
    deck["intent_store"] = intent
    deck["settings_store"] = SettingsStore(tmp_path / "s.json")
    client = TestClient(app)

    # A write against the SLOT's translated scope must not light up the
    # declared engine's drift flag.
    client.put("/api/settings/engines/nimbus/vllm",
               json={"namespace": "env", "values": {"VLLM_MAX_SEQ": "8"}})

    entry = client.get("/api/state").json()["lifecycle"]["nimbus/gguf-r"]
    assert entry["settings_drift"] is None


def test_adopting_a_remote_engine_404s_with_no_engine_owns(
        tmp_path, monkeypatch, hand_built_registry):
    """Disclosed gap, pinned rather than left latent: a remote engine is now
    OBSERVED (so `/api/lifecycle/adopt` finds a record for its key), but
    `app.observe.engine_for` still resolves only local resources and swap
    slots — so adopt refuses with "no engine owns", never a crash and never
    a wrong-kind intent record. Extending `engine_for` to remote keys
    belongs with the remote verb/reconcile work (Tasks 8/9), which is what
    gives an adopted remote record anything to drive."""
    app, deck = make_app(tmp_path, monkeypatch)
    _wire_remote_engine(
        deck, hand_built_registry, agent=_StubAgent({"gpus": []}),
        engine_client=FakeLemonade(loaded="extra.r.gguf"))

    r = TestClient(app).post("/api/lifecycle/adopt/nimbus/gguf-r")

    assert r.status_code == 404
    assert "no engine owns" in r.json()["detail"]


def test_state_paces_the_remote_probes_across_requests(
        tmp_path, monkeypatch, hand_built_registry):
    """sglang-omni Task 7 — the pacing obligation at the real surface.

    Every /api/state used to cost one `GET /v1/node/gpu` per declaring node
    plus one status probe per declared engine, each behind a 5 s transport
    timeout, and the ~2 s arbiter tick pays the same bill. Three requests
    inside the TTL now cost ONE probe (deck["remote_observer"], shared with
    the watcher so both describe one world).

    Vacuity guard: the remote engine must still be REPORTED — pacing that
    worked by observing nothing would pass a probe-count assertion."""
    app, deck = make_app(tmp_path, monkeypatch)
    agent = _StubAgent({"gpus": []})
    _wire_remote_engine(deck, hand_built_registry, agent=agent,
                        engine_client=FakeLemonade(loaded="extra.r.gguf"))
    client = TestClient(app)

    bodies = [client.get("/api/state").json() for _ in range(3)]

    assert agent.calls == 1
    for body in bodies:
        assert body["world"]["remote_tenants"]["nimbus/gguf-r"]["state"] == "loaded"
