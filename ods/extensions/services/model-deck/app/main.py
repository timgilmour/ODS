"""
ODS Model Deck — GPU/VRAM supervisor.

FastAPI backend that monitors and arbitrates GPU/VRAM usage across ODS
inference engines (Lemonade, hipfire, ComfyUI, llama-server), parking
containers under memory pressure. This module wires the app factory, the
HTTP API (routers/), the health endpoint, and the arbiter
watcher's lifecycle into the FastAPI lifespan.

Modules:
  settings.py  — Settings (pydantic-settings), env-driven configuration
  arbiter.py   — decide() (pure) + Watcher (daemon thread)
  (no auth module — the admin gate was deliberately removed 2026-07-22)
  routers/     — the HTTP API, one module per resource area, mounted under /api
  main.py      — create_app() factory + module-level app for uvicorn

Dependency injection: every router pulls its clients/stores from
``request.app.state.deck`` (built by ``_build_deck``) instead of
constructing anything itself. Tests call ``create_app()`` and then replace
individual ``app.state.deck[...]`` entries with fakes before issuing
requests — no env vars, no real sockets required.

The watcher starts on app startup and stops on shutdown, UNLESS the env var
``MODEL_DECK_NO_WATCHER=1`` is set (tests set it, and bare-uvicorn runs that
don't want the background loop can too). When it does run, it shares the
exact same World/engine-client/store instances as the HTTP routers (see
``_build_deck`` / ``_build_watcher`` below) — important because ``World``
carries real in-memory idle-clock state that must stay single-sourced, not
forked into two silently-diverging copies.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.engines import BusyError, EngineError, GuardError
from app.gateway import detect_default_gateway
from app.routers import (
    control,
    facts,
    lifecycle,
    policy,
    rename,
    sets,
    spark,
    status,
    storage,
)
from app.routers import provenance as provenance_router
# Aliased: create_app() below has its own local `settings` (the Settings()
# instance) and app.settings already owns that name for env config — the
# router module would silently shadow (or be shadowed by) either otherwise.
from app.routers import settings as settings_router
from app.settings import Settings

# The GGUF store is bound read-only into the container at this path (see
# compose.yaml). It has no Settings field of its own.
_GGUF_DIR = Path("/gguf-store")

# hipfire runs as a sibling container on the compose network; its health
# endpoint is <container>:11435/health (config/ports.json + manifest.yaml).
_HIPFIRE_PORT = 11435

# Deck cache keyed by the id() of the Settings instance that produced it.
#
# Why this exists: tests/test_health.py (unmodifiable — see Task 11 brief)
# requires two things simultaneously: (1) lifespan must construct the real
# watcher via a bare call `_build_watcher(settings)` — one positional arg,
# no more — so it stays swappable via `monkeypatch.setattr(main,
# "_build_watcher", lambda settings: fake)`; (2) `_build_watcher(Settings())`
# must also work completely standalone, with no app/deck in scope. Neither
# constraint leaves room to thread an already-built deck into
# `_build_watcher` as an explicit argument. Caching `_build_deck`'s result by
# the identity of the `settings` object it was built from lets create_app()
# and (later, from lifespan) `_build_watcher(settings)` resolve to the exact
# same dict — same World, same engine clients — whenever they're handed the
# same settings instance, while still building fresh, independent instances
# for any other settings object (e.g. the standalone unit test's).
#
# Safe against id() reuse after garbage collection: each cache entry holds a
# strong reference to the settings object it was built from, so that exact
# id() cannot be recycled for a *different* object while the entry lives,
# and the `cached_settings is settings` check below refuses a stale hit even
# if it somehow could be.
_deck_by_settings_id: dict[int, tuple[Settings, dict]] = {}


def _build_deck(settings: Settings) -> dict:
    """Construct (once per distinct `settings` instance — see the cache
    comment above) every client/store/World instance the HTTP routers and
    the watcher share. Imports are local so importing app.main stays light
    and free of import cycles, and so the clients (which open no sockets at
    construction) are only built when an app is actually being assembled."""
    cached = _deck_by_settings_id.get(id(settings))
    if cached is not None and cached[0] is settings:
        return cached[1]

    from app.arbiter import HealSuppressor
    from app.catalog import Catalog
    from app.characteristics import CharacteristicsStore
    from app.declared import DeclaredStore
    from app.engines.comfyui import ComfyClient
    from app.engines.docker_ctl import DockerCtl
    from app.engines.hipfire import HipfireClient
    from app.engines.hostagent import HostAgent
    from app.engines.lemonade import LemonadeClient
    from app.engines.litellm import LiteLLMClient
    from app.gpu import read_gpus
    from app.intent import IntentStore
    from app.locations import LocationStore
    from app.mover import JobQueue, Mover
    from app.policy import PolicyStore, StoragePolicyStore
    from app.provenance import ProvenanceStore
    from app.registry import Registry
    from app.sets import SetStore
    from app.settings_store import SettingsStore
    from app.state import World

    data_dir = Path(settings.data_dir)

    lemonade = LemonadeClient(
        settings.lemonade_url,
        settings.lemonade_key,
        metrics_url=settings.lemonade_metrics_url,
    )
    comfy = ComfyClient(settings.comfyui_url)
    litellm = LiteLLMClient(settings.litellm_url, settings.litellm_key)
    dockerctl = DockerCtl(settings.dockerctl_url, settings.park_allowlist)
    hipfire = HipfireClient(
        health_url=f"http://{settings.hipfire_container}:{_HIPFIRE_PORT}/health",
        dockerctl=dockerctl,
        container=settings.hipfire_container,
        litellm=litellm,
        stats_url=f"http://{settings.hipfire_container}:{_HIPFIRE_PORT}/stats",
        activity_window_s=settings.hipfire_activity_window_s,
    )
    hostagent_url = settings.hostagent_url or (
        f"http://{detect_default_gateway() or 'host.docker.internal'}:7710"
    )
    hostagent = HostAgent(hostagent_url, settings.hostagent_key)

    from app.engines.spark import SparkClient

    spark_client = None
    if settings.spark_node_url and settings.spark_serving_url:
        try:
            node_keys = json.loads(settings.spark_node_keys_json or "{}")
        except ValueError:
            node_keys = {}
        spark_key = node_keys.get(settings.spark_node_name, "")
        if isinstance(spark_key, str) and spark_key:
            spark_client = SparkClient(
                node_url=settings.spark_node_url,
                node_key=spark_key,
                serving_url=settings.spark_serving_url,
                litellm=litellm,
            )

    location_store = LocationStore(data_dir / "locations.json")
    catalog = Catalog(data_dir / "catalog.json", location_store)
    mover = Mover()
    job_queue = JobQueue(mover, catalog, location_store, data_dir / "events.jsonl")

    deck = {
        "settings": settings,
        "world": World(placement={
            "hipfire": settings.hipfire_gpu_index,
            "lemonade": settings.lemonade_gpu_index,
            "comfyui": settings.lemonade_gpu_index,  # comfy shares the lemonade GPU (settings.py comment)
        }),
        "lemonade": lemonade,
        "comfy": comfy,
        "hipfire": hipfire,
        "hostagent": hostagent,
        "spark": spark_client,
        "litellm": litellm,
        "registry": Registry(data_dir / "registry.json", _GGUF_DIR),
        "characteristics_store": CharacteristicsStore(data_dir / "characteristics.json"),
        "declared_store": DeclaredStore(data_dir / "declared.json"),
        # Human/UI-owned launch configuration (app.settings_store) — a
        # sibling of characteristics/declared, not a replacement: this store
        # asserts what things are launched WITH, the other two what they ARE.
        "settings_store": SettingsStore(data_dir / "settings.json"),
        "gguf_dir": _GGUF_DIR,
        "policy_store": PolicyStore(data_dir / "policy.json"),
        # Durable desired state. Shared, like policy_store, between the HTTP
        # routers (which write it on every deliberate action) and — once the
        # reconcile pass lands — the watcher, which reads it.
        "intent_store": IntentStore(data_dir / "intent.json"),
        # The provenance ledger: where each artifact came from and what
        # version of it is here now. RECORDS ONLY — nothing converges to a
        # desired version (see Watcher._provenance_pass). Its history is a
        # separate append-only file, kept out of events.jsonl deliberately:
        # that log is display-only and accepts losing its tail.
        "provenance_store": ProvenanceStore(
            data_dir / "provenance.json", data_dir / "provenance-history.jsonl"),
        "provenance_history_path": data_dir / "provenance-history.jsonl",
        "set_store": SetStore(data_dir / "sets"),
        "events_path": data_dir / "events.jsonl",
        "read_gpus": read_gpus,
        "drm_root": settings.drm_root,
        "kfd_root": settings.kfd_root,
        # Shared between the watcher and the HTTP routers (manual load/unload,
        # set-apply) so every deck-initiated unload/load coordinates on one
        # suppression window.
        "heal_suppressor": HealSuppressor(settings.heal_suppress_s),
        "dockerctl": dockerctl,
        "location_store": location_store,
        "catalog": catalog,
        "storage_policy_store": StoragePolicyStore(data_dir / "storage_policy.json"),
        "mover": mover,
        "job_queue": job_queue,
    }
    # Late-bound so the queue's execution-start guard can re-snapshot the world
    # (spec section 2). It has to be assigned rather than injected: the queue is
    # constructed above, before the deck dict it would need to read from exists.
    from app.routers import build_world_snapshot

    job_queue.world_fn = lambda: build_world_snapshot(deck)

    # Also late-bound, and for a second reason: it reads deck["spark"] on
    # every call rather than capturing the client, so a test that swaps the
    # deck entry after create_app() is still observed by the one shared cache.
    from app.observe import SparkObserver

    deck["spark_observer"] = SparkObserver(lambda: deck["spark"])

    # Catalog harvest (app.arbiter.Watcher._harvest_catalogs, and the manual
    # force-harvest route app.routers.settings.harvest_now): routes maps
    # each configurable (node, engine) pair to the one adapter that can
    # actually produce a catalog for it. Spark is this box's one real
    # vLLM-backed target (live-verified 2026-08-07 — see
    # Watcher._configurable_engines's docstring: hipfire is confirmed a
    # Bun daemon, not vLLM, so no local docker-exec route belongs here).
    # DockerEngineExec (app.engines.docker_ctl) stays defined for a future
    # local vLLM engine but is deliberately not constructed below — there
    # is nothing local to route it to today. routes stays {} (engine_exec
    # None, harvest disabled entirely, same as every pre-C2 build) on a
    # box with no spark configured — the node half of the pair always
    # comes from spark_node_id(), never settings.node_label or
    # settings.spark_node_name (see _configurable_engines' docstring for
    # the historical bug that rule guards against).
    #
    # Built HERE, in _build_deck, not in _build_watcher (moved 2026-08-08,
    # task 3 review finding 1): _build_deck runs in EVERY mode, including
    # MODEL_DECK_NO_WATCHER=1 (this module's own documented "bare-uvicorn
    # runs that don't want the background loop" support, see the module
    # docstring) — lifespan() skips _build_watcher ENTIRELY under that env
    # var. Building the routes here means deck["engine_exec"]/
    # deck["configurable_engines"] (and therefore app.state.deck, the same
    # dict via the cache above) are always populated whenever spark is
    # configured, watcher running or not, so harvest_now's pair check can
    # never misdiagnose "watcher never wired" as "pair not configured".
    from app.engines.docker_ctl import EngineExecRouter
    from app.engines.spark import SparkCatalogExec
    from app.observe import spark_node_id

    routes = {}
    if deck["spark"] is not None:
        routes[(spark_node_id(), "vllm")] = SparkCatalogExec(deck["spark"])
    deck["engine_exec"] = EngineExecRouter(routes) if routes else None
    deck["configurable_engines"] = sorted(routes)

    _deck_by_settings_id[id(settings)] = (settings, deck)
    return deck


def _build_watcher(settings: Settings):
    """Construct the arbiter Watcher from the same shared deck _build_deck
    hands the HTTP routers (see the cache comment above) — real clients +
    stores + World, wired from Settings."""
    from app.arbiter import Watcher

    deck = _build_deck(settings)

    return Watcher(
        settings=deck["settings"],
        world=deck["world"],
        lemonade=deck["lemonade"],
        comfy=deck["comfy"],
        hipfire=deck["hipfire"],
        litellm=deck["litellm"],
        registry=deck["registry"],
        policy_store=deck["policy_store"],
        events_path=deck["events_path"],
        read_gpus=deck["read_gpus"],
        heal_suppressor=deck["heal_suppressor"],
        hostagent=deck["hostagent"],
        catalog=deck["catalog"],
        # Lifecycle reconciliation: the same intent store the HTTP routers
        # write on every deliberate action, plus the spark client whose slot
        # is one of the reconciled resources (None on a box without one).
        intent_store=deck["intent_store"],
        spark=deck["spark"],
        # One cached spark probe for the whole process, shared with the HTTP
        # paths: status() costs two node-agent requests and an absent sparky
        # burns a 5 s timeout on each.
        spark_observer=deck["spark_observer"],
        # Characteristics derive pass: same shared store the HTTP routers
        # will read from, and the read-only GGUF mount to scan.
        characteristics_store=deck["characteristics_store"],
        gguf_dir=deck["gguf_dir"],
        # Built in _build_deck (same shared dict as app.state.deck), not
        # here — see that function's routes comment: None (harvest fully
        # disabled) on a box with no spark configured, otherwise the
        # router built from the one real (spark_node_id(), "vllm") route.
        engine_exec=deck["engine_exec"],
        configurable_engines=deck["configurable_engines"],
        # Provenance pass: the same shared ledger the HTTP routers read and
        # declare into, plus the socket-proxy client that supplies local
        # image identity (one inspect per park-allowlist container — no new
        # proxy permission, see DockerCtl.inspect).
        provenance_store=deck["provenance_store"],
        dockerctl=deck["dockerctl"],
    )


def _build_storage_watcher(settings: Settings):
    """Construct the StorageWatcher from the same shared deck _build_deck
    hands the HTTP routers (see the cache comment above) — same
    settings-id-keyed cache pattern as _build_watcher."""
    from app.routers import build_world_snapshot
    from app.storage import StorageWatcher

    deck = _build_deck(settings)
    return StorageWatcher(settings=deck["settings"], location_store=deck["location_store"],
                          catalog=deck["catalog"], storage_policy_store=deck["storage_policy_store"],
                          job_queue=deck["job_queue"], world_fn=lambda: build_world_snapshot(deck),
                          events_path=deck["events_path"])


def _build_update_checker(deck: dict):
    """Construct the UpdateChecker from the already-built deck (unlike
    _build_watcher/_build_storage_watcher, this one is only ever called from
    inside lifespan, where `deck` is already in scope — no standalone/swap
    test requires a bare `_build_update_checker(settings)` entry point the
    way test_health.py's `monkeypatch.setattr(main, "_build_watcher", ...)`
    does for the watcher).

    `deck.get("provenance_store")`, not `deck["provenance_store"]`: this
    thread must behave sanely (construct cleanly, `tick()` a clean no-op)
    even on a deck that has no provenance store, even though every real
    `_build_deck` call populates one today (see UpdateChecker.tick`'s own
    `self._store is None` guard)."""
    from app.update_check import UpdateChecker

    return UpdateChecker(settings=deck["settings"],
                         provenance_store=deck.get("provenance_store"),
                         events_path=deck["events_path"])


def create_app() -> FastAPI:
    """Build the Model Deck FastAPI app. Requires no environment variables."""
    settings = Settings()
    deck = _build_deck(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from pathlib import Path as _Path
        roots = [_Path(loc["path"]) for loc in deck["location_store"].list()
                 if deck["location_store"].available(loc)]
        deck["mover"].janitor(roots)
        deck["job_queue"].start()
        watcher = None
        storage_watcher = None
        update_checker = None
        if os.environ.get("MODEL_DECK_NO_WATCHER") != "1":
            watcher = _build_watcher(settings)
            watcher.start()
            storage_watcher = _build_storage_watcher(settings)
            storage_watcher.start()
            update_checker = _build_update_checker(deck)
            deck["update_checker"] = update_checker
            update_checker.start()
        try:
            yield
        finally:
            # Reverse start order: update_checker started last, stops first.
            if update_checker is not None:
                update_checker.stop()
            if storage_watcher is not None:
                storage_watcher.stop()
            if watcher is not None:
                watcher.stop()
            deck["job_queue"].stop()

    app = FastAPI(title="Model Deck", lifespan=lifespan)
    app.state.settings = settings
    app.state.deck = deck

    @app.exception_handler(GuardError)
    async def _handle_guard_error(request: Request, exc: GuardError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(BusyError)
    async def _handle_busy_error(request: Request, exc: BusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def _handle_engine_error(request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(status.router, prefix="/api")
    app.include_router(control.router, prefix="/api")
    app.include_router(sets.router, prefix="/api")
    app.include_router(policy.router, prefix="/api")
    app.include_router(spark.router, prefix="/api")
    app.include_router(lifecycle.router, prefix="/api")
    app.include_router(storage.router, prefix="/api")
    app.include_router(facts.router, prefix="/api")
    app.include_router(settings_router.router, prefix="/api")
    app.include_router(rename.router, prefix="/api")
    app.include_router(provenance_router.router, prefix="/api")

    # ui/dist doesn't exist until the UI build lands — mount only when
    # present so the API keeps working standalone until then. Mounted LAST:
    # a StaticFiles Mount("/") matches every path Starlette hasn't already
    # matched, so it must never be registered before /health or /api/*.
    ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
