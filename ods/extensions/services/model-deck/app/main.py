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

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.engines import BusyError, EngineError, GuardError
from app.gateway import detect_default_gateway
from app.routers import control, policy, sets, status
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
    from app.engines.comfyui import ComfyClient
    from app.engines.docker_ctl import DockerCtl
    from app.engines.hipfire import HipfireClient
    from app.engines.hostagent import HostAgent
    from app.engines.lemonade import LemonadeClient
    from app.engines.litellm import LiteLLMClient
    from app.gpu import read_gpus
    from app.policy import PolicyStore
    from app.registry import Registry
    from app.sets import SetStore
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

    deck = {
        "settings": settings,
        "world": World(),
        "lemonade": lemonade,
        "comfy": comfy,
        "hipfire": hipfire,
        "hostagent": hostagent,
        "litellm": litellm,
        "registry": Registry(data_dir / "registry.json", _GGUF_DIR),
        "policy_store": PolicyStore(data_dir / "policy.json"),
        "set_store": SetStore(data_dir / "sets"),
        "events_path": data_dir / "events.jsonl",
        "read_gpus": read_gpus,
        "drm_root": settings.drm_root,
        "kfd_root": settings.kfd_root,
        # Shared between the watcher and the HTTP routers (manual load/unload,
        # set-apply) so every deck-initiated unload/load coordinates on one
        # suppression window.
        "heal_suppressor": HealSuppressor(settings.heal_suppress_s),
    }
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
    )


def create_app() -> FastAPI:
    """Build the Model Deck FastAPI app. Requires no environment variables."""
    settings = Settings()
    deck = _build_deck(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher = None
        if os.environ.get("MODEL_DECK_NO_WATCHER") != "1":
            watcher = _build_watcher(settings)
            watcher.start()
        try:
            yield
        finally:
            if watcher is not None:
                watcher.stop()

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

    # ui/dist doesn't exist until the UI build lands — mount only when
    # present so the API keeps working standalone until then. Mounted LAST:
    # a StaticFiles Mount("/") matches every path Starlette hasn't already
    # matched, so it must never be registered before /health or /api/*.
    ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
