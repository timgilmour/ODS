"""
ODS Model Deck — GPU/VRAM supervisor.

FastAPI backend that monitors and arbitrates GPU/VRAM usage across ODS
inference engines (Lemonade, hipfire, ComfyUI, llama-server), parking
containers under memory pressure. This module wires the app factory, the
health endpoint, and the arbiter watcher's lifecycle into the FastAPI
lifespan.

Modules:
  settings.py — Settings (pydantic-settings), env-driven configuration
  arbiter.py  — decide() (pure) + Watcher (daemon thread)
  main.py     — create_app() factory + module-level app for uvicorn

The watcher starts on app startup and stops on shutdown, UNLESS the env var
``MODEL_DECK_NO_WATCHER=1`` is set (tests set it, and bare-uvicorn runs that
don't want the background loop can too).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.settings import Settings

# The GGUF store is bound read-only into the container at this path (see
# compose.yaml). It has no Settings field of its own.
_GGUF_DIR = Path("/gguf-store")

# hipfire runs as a sibling container on the compose network; its health
# endpoint is <container>:11435/health (config/ports.json + manifest.yaml).
_HIPFIRE_PORT = 11435


def _build_watcher(settings: Settings):
    """Construct the arbiter Watcher with real engine clients + stores from
    Settings. Imports are local so importing app.main stays light and free of
    import cycles, and so the clients (which open no sockets at construction)
    are only built when the watcher is actually wanted."""
    from app.arbiter import Watcher
    from app.engines.comfyui import ComfyClient
    from app.engines.docker_ctl import DockerCtl
    from app.engines.hipfire import HipfireClient
    from app.engines.lemonade import LemonadeClient
    from app.engines.litellm import LiteLLMClient
    from app.gpu import read_gpus
    from app.policy import PolicyStore
    from app.registry import Registry
    from app.state import World

    data_dir = Path(settings.data_dir)

    lemonade = LemonadeClient(settings.lemonade_url, settings.lemonade_key)
    comfy = ComfyClient(settings.comfyui_url)
    litellm = LiteLLMClient(settings.litellm_url, settings.litellm_key)
    dockerctl = DockerCtl(settings.dockerctl_url, settings.park_allowlist)
    hipfire = HipfireClient(
        health_url=f"http://{settings.hipfire_container}:{_HIPFIRE_PORT}/health",
        dockerctl=dockerctl,
        container=settings.hipfire_container,
        litellm=litellm,
    )

    return Watcher(
        settings=settings,
        world=World(),
        lemonade=lemonade,
        comfy=comfy,
        hipfire=hipfire,
        litellm=litellm,
        registry=Registry(data_dir / "registry.json", _GGUF_DIR),
        policy_store=PolicyStore(data_dir / "policy.json"),
        events_path=data_dir / "events.jsonl",
        read_gpus=read_gpus,
    )


def create_app() -> FastAPI:
    """Build the Model Deck FastAPI app. Requires no environment variables."""
    settings = Settings()

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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ui/dist doesn't exist until the UI build lands (Task 11) — mount only
    # when present so the API keeps working standalone until then.
    ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
