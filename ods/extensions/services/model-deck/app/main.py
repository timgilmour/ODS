"""
ODS Model Deck — GPU/VRAM supervisor.

FastAPI backend that will monitor and arbitrate GPU/VRAM usage across ODS
inference engines (Lemonade, hipfire, ComfyUI, llama-server), parking
containers under memory pressure. This module only wires up the app
factory and health endpoint; engine clients and the arbiter land in later
tasks.

Modules:
  settings.py — Settings (pydantic-settings), env-driven configuration
  main.py     — create_app() factory + module-level app for uvicorn
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.settings import Settings


def create_app() -> FastAPI:
    """Build the Model Deck FastAPI app. Requires no environment variables."""
    settings = Settings()

    app = FastAPI(title="Model Deck")
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
