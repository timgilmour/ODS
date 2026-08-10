"""Shared test fixtures.

Autouse default for MODEL_DECK_DATA_DIR: Settings.data_dir defaults to the
production path "/data" (the compose bind mount). Outside the deployed
container that path doesn't exist and isn't writable by a non-root test
runner. Historically no test needed it to be writable — every store built
from data_dir (registry/policy_store/set_store) got swapped for a fake or a
tmp_path-backed real instance before any route touched it (see
tests/test_api.py's make_app).

Task 11 (storage tiering wiring) changes that: app.routers.storage /
app.routers.control's pull-through path now call real Catalog/LocationStore
methods (scan/describe) that unconditionally persist to data_dir, even from
fixtures (tests/test_api.py, tests/test_health.py) that only ever intended
to fake out the engine clients, not the storage stores. Rather than editing
those already-passing, task-11-brief-designated-unmodifiable test files,
give every test a writable, per-test-isolated data dir by default here; a
test that wants its own (e.g. to assert exact on-disk layout) simply calls
monkeypatch.setenv("MODEL_DECK_DATA_DIR", ...) itself, which — running after
this fixture, in the same test's monkeypatch stack — wins.

Module-level fallback below: Task 6 (node registry wiring) makes
``_build_deck`` write through ``seed_if_missing`` unconditionally, which
means even the bare ``app = create_app()`` at the bottom of app/main.py
(the uvicorn entrypoint) now touches disk. That line runs the instant
anything imports app.main — including every test module's own `from app
import main`, which happens at COLLECTION time, before the autouse fixture
above ever gets to run. `os.environ.setdefault` here (module scope, so it
executes once, before pytest imports any test file) gives that first import
a writable scratch dir instead of the unwritable "/data" default; the
autouse fixture then overrides it per-test as before.
"""

import os
import tempfile

import pytest

os.environ.setdefault("MODEL_DECK_DATA_DIR",
                      tempfile.mkdtemp(prefix="model-deck-collect-"))


@pytest.fixture(autouse=True)
def _model_deck_default_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DECK_DATA_DIR", str(tmp_path / "default-data"))
