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


class HandBuiltRegistry:
    """A registry state built BY HAND, past app.node_store's write gate.

    Task 5 of the sglang-omni plan refuses declaring any engine kind on a
    node-agent entry unless that kind is ``remote_capable``, and
    ``NodeStore._heal_engines`` strips a refused list back to ``[]`` on
    every load. Task 7 made ONE kind remote-capable (sglang-omni), so a real
    ``NodeStore`` can now hold a remote declaration of THAT kind — but the
    three E1 kinds are still refused there, and most remote tests are
    deliberately written against a lemonade-kind remote engine (fixture
    discipline: the kind under test must not be the only kind that could
    possibly work). Weakening the gate to make testing easier is exactly the
    shortcut this branch forbids, so those tests stand the registry shape up
    at the store BOUNDARY instead, exposing only the methods those paths
    actually call (``list``/``get`` + the three credential accessors), plus
    two test-side mutators for the registry edits the lazy clients must
    notice. A test that wants the REAL write gate (tests/test_api.py's
    remote sglang-omni declarations) simply uses NodeStore directly now.

    Lives here rather than in one test module because four modules need it
    (observe/state/node_clients/api wiring); a conftest fixture is
    available to all of them with no cross-test-module import.
    """

    def __init__(self, entries, credentials=None) -> None:
        self._entries = [dict(e) for e in entries]
        self._credentials = dict(credentials or {})

    def list(self) -> list[dict]:
        return [dict(e) for e in self._entries]

    def get(self, node_id: str) -> dict | None:
        for entry in self._entries:
            if entry["id"] == node_id:
                return dict(entry)
        return None

    def credential_set(self, node_id: str) -> bool:
        return bool(self._credentials.get(node_id))

    def credential_for(self, node_id: str) -> str | None:
        return self._credentials.get(node_id)

    def credential_fingerprint(self, node_id: str) -> str | None:
        credential = self._credentials.get(node_id)
        return None if not credential else f"fp:{credential}"

    # --- test-side mutation (the edits self-healing clients must notice) ---

    def patch(self, node_id: str, **fields) -> None:
        for entry in self._entries:
            if entry["id"] == node_id:
                entry.update(fields)
                return
        raise KeyError(node_id)

    def set_credential(self, node_id: str, credential: str | None) -> None:
        if credential is None:
            self._credentials.pop(node_id, None)
        else:
            self._credentials[node_id] = credential


@pytest.fixture
def hand_built_registry():
    """The ``HandBuiltRegistry`` class itself — call it with
    ``(entries, credentials)``. See that class's docstring for why the
    remote tests cannot use a real ``NodeStore``."""
    return HandBuiltRegistry
