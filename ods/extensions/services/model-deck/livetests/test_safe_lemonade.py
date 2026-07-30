"""S7–S10: manual load/unload with engine + sysfs ground truth, plus the
instant-evict and heal-suppression regression tripwires."""

import time

import pytest

from clients import EXTRA, read_vram
from conftest import TICK

pytestmark = pytest.mark.safe

LEMONADE_GPU = 1  # read_gpus filtered index; hipfire owns 0 (settings.py)


def load_drill(deck, drill_model) -> None:
    deck.post("/api/tenants/lemonade/load",
              json={"model": drill_model}, timeout=240.0).raise_for_status()


def test_s7_s8_s9_load_survive_unload(deck, lemonade_direct, lemonade_guard, drill_model):
    # The guard restores whatever was loaded before; start from empty so the
    # VRAM deltas below are attributable to the drill model alone.
    if lemonade_direct.loaded() is not None:
        deck.post("/api/tenants/lemonade/unload", json={"model": None}).raise_for_status()
        time.sleep(2)
    _, used_before = read_vram(LEMONADE_GPU)

    # S7 — load: the engine itself must confirm, and VRAM must rise.
    load_drill(deck, drill_model)
    assert lemonade_direct.loaded() == f"{EXTRA}{drill_model}"
    _, used_loaded = read_vram(LEMONADE_GPU)
    assert used_loaded - used_before > 500 * 1024**2, "VRAM did not rise on load"

    # S8 — no instant evict (d46af040): survive 5 watcher ticks on default TTLs.
    time.sleep(5 * TICK + 1)
    assert lemonade_direct.loaded() == f"{EXTRA}{drill_model}", \
        "model evicted within 5 ticks — stale-idle-clock regression"

    # S9 — unload: engine confirms, VRAM returns.
    deck.post("/api/tenants/lemonade/unload", json={"model": None}).raise_for_status()
    assert lemonade_direct.loaded() is None
    time.sleep(2)
    _, used_after = read_vram(LEMONADE_GPU)
    assert used_loaded - used_after > 500 * 1024**2, "VRAM did not drop on unload"


def test_s10_deliberate_unload_is_not_healed_back(deck, lemonade_direct, lemonade_guard,
                                                  drill_model, events):
    state = deck.get("/api/state").json()
    route = state["world"]["default_route"]
    footprint = next((m["footprint"] for m in state["models"]
                      if route and m["file"] == route.removeprefix(EXTRA)), None)
    if footprint is None:
        pytest.skip("vacuous: default route is unset or not a store GGUF")

    load_drill(deck, drill_model)          # ensure something is loaded to unload
    deck.post("/api/tenants/lemonade/unload", json={"model": None}).raise_for_status()

    gpu = next(g for g in deck.get("/api/state").json()["world"]["gpus"]
               if g["index"] == LEMONADE_GPU)
    if gpu["free"] >= footprint:
        pytest.skip("vacuous: route model fits free VRAM — no pending load to suppress")

    # Suppression must hold: no auto-reload for at least 10 ticks.
    events.expect_absent("load-retriggered", window=10 * TICK)
    assert lemonade_direct.loaded() is None
