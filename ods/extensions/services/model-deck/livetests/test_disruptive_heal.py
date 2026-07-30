"""D3: the 2026-07-21 incident, replayed automatically. Lemonade loses its model
OUTSIDE the deck (direct engine unload — no suppression arms), comfy fills GPU 1,
and the arbiter must recover UNAIDED. The expected recovery branch is computed
from measured VRAM vs the route model's footprint, so the drill is honest even
when the fill doesn't reach contention."""

import json
import time
from pathlib import Path

import pytest

from clients import EXTRA, read_vram
from conftest import wait_bound
from test_safe_lemonade import LEMONADE_GPU

pytestmark = pytest.mark.disruptive

ASSET = Path(__file__).parent / "assets" / "comfy_fill_api.json"
HEAL_TIMEOUT = 300  # free + 27B reload can take a few minutes


def test_d3_contention_heal_replay(deck, lemonade_direct, comfy_direct,
                                   litellm_direct, events, lemonade_guard):
    # lemonade_guard: the drill deliberately ends with the healed route model
    # resident; the guard returns lemonade to its pre-drill state so the
    # session bookend's left-as-found holds.
    state = deck.get("/api/state").json()
    route = state["world"]["default_route"]
    if not route:
        pytest.skip("no default route — nothing for healing to restore")
    footprint = next((m["footprint"] for m in state["models"]
                      if m["file"] == route.removeprefix(EXTRA)), None)
    if footprint is None:
        pytest.skip("default route is not a store GGUF")
    if comfy_direct.queue_len() != 0:
        pytest.skip("comfy busy — will not interleave with a real generation")

    # Ensure the route model is resident via the DECK (clears any suppression
    # window a previous drill armed — makes D3 self-sufficient), then lose it
    # at the ENGINE (no suppression arms: this is the incident shape).
    deck.post("/api/tenants/lemonade/load",
              json={"model": route.removeprefix(EXTRA)}, timeout=700.0).raise_for_status()
    lemonade_direct.unload(route)
    assert lemonade_direct.loaded() is None

    # Fill GPU 1: run the two-checkpoint graph and let the weights stay cached.
    comfy_direct.submit(json.loads(ASSET.read_text()))
    comfy_direct.wait_idle(timeout=240)

    total, used = read_vram(LEMONADE_GPU)
    free = total - used
    expect_eviction = free < footprint

    if expect_eviction:
        events.expect("free_comfyui", timeout=wait_bound(10) + 60)
    events.expect("load-retriggered", timeout=HEAL_TIMEOUT)

    deadline = time.monotonic() + wait_bound(10)
    while time.monotonic() < deadline and lemonade_direct.loaded() != route:
        time.sleep(2)
    assert lemonade_direct.loaded() == route, "healed load did not stick"
    assert litellm_direct.completion("default"), "route serves after heal"

    if not expect_eviction:
        pytest.xfail("heal verified on the fits-branch only — fill did not reach "
                     f"contention (free {free // 1024**3} GiB ≥ footprint "
                     f"{footprint // 1024**3} GiB)")
