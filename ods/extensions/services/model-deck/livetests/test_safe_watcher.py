"""S11–S12: the arbiter acts UNAIDED — idle release and comfy free, with the
free-spam regression tripwire. TTLs are lowered via the policy API and restored
by policy_guard's finalizer."""

import time

import pytest

from clients import EXTRA
from conftest import TICK, wait_bound
from test_safe_lemonade import load_drill

pytestmark = pytest.mark.safe

SHORT_TTL = 6


def test_s11_watcher_idle_releases_lemonade(deck, lemonade_direct, lemonade_guard,
                                            policy_guard, drill_model, events):
    load_drill(deck, drill_model)
    deck.put("/api/policy", json={
        "lemonade": dict(policy_guard["lemonade"], idle_ttl=SHORT_TTL, pinned=False),
    }).raise_for_status()

    event = events.expect("unload_lemonade", timeout=wait_bound(SHORT_TTL))
    assert event["detail"]["model"] == f"{EXTRA}{drill_model}"

    deadline = time.monotonic() + 3 * TICK
    while time.monotonic() < deadline and lemonade_direct.loaded() is not None:
        time.sleep(1)
    assert lemonade_direct.loaded() is None, "event logged but engine still has the model"


def test_s12_watcher_frees_comfy_once_not_every_tick(deck, policy_guard, events,
                                                     require_comfy_idle):
    deck.put("/api/policy", json={
        "comfyui": dict(policy_guard["comfyui"], idle_ttl=SHORT_TTL, pinned=False),
    }).raise_for_status()

    events.expect("free_comfyui", timeout=wait_bound(SHORT_TTL))

    # Regression 6e6c3f26: without the idle-clock re-arm, the watcher re-emits
    # free_comfyui EVERY 2 s tick. With the fix, a repeat is legitimate only
    # after another full (lowered) TTL — so spacing below TTL-1 is the bug.
    time.sleep(2 * SHORT_TTL)
    assert events.min_spacing("free_comfyui") >= SHORT_TTL - 1, \
        "free_comfyui spam regression: frees closer together than the idle TTL"
