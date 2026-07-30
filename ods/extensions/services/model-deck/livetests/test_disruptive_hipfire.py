"""D1: hipfire park/resume round-trip with force semantics.
Declared-window only: the litellm probe deliberately evicts hipfire's
single-slot conversation cache."""

import time

import pytest

from clients import hipfire_health_status, read_vram
from conftest import HIPFIRE_URL, TICK

pytestmark = pytest.mark.disruptive

HIPFIRE_GPU = 0
RESUME_TIMEOUT = 300


@pytest.fixture
def hipfire_window(deck, litellm_direct):
    """Window preconditions for touching hipfire; ensures it ends running."""
    hip = deck.get("/api/state").json()["world"]["tenants"]["hipfire"]
    if hip["state"] not in ("running", "loading"):
        pytest.skip(f"hipfire is {hip['state']} — window expects it serving")
    route = litellm_direct.route_table().get("default", "")
    if hip["model"] and route.endswith(hip["model"]):
        pytest.skip("default route targets hipfire — D1 must not park the default route")
    yield
    deck.post("/api/tenants/hipfire/resume")
    _wait_running(deck)


def _wait_running(deck, timeout=RESUME_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["state"] == "running":
            return
        time.sleep(5)
    raise AssertionError(f"hipfire not running within {timeout}s of resume")


def test_d1_park_resume_roundtrip_with_force_semantics(deck, litellm_direct, hipfire_window):
    # Warm the conversation tracker with a real served request (window-only act).
    assert litellm_direct.completion("hipfire")
    time.sleep(3 * TICK)  # ≥1 watcher tick so its stats poll records the activity

    # Warm ⇒ a no-force park must refuse.
    refused = deck.post("/api/tenants/hipfire/park")
    assert refused.status_code == 409, refused.text
    assert hipfire_health_status(HIPFIRE_URL) == "up"

    # Force ⇒ parks. Ground truth: health socket down, VRAM released.
    _, used_before = read_vram(HIPFIRE_GPU)
    deck.post("/api/tenants/hipfire/park", params={"force": "true"}).raise_for_status()
    assert deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["state"] == "parked"
    assert hipfire_health_status(HIPFIRE_URL) == "down"
    time.sleep(5)
    _, used_parked = read_vram(HIPFIRE_GPU)
    assert used_before - used_parked > 15 * 1024**3, "GPU0 VRAM did not release on park"

    # Resume: back to running, and it really serves again.
    deck.post("/api/tenants/hipfire/resume").raise_for_status()
    _wait_running(deck)
    assert litellm_direct.completion("hipfire")
