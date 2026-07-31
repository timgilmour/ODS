"""D5: deck-driven spark swap round-trip — vllm profile -> comfyui -> vllm profile.

Verifies the node-agent's dict-shaped profile listing (per-profile engine
metadata) and the deck's engine-aware busy guard survive a real round trip:
swapping the Spark's single serving slot to ComfyUI and back must not 502 on
the return leg, and the comfyui profile must self-report engine="comfyui".
"""

import time

import pytest

pytestmark = pytest.mark.disruptive

# comfyui boot is fast; the vllm boot-back can autotune ~13-15 min
# (FlashInfer JIT cache is not volume-mounted, see sparky-operations.md).
SWAP_TIMEOUT = 900


def _status(deck):
    return deck.get("/api/spark/status").json()


def _swap(deck, profile):
    r = deck.post("/api/spark/swap", json={"profile": profile})
    r.raise_for_status()
    return r.json()


def _wait_serving(deck, model, timeout_s=SWAP_TIMEOUT):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = _status(deck)
        if s["serving"]["endpoint_ok"] and s["serving"]["model"] == model:
            return s
        time.sleep(10)
    pytest.fail(f"{model} not serving within {timeout_s}s: {_status(deck)}")


def test_d5_spark_comfyui_round_trip(deck):
    start = _status(deck)
    assert any(p["name"] == "comfyui" for p in start["profiles"]), \
        "comfyui profile missing from node-agent listing"
    original = start["serving"]["model"] or "heretic"

    _swap(deck, "comfyui")
    s = _wait_serving(deck, "comfyui")
    comfy = next(p for p in s["profiles"] if p["name"] == "comfyui")
    assert comfy["engine"] == "comfyui"

    _swap(deck, original)           # engine-aware busy guard must NOT 502 here
    _wait_serving(deck, original)
