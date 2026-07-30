"""S1–S4: liveness, state shape, events, registry ground truth."""

import pytest

from conftest import GGUF_STORE

pytestmark = pytest.mark.safe


def test_s1_health_and_state_shape(deck):
    assert deck.get("/health").json() == {"status": "ok"}

    state = deck.get("/api/state")
    assert state.status_code == 200
    world = state.json()["world"]
    gpus = world["gpus"]
    assert len(gpus) == 2, f"expected the two R9700s, got {gpus}"
    for gpu in gpus:
        assert gpu["total"] > 20 * 1024**3  # 32 GiB cards, sanity floor
        assert 0 <= gpu["used"] <= gpu["total"]
    assert set(world["tenants"]) == {"lemonade", "comfyui", "hipfire"}
    assert "default_route" in world


def test_s2_state_is_fresh_not_cached(deck, policy_guard):
    mutated = {"comfyui": dict(policy_guard["comfyui"], idle_ttl=4321)}
    deck.put("/api/policy", json=mutated).raise_for_status()
    seen = deck.get("/api/state").json()["policy"]["comfyui"]["idle_ttl"]
    assert seen == 4321  # policy_guard restores afterwards


def test_s3_events_tail_parses(deck):
    events = deck.get("/api/events", params={"n": 50}).json()["events"]
    for e in events:
        assert {"ts", "kind", "detail"} <= set(e)


def test_s4_registry_matches_gguf_store(deck):
    models = deck.get("/api/state").json()["models"]
    listed = {m["file"] for m in models}
    on_disk = {p.name for p in GGUF_STORE.glob("*.gguf")}
    assert listed == on_disk
    # Unmeasured footprints are the size×1.2 estimate (registry.observe is unwired in v1,
    # so measured entries are possible only if something wrote registry.json by hand).
    for m in models:
        size = (GGUF_STORE / m["file"]).stat().st_size
        estimate = int(size * 1.2)
        assert m["footprint"] >= size, m
        if m["footprint"] != estimate:  # tolerated: a measured value overrides
            assert 0 < m["footprint"] < 2 * size, m
