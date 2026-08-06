"""S5–S6: policy round-trip and validation rejections (422 + file untouched)."""

import pytest

pytestmark = pytest.mark.safe


def test_s5_policy_partial_put_roundtrips(deck, policy_guard):
    override = {"lemonade": dict(policy_guard["lemonade"], idle_ttl=1234)}
    body = deck.put("/api/policy", json=override).json()
    assert body["lemonade"]["idle_ttl"] == 1234
    # tenants not named stay untouched
    assert body["comfyui"] == policy_guard["comfyui"]
    assert body["hipfire"] == policy_guard["hipfire"]


@pytest.mark.parametrize("bad", [
    # Reserved config key — NOT a tenant. Arbitrary tenant names are ACCEPTED
    # since 1ee64611 (runtime tenants); the old unknown-tenant case would be
    # a valid write now, and one policy_guard cannot undo (put() merges).
    {"_auto": {"priority": 1, "pinned": False, "idle_ttl": 0}},
    {"lemonade": {"priority": True, "pinned": False, "idle_ttl": 0}}, # bool priority
    {"lemonade": {"priority": 1, "pinned": False, "idle_ttl": -5}},   # negative ttl
    {"lemonade": {"priority": 1, "pinned": False}},                   # missing field
    {"lemonade": {"priority": 1, "pinned": False, "idle_ttl": 0, "x": 1}},  # extra field
])
def test_s6_policy_validation_rejects_and_leaves_file_untouched(deck, policy_guard, bad):
    before = deck.get("/api/policy").json()
    resp = deck.put("/api/policy", json=bad)
    assert resp.status_code == 422, resp.text
    assert deck.get("/api/policy").json() == before
