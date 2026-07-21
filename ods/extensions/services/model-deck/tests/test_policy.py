"""Tests for app.policy — the tenant policy store.

policy.json is a flat mapping of {tenant: {priority, pinned, idle_ttl}} for
the three known tenants (hipfire, lemonade, comfyui). Defaults materialize
and persist on first get() (missing or corrupt file self-heals to defaults).
Writes are atomic (temp file + os.replace), matching app.registry's idiom.
"""

import json

import pytest

from app.policy import DEFAULT_POLICIES, PolicyStore


def test_defaults_exact_values():
    assert DEFAULT_POLICIES == {
        "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
        "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
        "comfyui": {"priority": 40, "pinned": False, "idle_ttl": 300},
    }


def test_get_materializes_defaults_when_file_missing(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)

    result = store.get()

    assert result == DEFAULT_POLICIES


def test_get_persists_defaults_to_disk_on_first_read(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)

    store.get()

    assert json.loads(policy_path.read_text()) == DEFAULT_POLICIES


def test_get_parent_dir_created_on_first_read(tmp_path):
    policy_path = tmp_path / "nested" / "does" / "not" / "exist" / "policy.json"
    store = PolicyStore(policy_path)

    store.get()

    assert policy_path.is_file()


def test_get_does_not_rewrite_file_once_it_exists(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)
    store.get()
    custom = {"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5}}
    policy_path.write_text(json.dumps(custom))

    result = PolicyStore(policy_path).get()

    assert result == custom


def test_put_then_get_roundtrips_across_instances(tmp_path):
    policy_path = tmp_path / "policy.json"
    first = PolicyStore(policy_path)
    first.get()

    first.put({"hipfire": {"priority": 200, "pinned": False, "idle_ttl": 60}})

    second = PolicyStore(policy_path)
    result = second.get()

    assert result["hipfire"] == {"priority": 200, "pinned": False, "idle_ttl": 60}


def test_put_is_partial_update_other_tenants_keep_current_values(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)
    store.get()

    store.put({"lemonade": {"priority": 77, "pinned": True, "idle_ttl": 42}})

    result = store.get()
    assert result["lemonade"] == {"priority": 77, "pinned": True, "idle_ttl": 42}
    assert result["hipfire"] == DEFAULT_POLICIES["hipfire"]
    assert result["comfyui"] == DEFAULT_POLICIES["comfyui"]


def test_put_write_is_atomic_no_temp_files_left_behind(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)
    store.get()

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    leftovers = list(policy_path.parent.glob("*.tmp"))
    assert leftovers == []


def test_put_parent_dir_created_when_file_never_read(tmp_path):
    policy_path = tmp_path / "nested" / "policy.json"
    store = PolicyStore(policy_path)

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    assert policy_path.is_file()


def test_put_rejects_unknown_tenant_naming_it(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError, match="sdxl"):
        store.put({"sdxl": {"priority": 1, "pinned": False, "idle_ttl": 5}})


def test_put_rejects_missing_field(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False}})


def test_put_rejects_extra_field(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put(
            {"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5, "extra": 1}}
        )


def test_put_rejects_priority_bool_instead_of_int(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": True, "pinned": False, "idle_ttl": 5}})


def test_put_rejects_priority_wrong_type(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": "high", "pinned": False, "idle_ttl": 5}})


def test_put_rejects_pinned_wrong_type(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": "yes", "idle_ttl": 5}})


def test_put_rejects_idle_ttl_wrong_type(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": "5"}})


def test_put_rejects_idle_ttl_bool(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": True}})


def test_put_rejects_negative_idle_ttl(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": -1}})


def test_put_accepts_idle_ttl_zero(tmp_path):
    store = PolicyStore(tmp_path / "policy.json")

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 0}})

    assert store.get()["hipfire"]["idle_ttl"] == 0


def test_put_rejected_payload_leaves_file_unchanged(tmp_path):
    policy_path = tmp_path / "policy.json"
    store = PolicyStore(policy_path)
    before = store.get()

    with pytest.raises(ValueError):
        store.put({"sdxl": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    assert store.get() == before


def test_corrupt_policy_json_treated_as_defaults(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not valid json")
    store = PolicyStore(policy_path)

    assert store.get() == DEFAULT_POLICIES


def test_corrupt_policy_json_self_heals_on_get(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not valid json")
    store = PolicyStore(policy_path)

    store.get()

    assert json.loads(policy_path.read_text()) == DEFAULT_POLICIES


def test_non_dict_json_treated_as_defaults(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(["not", "a", "dict"]))
    store = PolicyStore(policy_path)

    assert store.get() == DEFAULT_POLICIES
