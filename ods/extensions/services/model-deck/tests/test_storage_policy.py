"""Tests for app.policy.StoragePolicyStore — storage_policy.json owner."""
from app.policy import StoragePolicyStore


def test_get_materializes_default_on_missing_file(tmp_path):
    store = StoragePolicyStore(tmp_path / "storage_policy.json")
    assert store.get() == {"auto": False}
    assert (tmp_path / "storage_policy.json").exists()  # persisted, not just returned


def test_put_then_get_roundtrip(tmp_path):
    store = StoragePolicyStore(tmp_path / "storage_policy.json")
    store.put({"auto": True})
    assert store.get() == {"auto": True}


def test_corrupt_file_self_heals_to_default(tmp_path):
    path = tmp_path / "storage_policy.json"
    path.write_text("{not json")
    assert StoragePolicyStore(path).get() == {"auto": False}


def test_put_rejects_bad_payloads(tmp_path):
    import pytest
    store = StoragePolicyStore(tmp_path / "storage_policy.json")
    for bad in [{}, {"auto": 1}, {"auto": True, "extra": 1}, {"a": True}]:
        with pytest.raises(ValueError):
            store.put(bad)
