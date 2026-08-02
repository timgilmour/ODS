"""Tests for app.locations.LocationStore — locations.json + marker files."""
import json
import pytest

from app.engines import GuardError
from app.locations import MARKER_NAME, LocationStore


def _spec(root, **over):
    spec = {"name": "cold", "path": str(root), "role": "cold", "store_type": "gguf",
            "engine": "none", "watermark_gb": None, "archive_to": None, "readonly": False}
    spec.update(over)
    return spec


def _store(tmp_path, free=100 * 10**9, total=200 * 10**9):
    return LocationStore(tmp_path / "locations.json",
                         disk_usage=lambda p: type("u", (), {"total": total, "used": total - free, "free": free})())


def test_register_writes_marker_and_persists(tmp_path):
    root = tmp_path / "drive"; root.mkdir()
    store = _store(tmp_path)
    loc = store.register(_spec(root))
    assert loc["uuid"]
    marker = json.loads((root / MARKER_NAME).read_text())
    assert marker == {"uuid": loc["uuid"], "name": "cold"}
    assert store.get("cold")["path"] == str(root)


def test_register_missing_path_is_guard_error(tmp_path):
    with pytest.raises(GuardError):
        _store(tmp_path).register(_spec(tmp_path / "not-mounted"))


def test_register_validates_fields(tmp_path):
    root = tmp_path / "d"; root.mkdir()
    store = _store(tmp_path)
    for bad in [_spec(root, role="warm"), _spec(root, store_type="zip"),
                _spec(root, engine="vllm"), _spec(root, name="Has Spaces"),
                _spec(root, engine="lemonade", store_type="hf"),      # engine/type pairing
                _spec(root, engine="comfyui", store_type="gguf")]:
        with pytest.raises(ValueError):
            store.register(bad)


def test_missing_marker_means_unavailable_not_empty(tmp_path):
    root = tmp_path / "drive"; root.mkdir()
    store = _store(tmp_path)
    store.register(_spec(root))
    (root / MARKER_NAME).unlink()          # simulates unmounted drive / bare mountpoint
    desc = store.describe()[0]
    assert desc["available"] is False
    assert desc["free_bytes"] is None      # never report space for a phantom mount


def test_wrong_uuid_marker_means_unavailable(tmp_path):
    root = tmp_path / "drive"; root.mkdir()
    store = _store(tmp_path)
    store.register(_spec(root))
    (root / MARKER_NAME).write_text(json.dumps({"uuid": "someone-else", "name": "cold"}))
    assert store.describe()[0]["available"] is False


def test_describe_reports_free_space_when_available(tmp_path):
    root = tmp_path / "drive"; root.mkdir()
    store = _store(tmp_path, free=42)
    store.register(_spec(root))
    desc = store.describe()[0]
    assert desc["available"] is True and desc["free_bytes"] == 42


def test_update_patches_only_allowed_fields(tmp_path):
    root = tmp_path / "d"; root.mkdir()
    store = _store(tmp_path)
    store.register(_spec(root))
    updated = store.update("cold", {"watermark_gb": 50.0, "archive_to": "cold"})
    assert updated["watermark_gb"] == 50.0
    with pytest.raises(ValueError):
        store.update("cold", {"path": "/elsewhere"})
    with pytest.raises(ValueError):
        store.update("nope", {"readonly": True})


def test_deregister_leaves_files(tmp_path):
    root = tmp_path / "d"; root.mkdir()
    store = _store(tmp_path)
    store.register(_spec(root))
    store.deregister("cold")
    assert store.list() == [] and (root / MARKER_NAME).exists()
