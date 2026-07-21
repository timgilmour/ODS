"""Tests for app.registry — the footprint registry.

Cold-start estimates come from on-disk GGUF file size (x1.2 headroom);
measured actuals observed at runtime override the estimate once known.
Builds real files under tmp_path (gguf_dir) and a registry.json path
(also under tmp_path, parent dir intentionally absent until first save)
rather than touching any real host paths.
"""

import json
import os

import pytest

from app.registry import COMFYUI_RESERVE_DEFAULT, HIPFIRE_FOOTPRINT, Registry


def _make_gguf(gguf_dir, name: str, size: int) -> None:
    """Create <gguf_dir>/<name> truncated to exactly `size` bytes (sparse, fast)."""
    gguf_dir.mkdir(parents=True, exist_ok=True)
    path = gguf_dir / name
    path.touch()
    os.truncate(path, size)


def test_constants_exact_values():
    assert HIPFIRE_FOOTPRINT == 33_000_000_000
    assert COMFYUI_RESERVE_DEFAULT == 24_000_000_000


def test_estimate_is_file_size_times_1_2(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 1_000_000_000)
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    assert registry.estimate("model-a.gguf") == 1_200_000_000


def test_estimate_raises_filenotfounderror_when_gguf_missing(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    gguf_dir.mkdir()
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    with pytest.raises(FileNotFoundError):
        registry.estimate("nope.gguf")


def test_footprint_falls_back_to_estimate_when_unobserved(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    assert registry.footprint("model-a.gguf") == 600_000_000


def test_observe_overrides_estimate_in_footprint(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    registry.observe("model-a.gguf", 999_000_000)

    assert registry.footprint("model-a.gguf") == 999_000_000


def test_observe_persists_immediately_to_registry_json(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "registry.json"
    registry = Registry(registry_path, gguf_dir)

    registry.observe("model-a.gguf", 999_000_000)

    assert json.loads(registry_path.read_text()) == {"model-a.gguf": 999_000_000}


def test_observed_value_visible_to_a_second_registry_instance(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "registry.json"
    first = Registry(registry_path, gguf_dir)
    first.observe("model-a.gguf", 999_000_000)

    second = Registry(registry_path, gguf_dir)

    assert second.footprint("model-a.gguf") == 999_000_000


def test_registry_json_parent_dir_created_on_first_save(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "nested" / "does" / "not" / "exist" / "registry.json"
    registry = Registry(registry_path, gguf_dir)

    registry.observe("model-a.gguf", 999_000_000)

    assert registry_path.is_file()
    assert json.loads(registry_path.read_text()) == {"model-a.gguf": 999_000_000}


def test_observe_write_is_atomic_no_temp_files_left_behind(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "registry.json"
    registry = Registry(registry_path, gguf_dir)

    registry.observe("model-a.gguf", 999_000_000)

    leftovers = list(registry_path.parent.glob("*.tmp"))
    assert leftovers == []


def test_corrupt_registry_json_treated_as_empty(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{not valid json")
    registry = Registry(registry_path, gguf_dir)

    assert registry.footprint("model-a.gguf") == 600_000_000


def test_corrupt_registry_json_self_heals_on_next_observe(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 500_000_000)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{not valid json")
    registry = Registry(registry_path, gguf_dir)

    registry.observe("model-a.gguf", 999_000_000)

    assert json.loads(registry_path.read_text()) == {"model-a.gguf": 999_000_000}


def test_scan_lists_gguf_files_sorted_with_size_and_footprint(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "zeta.gguf", 100)
    _make_gguf(gguf_dir, "alpha.gguf", 200)
    registry = Registry(tmp_path / "registry.json", gguf_dir)
    registry.observe("zeta.gguf", 5000)

    result = registry.scan()

    assert result == [
        {"file": "alpha.gguf", "size": 200, "footprint": 240},
        {"file": "zeta.gguf", "size": 100, "footprint": 5000},
    ]


def test_scan_is_non_recursive(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "top.gguf", 100)
    _make_gguf(gguf_dir / "subdir", "nested.gguf", 100)
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    result = registry.scan()

    assert [entry["file"] for entry in result] == ["top.gguf"]


def test_scan_ignores_non_gguf_files(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model.gguf", 100)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    (gguf_dir / "readme.txt").write_text("not a model")

    registry = Registry(tmp_path / "registry.json", gguf_dir)

    result = registry.scan()

    assert [entry["file"] for entry in result] == ["model.gguf"]


def test_scan_returns_empty_list_when_gguf_dir_missing(tmp_path):
    gguf_dir = tmp_path / "does-not-exist"
    registry = Registry(tmp_path / "registry.json", gguf_dir)

    assert registry.scan() == []
