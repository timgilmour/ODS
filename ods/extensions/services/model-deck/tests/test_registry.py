"""Tests for app.registry — the footprint registry.

Cold-start estimates come from on-disk GGUF file size (x1.2 headroom).
The measured-actuals half (observe()/registry.json) was deleted (ruling #4a)
since it had zero production callers. Builds real files under tmp_path (gguf_dir)
rather than touching any real host paths.
"""

import os

import pytest

from app.registry import HIPFIRE_FOOTPRINT, Registry


def _make_gguf(gguf_dir, name: str, size: int) -> None:
    """Create <gguf_dir>/<name> truncated to exactly `size` bytes (sparse, fast)."""
    gguf_dir.mkdir(parents=True, exist_ok=True)
    path = gguf_dir / name
    path.touch()
    os.truncate(path, size)


def test_constants_exact_values():
    assert HIPFIRE_FOOTPRINT == 33_000_000_000


def test_estimate_is_file_size_times_1_2(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model-a.gguf", 1_000_000_000)
    registry = Registry(gguf_dir)

    assert registry.estimate("model-a.gguf") == 1_200_000_000


def test_estimate_raises_filenotfounderror_when_gguf_missing(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    gguf_dir.mkdir()
    registry = Registry(gguf_dir)

    with pytest.raises(FileNotFoundError):
        registry.estimate("nope.gguf")


def test_footprint_is_the_size_estimate(tmp_path):
    """footprint() has exactly one source now: the on-disk size x1.2 estimate.
    The measured-actuals half (observe()/registry.json) was deleted (ruling
    #4a) — it had zero production callers, so this is what footprint() always
    returned in every real run anyway."""
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "m.gguf").write_bytes(b"x" * 1000)
    registry = Registry(gguf_dir)
    assert registry.footprint("m.gguf") == 1200



def test_scan_lists_gguf_files_sorted_with_size_and_footprint(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "zeta.gguf", 100)
    _make_gguf(gguf_dir, "alpha.gguf", 200)
    registry = Registry(gguf_dir)

    result = registry.scan()

    assert result == [
        {"file": "alpha.gguf", "size": 200, "footprint": 240},
        {"file": "zeta.gguf", "size": 100, "footprint": 120},
    ]


def test_scan_is_non_recursive(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "top.gguf", 100)
    _make_gguf(gguf_dir / "subdir", "nested.gguf", 100)
    registry = Registry(gguf_dir)

    result = registry.scan()

    assert [entry["file"] for entry in result] == ["top.gguf"]


def test_scan_ignores_non_gguf_files(tmp_path):
    gguf_dir = tmp_path / "gguf-store"
    _make_gguf(gguf_dir, "model.gguf", 100)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    (gguf_dir / "readme.txt").write_text("not a model")

    registry = Registry(gguf_dir)

    result = registry.scan()

    assert [entry["file"] for entry in result] == ["model.gguf"]


def test_scan_returns_empty_list_when_gguf_dir_missing(tmp_path):
    gguf_dir = tmp_path / "does-not-exist"
    registry = Registry(gguf_dir)

    assert registry.scan() == []
