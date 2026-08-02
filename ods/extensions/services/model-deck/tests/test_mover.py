"""Tests for app.mover.Mover — file mechanics, verify-before-delete, janitor."""
import pytest

from app.mover import Mover, MoveCancelled, MoveVerifyError


def _mk(tmp_path, name="src.gguf", size=3 * 1024 * 1024):
    src_dir = tmp_path / "a"; src_dir.mkdir(exist_ok=True)
    dst_dir = tmp_path / "b"; dst_dir.mkdir(exist_ok=True)
    src = src_dir / name
    src.write_bytes(b"\x5a" * size)
    return src, dst_dir / name


def _cross_fs_mover():
    return Mover(same_fs=lambda a, b: False)   # force the copy path (see Interfaces)


def test_same_fs_move_is_single_rename(tmp_path):
    src, dst = _mk(tmp_path)
    Mover().execute(src, dst, lambda b: None, lambda: False)   # default: same device
    assert not src.exists() and dst.exists()


def test_move_copies_verifies_and_deletes_source(tmp_path):
    src, dst = _mk(tmp_path)
    data = src.read_bytes()
    progress = []
    _cross_fs_mover().execute(src, dst, progress.append, lambda: False)
    assert not src.exists() and dst.read_bytes() == data
    assert progress and progress[-1] == len(data)
    assert not dst.with_name(dst.name + ".part").exists()


def test_cancel_mid_copy_keeps_source_removes_part(tmp_path):
    src, dst = _mk(tmp_path, size=32 * 1024 * 1024)      # several chunks
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1                             # cancel on 2nd chunk

    with pytest.raises(MoveCancelled):
        _cross_fs_mover().execute(src, dst, lambda b: None, cancel)
    assert src.exists() and not dst.exists()
    assert not dst.with_name(dst.name + ".part").exists()


def test_verify_mismatch_keeps_source(tmp_path, monkeypatch):
    src, dst = _mk(tmp_path)
    import app.mover as mover_mod
    real = mover_mod._hash_file
    monkeypatch.setattr(mover_mod, "_hash_file", lambda p: "corrupted" + real(p))
    with pytest.raises(MoveVerifyError):
        _cross_fs_mover().execute(src, dst, lambda b: None, lambda: False)
    assert src.exists() and not dst.exists()


def test_janitor_removes_orphans(tmp_path):
    root = tmp_path / "store"; root.mkdir()
    orphan = root / "x.gguf.part"; orphan.write_bytes(b"junk")
    staging = root / "repo.deck-staging"; staging.mkdir()
    (staging / "f").write_bytes(b"junk")
    keep = root / "keep.gguf"; keep.write_bytes(b"real")
    removed = Mover().janitor([root])
    assert orphan in removed and staging in removed
    assert keep.exists() and not orphan.exists() and not staging.exists()


def test_tree_move_is_atomic_unit(tmp_path):
    src_root = tmp_path / "a"; src_root.mkdir()
    repo = src_root / "Qwen"; (repo / "sub").mkdir(parents=True)
    (repo / "config.json").write_bytes(b"{}")
    (repo / "sub" / "model-00001.safetensors").write_bytes(b"w" * (2 * 1024 * 1024))
    dst = tmp_path / "b" / "Qwen"
    progress = []
    _cross_fs_mover().execute(repo, dst, progress.append, lambda: False)
    assert not repo.exists()
    assert (dst / "config.json").exists() and (dst / "sub" / "model-00001.safetensors").exists()
    assert progress[-1] == 2 + 2 * 1024 * 1024
    assert not dst.with_name(dst.name + ".deck-staging").exists()


def test_tree_cancel_keeps_source_removes_staging(tmp_path):
    src_root = tmp_path / "a"; src_root.mkdir()
    repo = src_root / "Qwen"; repo.mkdir()
    (repo / "f1").write_bytes(b"x" * (16 * 1024 * 1024))
    (repo / "f2").write_bytes(b"y" * (16 * 1024 * 1024))
    dst = tmp_path / "b" / "Qwen"
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(MoveCancelled):
        _cross_fs_mover().execute(repo, dst, lambda b: None, cancel)
    assert (repo / "f1").exists() and (repo / "f2").exists()
    assert not dst.exists() and not dst.with_name(dst.name + ".deck-staging").exists()
