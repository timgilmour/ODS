"""
Model Deck mover — the imperative byte-moving shell.

THE INVARIANT (spec #12): there is no reachable state where a model exists
zero times. Order of operations for a cross-filesystem move:
copy → fsync → re-read+verify sha256 → atomic rename into place → fsync
parent dir → ONLY THEN delete source. A crash or cancel at any phase leaves
the source untouched; the startup janitor removes orphaned ``*.part`` files
and ``*.deck-staging`` dirs.

Same-filesystem moves are a single ``os.replace`` — instant, no copy.

Total cross-fs I/O: read source once (hash computed in-flight with the
copy — zero extra reads), write dest once, read dest once for the verify.
"""

import hashlib
import os
import shutil
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
PART_SUFFIX = ".part"
STAGING_SUFFIX = ".deck-staging"


class MoveCancelled(Exception):
    """Operator cancelled the move; source untouched, partial dest removed."""


class MoveVerifyError(Exception):
    """Destination hash mismatched the in-flight source hash; source kept."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tree_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class Mover:
    """Executes one move at a time (serialization is JobQueue's job).

    `same_fs` is an injectable predicate (default: st_dev comparison) so
    tests can force the cross-filesystem copy path — tmp dirs share a
    device, and a rename fast path would silently skip verify coverage."""

    def __init__(self, same_fs=None):
        self._same_fs = same_fs or (
            lambda src, dst_parent: src.stat().st_dev == dst_parent.stat().st_dev)

    def execute(self, src: Path, dst: Path, progress_cb, cancel_check) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self._same_fs(src, dst.parent):
            size = src.stat().st_size if src.is_file() else _tree_size_bytes(src)
            os.replace(src, dst)          # same fs: instant, atomic — no copy
            progress_cb(size)
            return
        if src.is_dir():
            self._move_tree(src, dst, progress_cb, cancel_check)
        else:
            self._move_file(src, dst, progress_cb, cancel_check)

    # -- single file ---------------------------------------------------------

    def _copy_verify(self, src: Path, dst_final: Path, progress_cb, cancel_check,
                     bytes_offset: int = 0) -> int:
        """Copy src → <dst_final>.part with in-flight sha256, fsync, re-read
        verify, atomic rename to dst_final. Returns bytes written. Raises
        MoveCancelled/MoveVerifyError with the part file already removed."""
        part = dst_final.with_name(dst_final.name + PART_SUFFIX)
        digest = hashlib.sha256()
        done = bytes_offset
        try:
            with src.open("rb") as fin, part.open("wb") as fout:
                while chunk := fin.read(CHUNK_BYTES):
                    if cancel_check():
                        raise MoveCancelled(str(src))
                    fout.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    progress_cb(done)
                fout.flush()
                os.fsync(fout.fileno())
            if _hash_file(part) != digest.hexdigest():
                raise MoveVerifyError(f"verify failed for {dst_final.name}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, dst_final)
        _fsync_dir(dst_final.parent)
        return done

    def _move_file(self, src: Path, dst: Path, progress_cb, cancel_check) -> None:
        self._copy_verify(src, dst, progress_cb, cancel_check)
        src.unlink()                      # the LAST step, always (invariant 12)

    # -- trees: Task 6 -------------------------------------------------------

    def _move_tree(self, src: Path, dst: Path, progress_cb, cancel_check) -> None:
        raise NotImplementedError  # implemented in the next task

    # -- janitor ---------------------------------------------------------------

    def janitor(self, roots: list[Path]) -> list[Path]:
        removed: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for p in root.rglob(f"*{PART_SUFFIX}"):
                p.unlink(missing_ok=True)
                removed.append(p)
            for p in root.rglob(f"*{STAGING_SUFFIX}"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(p)
        return removed
