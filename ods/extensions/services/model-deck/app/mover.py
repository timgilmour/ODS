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

import errno
import hashlib
import os
import shutil
import threading
import time
import uuid as uuidlib
from pathlib import Path

from app.events import log_event

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
            try:
                os.replace(src, dst)      # same fs: instant, atomic — no copy
            except OSError as exc:
                # st_dev matching is not a guarantee rename() will work: two
                # bind mounts of one filesystem are separate mount points, and
                # the kernel refuses to rename across them (EXDEV). That's not
                # a failure — it just means "you have to copy". Any other errno
                # is a real error and must surface.
                if exc.errno != errno.EXDEV:
                    raise
            else:
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
        staging = dst.with_name(dst.name + STAGING_SUFFIX)
        if staging.exists():
            shutil.rmtree(staging)
        done = 0
        try:
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                target = staging / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                done = self._copy_verify(f, target, progress_cb, cancel_check,
                                         bytes_offset=done)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        os.replace(staging, dst)
        _fsync_dir(dst.parent)
        shutil.rmtree(src)               # the LAST step (invariant 12)

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


_TERMINAL = frozenset({"done", "failed", "cancelled"})


class JobQueue:
    """One-at-a-time move executor (spec #15: serialize disk I/O; queue the
    rest). Owns the job list; every transition audited to events.jsonl."""

    def __init__(self, mover, catalog, location_store, events_path):
        self._mover = mover
        self._catalog = catalog
        self._locations = location_store
        self._events_path = events_path
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._cancel_flags: dict[str, bool] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public --------------------------------------------------------------

    def submit(self, plan: dict, label: str, on_success=None) -> dict:
        job = {"id": uuidlib.uuid4().hex[:12], "unit_id": plan["unit_id"],
               "from": plan["src_location"], "to": plan["dest_location"],
               "label": label, "state": "queued", "bytes_done": 0,
               "bytes_total": plan["bytes"], "error": None,
               "created_ts": time.time()}
        # Set catalog state BEFORE job becomes visible to worker thread
        self._catalog.set_state(plan["unit_id"], "moving")
        with self._lock:
            self._jobs[job["id"]] = job
            self._pending.append({**job, "_on_success": on_success})
            self._wake.set()
        return dict(job)

    def jobs(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(),
                                            key=lambda j: j["created_ts"], reverse=True)]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in _TERMINAL:
                return False
            if job["state"] == "queued":
                self._pending = [p for p in self._pending if p["id"] != job_id]
                job["state"] = "cancelled"
                unit_id = job["unit_id"]
            else:
                self._cancel_flags[job_id] = True
                return True
        self._catalog.set_state(unit_id, "resident")
        self._log("move_cancelled", job)
        return True

    def active(self) -> bool:
        with self._lock:
            return any(j["state"] not in _TERMINAL for j in self._jobs.values())

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="model-deck-mover", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    # -- worker ----------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pending = self._pending.pop(0) if self._pending else None
            if pending is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            self._process(pending)

    def _resolve(self, location_name: str) -> dict:
        loc = self._locations.get(location_name)
        if loc is None or not self._locations.available(loc):
            raise RuntimeError(f"location {location_name!r} is unavailable")
        return loc

    def _process(self, pending: dict) -> None:
        job_id = pending["id"]
        on_success = pending.get("_on_success")
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] == "cancelled":
                return
            job["state"] = "copying"
        self._log("move_started", job)
        unit_id = job["unit_id"]
        try:
            unit = self._catalog.get(unit_id)
            if unit is None:
                raise RuntimeError(f"unit {unit_id!r} vanished from catalog")
            src_loc = self._resolve(job["from"])
            dst_loc = self._resolve(job["to"])
            src = Path(src_loc["path"]) / unit["relpath"]
            dst = Path(dst_loc["path"]) / unit["relpath"]

            # THE CHOKE POINT (invariant 12 corollary): the same-fs fast path
            # is an os.replace, which silently clobbers a same-named file at
            # the destination — an auto-eviction could destroy an archived
            # older version of the model with no trace. Every producer
            # (manual move, watcher, pull-through) funnels through here, so
            # one refusal covers them all.
            if dst.exists():
                raise RuntimeError(
                    f"destination already exists: {dst} — refusing to overwrite")

            def progress(done: int) -> None:
                with self._lock:
                    job["bytes_done"] = done

            def cancelled() -> bool:
                with self._lock:
                    return self._cancel_flags.get(job_id, False)

            self._mover.execute(src, dst, progress, cancelled)
            with self._lock:
                job["state"] = "verifying"      # brief post-verify bookkeeping phase
            self._catalog.record_moved(unit_id, job["to"])
            self._log("move_done", job)
            if on_success is not None:
                try:
                    on_success(dict(job))
                except Exception as exc:  # noqa: BLE001 — post-move hook, job must record it
                    with self._lock:
                        job["state"] = "failed"
                        job["error"] = f"post-move: {exc}"
                    self._log("move_failed", job, error=str(exc), phase="post-move")
                    return
            with self._lock:
                job["state"] = "done"
        except MoveCancelled:
            with self._lock:
                job["state"] = "cancelled"
            self._catalog.set_state(unit_id, "resident")
            self._log("move_cancelled", job)
        except Exception as exc:  # noqa: BLE001 — worker loop must survive any job
            with self._lock:
                job["state"] = "failed"
                job["error"] = str(exc)
            try:
                self._catalog.set_state(unit_id, "resident")
            except ValueError:
                pass                      # unit really is gone; nothing to restore
            self._log("move_failed", job, error=str(exc))
        finally:
            self._cancel_flags.pop(job_id, None)

    def _log(self, kind: str, job: dict, **extra) -> None:
        detail = {"job": job["id"], "unit": job["unit_id"],
                  "from": job["from"], "to": job["to"], **extra}
        log_event(self._events_path, kind, detail)
