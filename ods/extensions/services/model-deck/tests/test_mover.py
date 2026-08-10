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
    real = mover_mod.hash_file
    monkeypatch.setattr(mover_mod, "hash_file", lambda p: "corrupted" + real(p))
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


# -- JobQueue tests (Task 7) -----------------------------------------------


from pathlib import Path

from app.catalog import Catalog
from app.events import tail_events
from app.locations import LocationStore
from app.mover import _MAX_TERMINAL_JOBS, _TERMINAL, JobQueue


def _queue_env(tmp_path):
    hot = tmp_path / "hot"; hot.mkdir()
    cold = tmp_path / "cold"; cold.mkdir()
    (hot / "a.gguf").write_bytes(b"g" * 1000)
    locs = LocationStore(tmp_path / "locations.json")
    for name, root, st in [("hot", hot, "gguf"), ("cold", cold, "gguf")]:
        locs.register({"name": name, "path": str(root), "role": "hot" if name == "hot" else "cold",
                       "store_type": st, "engine": "none", "watermark_gb": None,
                       "archive_to": None, "readonly": False})
    cat = Catalog(tmp_path / "catalog.json", locs)
    cat.scan()
    events = tmp_path / "events.jsonl"
    q = JobQueue(Mover(same_fs=lambda a, b: False), cat, locs, events)
    plan = {"unit_id": "hot:a.gguf", "src_location": "hot", "dest_location": "cold", "bytes": 1000}
    return q, cat, plan, events, hot, cold


def test_submit_marks_moving_and_process_completes(tmp_path):
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    job = q.submit(plan, label="manual move")
    assert cat.get("hot:a.gguf")["state"] == "moving"
    q._process(q._pending.pop(0))
    done = q.get(job["id"])
    assert done["state"] == "done" and done["bytes_done"] == 1000
    assert (cold / "a.gguf").exists() and not (hot / "a.gguf").exists()
    assert cat.get("cold:a.gguf")["state"] == "resident" and cat.get("hot:a.gguf") is None
    kinds = [e["kind"] for e in tail_events(events)]
    assert kinds == ["move_started", "move_done"]


def test_failed_move_restores_unit_state_and_logs(tmp_path, monkeypatch):
    q, cat, plan, events, hot, _ = _queue_env(tmp_path)
    monkeypatch.setattr(q._mover, "execute",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))
    job = q.submit(plan, label="manual move")
    q._process(q._pending.pop(0))
    assert q.get(job["id"])["state"] == "failed" and "disk gone" in q.get(job["id"])["error"]
    assert cat.get("hot:a.gguf")["state"] == "resident" and (hot / "a.gguf").exists()
    assert [e["kind"] for e in tail_events(events)] == ["move_started", "move_failed"]


def test_on_success_failure_marks_job_but_move_stays_done(tmp_path):
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    def boom(job):
        raise RuntimeError("engine notify blew up")
    job = q.submit(plan, label="pull", on_success=boom)
    q._process(q._pending.pop(0))
    got = q.get(job["id"])
    assert got["state"] == "failed" and "engine notify" in got["error"]
    assert (cold / "a.gguf").exists()                     # move itself completed
    assert cat.get("cold:a.gguf") is not None


def test_cancel_queued_job(tmp_path):
    q, cat, plan, *_ = _queue_env(tmp_path)
    job = q.submit(plan, label="manual move")
    assert q.cancel(job["id"]) is True
    assert q.get(job["id"])["state"] == "cancelled"
    assert cat.get("hot:a.gguf")["state"] == "resident"


def test_worker_thread_start_stop(tmp_path):
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    q.start()
    try:
        q.submit(plan, label="manual move")
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (cold / "a.gguf").exists():
            time.sleep(0.02)
        assert (cold / "a.gguf").exists()
    finally:
        q.stop()


# -- destination-collision guard (C1a: worker choke point) ------------------


def test_worker_refuses_when_destination_file_already_exists(tmp_path):
    """os.replace happily clobbers a same-named file at the destination, so an
    auto-eviction could silently destroy an archived older version. The worker
    refuses before the mover is ever entered: job failed, BOTH files intact,
    catalog state restored to resident."""
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    (cold / "a.gguf").write_bytes(b"older archived version")

    job = q.submit(plan, label="watermark archive")
    q._process(q._pending.pop(0))

    got = q.get(job["id"])
    assert got["state"] == "failed"
    assert "destination already exists" in got["error"]
    assert (hot / "a.gguf").read_bytes() == b"g" * 1000          # source untouched
    assert (cold / "a.gguf").read_bytes() == b"older archived version"
    assert cat.get("hot:a.gguf")["state"] == "resident"


# -- execution-start re-check carries the routes_known fail-closed rule (NEW-2) --


def test_execution_start_recheck_fails_closed_when_routes_unknown(tmp_path):
    """Same fail-closed rule as plan_move (I4): if litellm has gone
    unreachable by the time a queued gguf job reaches execution start,
    routes_known=False means 'couldn't ask', not 'no default route' — the
    job must fail rather than risk archiving the auto-reload target."""
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    q.world_fn = lambda: {"routes_known": False}

    job = q.submit(plan, label="manual move")
    q._process(q._pending.pop(0))

    got = q.get(job["id"])
    assert got["state"] == "failed"
    assert "litellm unreachable — cannot verify default route" in got["error"]
    assert (hot / "a.gguf").exists() and not (cold / "a.gguf").exists()
    assert cat.get("hot:a.gguf")["state"] == "resident"


# -- EXDEV fallback on the same-fs fast path (I1) ---------------------------


def _exdev_once(monkeypatch, errno_value):
    """Make the NEXT os.replace raise OSError(errno_value); later ones are real
    (the .part -> final rename inside the copy path must still work)."""
    import errno as errno_mod  # noqa: F401 - imported for callers' readability
    import os as os_mod

    import app.mover as mover_mod

    real = os_mod.replace
    calls = {"n": 0}

    def fake_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno_value, os_mod.strerror(errno_value))
        return real(a, b)

    monkeypatch.setattr(mover_mod.os, "replace", fake_replace)
    return calls


def test_same_fs_exdev_falls_back_to_copy_path(tmp_path, monkeypatch):
    """Bind mounts across mount points raise EXDEV from rename even when both
    sides live on one filesystem (so st_dev matched). The move must complete
    via the copy+verify path rather than failing the job."""
    import errno

    src, dst = _mk(tmp_path)
    data = src.read_bytes()
    _exdev_once(monkeypatch, errno.EXDEV)

    progress = []
    Mover(same_fs=lambda a, b: True).execute(src, dst, progress.append, lambda: False)

    assert not src.exists()
    assert dst.read_bytes() == data
    assert not dst.with_name(dst.name + ".part").exists()
    assert progress and progress[-1] == len(data)


def test_same_fs_non_exdev_oserror_reraises(tmp_path, monkeypatch):
    """Only EXDEV means "not actually the same filesystem". Any other errno is
    a real failure and must not be silently downgraded to a slow copy."""
    import errno

    src, dst = _mk(tmp_path)
    _exdev_once(monkeypatch, errno.EACCES)

    with pytest.raises(OSError) as excinfo:
        Mover(same_fs=lambda a, b: True).execute(src, dst, lambda b: None, lambda: False)
    assert excinfo.value.errno == errno.EACCES
    assert src.exists() and not dst.exists()


# -- progress keeps the heal-suppression window armed (I6) ------------------


def test_progress_callback_rearms_the_suppressor_during_a_long_copy(tmp_path,
                                                                    monkeypatch):
    """A pull of a 60 GB model can outlast the 600 s suppression window that
    was armed when it was submitted; healing would then wake up mid-pull and
    fight it. Each progress chunk re-arms the window."""
    from app.arbiter import HealSuppressor

    q, cat, plan, events, hot, cold = _queue_env(tmp_path)
    clock = {"t": 0.0}
    suppressor = HealSuppressor(600.0, clock=lambda: clock["t"])
    suppressor.note_deck_unload()

    real_execute = q._mover.execute

    def slow_execute(src, dst, progress_cb, cancel_check):
        clock["t"] = 700.0                       # the original window expires
        assert suppressor.suppressed() is False
        progress_cb(1)                           # a chunk lands -> re-arm
        assert suppressor.suppressed() is True
        return real_execute(src, dst, progress_cb, cancel_check)

    monkeypatch.setattr(q._mover, "execute", slow_execute)

    job = q.submit(plan, label="pull-through load",
                   on_progress=suppressor.note_deck_unload)
    q._process(q._pending.pop(0))

    assert q.get(job["id"])["state"] == "done"
    assert (cold / "a.gguf").exists()
    assert suppressor.suppressed() is True       # still armed at t=700


def test_hash_file_is_public_and_stable(tmp_path):
    """Promoted from _hash_file: provenance's on-demand deep check is the
    second consumer, and reaching into a private name across modules is how
    a refactor breaks a caller nobody remembers."""
    from app.mover import hash_file

    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")

    assert hash_file(p) == hash_file(p)
    assert len(hash_file(p)) == 64


def test_terminal_jobs_are_pruned_beyond_the_cap(tmp_path):
    """[max-review c14] Every terminal job was kept forever, and jobs()
    copies AND re-sorts the whole dict on each 3 s /api/storage/state poll —
    so a long-lived deck paid a growing cost to render a list nobody scrolls.
    Terminal jobs are history, not state: keep a bounded, newest-first tail.
    """
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)

    # Seed straight into the store: driving real copies would test the mover,
    # not the retention rule, and each one needs its own source file.
    with q._lock:
        for i in range(_MAX_TERMINAL_JOBS + 10):
            job_id = f"job{i:04d}"
            q._jobs[job_id] = {"id": job_id, "unit_id": "hot:a.gguf",
                               "from": "hot", "to": "cold", "label": "m",
                               "state": "done", "bytes_done": 1, "bytes_total": 1,
                               "error": None, "created_ts": float(i)}
            q._cancel_flags[job_id] = False

    q.submit(plan, label="the prune trigger")

    jobs = q.jobs()
    terminal = [j for j in jobs if j["state"] in _TERMINAL]
    assert len(terminal) == _MAX_TERMINAL_JOBS
    # The NEWEST survive: job0000..job0009 are the oldest by created_ts.
    surviving = {j["id"] for j in terminal}
    assert "job0000" not in surviving
    assert f"job{_MAX_TERMINAL_JOBS + 9:04d}" in surviving
    # Bookkeeping for a pruned job goes with it — otherwise _cancel_flags is
    # the same unbounded leak one dict over.
    assert "job0000" not in q._cancel_flags
    # ...and the live job is untouched by the prune.
    assert any(j["state"] not in _TERMINAL for j in jobs)


def test_pruning_never_drops_a_live_job(tmp_path):
    """The cap applies to TERMINAL jobs only. A deck with more than the cap's
    worth of queued/running work must keep every one of them — dropping a
    live job would strand its catalog entry in 'moving' forever."""
    q, cat, plan, events, hot, cold = _queue_env(tmp_path)

    with q._lock:
        for i in range(_MAX_TERMINAL_JOBS + 10):
            job_id = f"live{i:04d}"
            q._jobs[job_id] = {"id": job_id, "unit_id": "hot:a.gguf",
                               "from": "hot", "to": "cold", "label": "m",
                               "state": "running", "bytes_done": 0, "bytes_total": 1,
                               "error": None, "created_ts": float(i)}

    q.submit(plan, label="another live one")

    live = [j for j in q.jobs() if j["state"] not in _TERMINAL]
    assert len(live) == _MAX_TERMINAL_JOBS + 11
