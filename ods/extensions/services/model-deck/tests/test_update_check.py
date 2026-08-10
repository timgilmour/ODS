"""The update-check pass: dispatch to a checker, record a verdict, log only
on transition. Nothing here actuates -- see app/update_check.py's module
docstring for why that invariant holds structurally, not by convention.
"""

import json
from unittest import mock
import pathlib
import threading
import types

import pytest

from app import events, provenance, update_check, updates


def _store(tmp_path):
    return provenance.ProvenanceStore(tmp_path / "provenance.json",
                                      tmp_path / "history.jsonl")


@pytest.fixture
def store(tmp_path):
    # KNOWN BRIEF DEFECT (task-6-brief.md): the brief's tests assume a
    # `store` fixture from tests/conftest.py, but that fixture is defined
    # module-locally in tests/test_provenance.py:19 and was never promoted
    # to conftest.py. Defined here, following the same pattern, rather than
    # moved into conftest.py -- that would change shared state for the
    # whole suite (see conftest.py's own docstring on why fixtures there
    # are deliberately minimal and autouse-only).
    return _store(tmp_path)


def watched(store, artifact_id, source):
    store.set_watch(artifact_id, [source])


def ok_fetch(json_body, status_code=200, headers=None):
    def _fetch(url, *, headers=None, method="GET"):
        return {"status_code": status_code, "headers": headers or {},
                "json": json_body}
    return _fetch


# --- brief's Step 1 tests, verbatim except for the local `store` fixture ---

def test_pass_records_a_verdict_for_every_watched_artifact(store, tmp_path):
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    fetch = ok_fetch([{"name": "v1.0.0"}, {"name": "v2.0.0"}])

    result = update_check.run_pass(store, fetch, tmp_path / "events.jsonl", dedup={})

    assert result["checked"] == 1
    assert result["available"] == 1
    assert store.entry("oci:local:x")["update"]["status"] == updates.AVAILABLE


def test_an_artifact_with_no_watch_is_skipped_entirely(store, tmp_path):
    """watch: [] is an honest gap, not a check that always fails."""
    store.declare_origin("file:local:m.gguf", kind="file", node="local",
                         role="weights", origin=None)
    calls = []

    def _fetch(url, **kwargs):
        calls.append(url)
        raise AssertionError("must not be called")

    result = update_check.run_pass(store, _fetch, tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 0
    assert calls == []


def test_update_available_is_logged_once_not_every_pass(store, tmp_path):
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    fetch = ok_fetch([{"name": "v1.0.0"}, {"name": "v2.0.0"}])
    path = tmp_path / "events.jsonl"
    dedup = {}

    update_check.run_pass(store, fetch, path, dedup=dedup)
    update_check.run_pass(store, fetch, path, dedup=dedup)

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-available") == 1


def test_a_new_latest_logs_again(store, tmp_path):
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    path = tmp_path / "events.jsonl"
    dedup = {}
    update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]), path, dedup=dedup)
    update_check.run_pass(store, ok_fetch([{"name": "v3.0.0"}]), path, dedup=dedup)

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-available") == 2


def test_a_moved_repository_logs_origin_moved(store, tmp_path):
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_compare",
        "remote": "https://github.com/Kaden-Schutt/hipfire",
        "ref": "master", "pinned": "5d3683a7", "order": None})
    path = tmp_path / "events.jsonl"

    def _fetch(url, *, headers=None, method="GET"):
        return {"status_code": 301,
                "headers": {"Location": "https://api.github.com/repositories/1"},
                "json": None}

    update_check.run_pass(store, _fetch, path, dedup={})
    kinds = [e["kind"] for e in events.tail_events(path)]
    # Ruling 1 (post-review): a moved result logs origin-moved ONLY -- the
    # generic "no verdict" line adds nothing once the specific diagnosis
    # and remedy are already on record, and double-logging the same fact
    # reads as noise in the Events tab.
    assert kinds.count("origin-moved") == 1
    assert "update-check-failed" not in kinds


def test_one_failing_source_does_not_stop_the_pass(store, tmp_path):
    watched(store, "oci:local:a", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    watched(store, "oci:local:b", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/c/d",
        "pinned": "v1.0.0", "order": "semver"})

    def _fetch(url, *, headers=None, method="GET"):
        if "/a/b/" in url:
            raise RuntimeError("boom")
        return {"status_code": 200, "headers": {}, "json": [{"name": "v2.0.0"}]}

    result = update_check.run_pass(store, _fetch, tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 2
    assert store.entry("oci:local:a")["update"]["status"] == updates.UNAVAILABLE
    assert store.entry("oci:local:b")["update"]["status"] == updates.AVAILABLE


# --- additional coverage for the risks flagged in task-6-brief.md's ------
# --- "Specific risks to handle deliberately" section, and for a defect ---
# --- found while reading dispatch() adversarially (see task-6-report.md) --

def test_run_pass_tolerates_a_legacy_entry_with_no_watch_key_at_all(store, tmp_path):
    """Risk 4: live production data predates watch/update entirely -- an
    entry can be missing the 'watch' key outright, not merely carry an
    empty list. `_blank_entry` always writes `watch: []`, so this shape can
    only come from data written before Task 4 existed; simulate it by
    editing the file directly, the same way test_provenance.py exercises
    other on-disk shapes its own API cannot produce."""
    store.declare_origin("file:local:legacy.gguf", kind="file", node="local",
                         role="weights", origin=None)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    del data["file:local:legacy.gguf"]["watch"]
    path.write_text(json.dumps(data))

    def _fetch(url, **kwargs):
        raise AssertionError("must not be called")

    result = update_check.run_pass(store, _fetch, tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 0


def test_an_artifact_whose_every_source_fails_still_records_a_verdict(store, tmp_path):
    """Risk 3: a watch list that is present but entirely unreachable must
    still produce a recorded verdict (UNAVAILABLE), not be treated as
    unwatched and silently skipped."""
    store.set_watch("oci:local:x", [
        {"id": "s1", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1.0.0", "order": "semver"},
        {"id": "s2", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v1.0.0", "order": "semver"},
    ])

    def _always_fails(url, **kwargs):
        raise RuntimeError("boom")

    result = update_check.run_pass(store, _always_fails, tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 1
    assert result["available"] == 0
    entry = store.entry("oci:local:x")
    assert entry["update"] is not None
    assert entry["update"]["status"] == updates.UNAVAILABLE
    assert len(entry["update"]["sources"]) == 2


def test_a_transition_through_undetermined_clears_dedup_for_a_later_recurrence(
        store, tmp_path):
    """Risk 2: the brief's `else` branch that clears the AVAILABLE dedup key
    on 'back to healthy' must fire for UNDETERMINED too, not only CURRENT --
    otherwise passing through 'reached it, cannot rank it' on the way back
    to the SAME latest value would permanently suppress the second
    announcement. Forces UNDETERMINED via `order: none` (unrankable but
    something is present), which is the only way check_tags produces it."""
    path = tmp_path / "events.jsonl"
    dedup = {}
    source = {"id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
              "pinned": "v1.0.0", "order": "semver"}

    watched(store, "oci:local:x", source)
    update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]), path, dedup=dedup)

    watched(store, "oci:local:x", {**source, "order": "none"})
    update_check.run_pass(store, ok_fetch([{"name": "banana"}]), path, dedup=dedup)

    watched(store, "oci:local:x", source)
    update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]), path, dedup=dedup)

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-available") == 2
    # UNDETERMINED is a successful read, not a failure -- it must not have
    # produced a spurious update-check-failed line along the way.
    assert "update-check-failed" not in kinds


def test_a_recovered_failure_that_recurs_with_an_identical_note_logs_again(
        store, tmp_path):
    """Ruling 2 (post-review), a latent bug not a defensible choice: as
    originally implemented, the `#failed` dedup key never cleared on
    recovery, so fail -> recover -> fail again with the SAME note was
    silently suppressed forever -- exactly the recurring/flapping failure
    an operator most needs to see, made invisible after its first
    occurrence. `#failed` must clear whenever a source reports anything
    other than UNAVAILABLE, symmetric with how the AVAILABLE key already
    clears on any non-available status."""
    source = {"id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
              "pinned": "v1.0.0", "order": "semver"}
    watched(store, "oci:local:x", source)
    path = tmp_path / "events.jsonl"
    dedup = {}

    def _fails(url, **kwargs):
        raise RuntimeError("boom")

    update_check.run_pass(store, _fails, path, dedup=dedup)                       # fail #1
    update_check.run_pass(store, ok_fetch([{"name": "v1.0.0"}]), path, dedup=dedup)  # recovers to CURRENT
    update_check.run_pass(store, _fails, path, dedup=dedup)                       # fail #2, identical note

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-check-failed") == 2


def test_dispatch_survives_a_source_missing_an_id():
    """DEFECT FOUND (not in the brief's tests): every checker builds its
    result via `source["id"]` (app/updates/git.py's and oci.py's `_result`
    helpers), so a source missing "id" makes the checker itself raise
    KeyError -- which is exactly what dispatch's `except Exception` exists
    to catch. But the brief's except-handler ALSO does `source["id"]` to
    build its own fallback result, so it raises the identical KeyError and
    the exception escapes dispatch (and would take down the whole pass, not
    just this source). app.provenance.record_update already defends against
    a stored source missing "id" (see its docstring) -- dispatch must too."""
    source = {"check": "git_tags", "remote": "https://github.com/a/b",
              "pinned": "v1.0.0", "order": "semver"}  # no "id"
    result = update_check.dispatch(source, ok_fetch([{"name": "v1.0.0"}]))
    assert result["status"] == updates.UNAVAILABLE


def test_run_pass_survives_a_watch_source_missing_an_id(store, tmp_path):
    """Same defect as above, exercised through run_pass end-to-end with an
    on-disk shape set_watch's own validation would never write (validate_watch
    requires "id"), simulating a hand-edited provenance.json."""
    store.set_watch("oci:local:x", [
        {"id": "placeholder", "check": "git_tags",
         "remote": "https://github.com/a/b", "pinned": "v1.0.0", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    del data["oci:local:x"]["watch"][0]["id"]
    path.write_text(json.dumps(data))

    result = update_check.run_pass(store, ok_fetch([{"name": "v1.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 1


def test_update_check_failed_is_logged_once_not_every_pass(store, tmp_path):
    """Baseline for the failed-event dedup path -- none of the brief's 6
    tests exercise it at all (only the AVAILABLE dedup is tested)."""
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    path = tmp_path / "events.jsonl"
    dedup = {}

    def _always_fails(url, **kwargs):
        raise RuntimeError("boom")

    update_check.run_pass(store, _always_fails, path, dedup=dedup)
    update_check.run_pass(store, _always_fails, path, dedup=dedup)

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-check-failed") == 1


def test_a_flapping_exception_message_does_not_defeat_the_failed_dedup(store, tmp_path):
    """DEFECT FOUND (not in the brief's tests): dispatch's except-handler
    built its note as `f"{type(exc).__name__}: {exc}"`. A raising fetch's
    exception text can be non-deterministic across calls even for the
    identical underlying failure -- the standard example is urllib3/requests
    wrapping a socket failure in a connection-object repr that embeds a
    memory address, which differs every call. Keying the failed-event dedup
    on that text defeats the dedup entirely: a permanently-broken remote
    would log a new "update-check-failed" event every single pass. This is
    precisely the spam app/arbiter.py:438 was fixed to avoid by logging the
    exception's CLASS only, never its message -- a precedent this module's
    own docstring cites by line number, which makes the brief's version of
    dispatch() inconsistent with the very file it invokes."""
    watched(store, "oci:local:x", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    path = tmp_path / "events.jsonl"
    dedup = {}
    calls = {"n": 0}

    def _flaky_fetch(url, **kwargs):
        calls["n"] += 1
        raise RuntimeError(f"connection object at 0x{id(calls) + calls['n']:x}")

    update_check.run_pass(store, _flaky_fetch, path, dedup=dedup)
    update_check.run_pass(store, _flaky_fetch, path, dedup=dedup)

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("update-check-failed") == 1


# --- Fix round 2: an exception could still escape run_pass -- both paths --
# --- flagged Important by the coordinator, both confirmed reachable ------
# --- before the fix, plus the bundled #moved recovery Minor. -------------

def test_dispatch_survives_a_non_dict_source():
    """`dispatch(None, fetch)` raised AttributeError before this guard --
    `source.get("check")` on a non-dict executes before `dispatch`'s own
    try/except even starts, so the exception escaped dispatch entirely."""
    assert update_check.dispatch(None, ok_fetch([]))["status"] == updates.UNAVAILABLE
    assert update_check.dispatch("oops", ok_fetch([]))["status"] == updates.UNAVAILABLE
    assert update_check.dispatch(42, ok_fetch([]))["status"] == updates.UNAVAILABLE


def test_run_pass_tolerates_non_dict_elements_in_a_stored_watch_list(store, tmp_path):
    """Required coverage 1: a stored `watch` list containing `None` and a
    bare string (the same hand-edited-provenance.json threat model as the
    missing-"id" tests) must not crash the pass -- it should still record
    a verdict for the artifact's well-formed sibling source."""
    store.set_watch("oci:local:x", [
        {"id": "s1", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1.0.0", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"].append(None)
    data["oci:local:x"]["watch"].append("not-a-source")
    path.write_text(json.dumps(data))

    result = update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 1
    entry = store.entry("oci:local:x")
    assert entry["update"] is not None
    assert entry["update"]["status"] == updates.AVAILABLE


def test_run_pass_tolerates_a_per_artifact_entry_that_is_not_a_dict(store, tmp_path):
    """Required coverage 2, END TO END. A hand-edited file can hold a bare
    string where an entry belongs; the pass must skip it and check nothing.

    STALE DOCSTRING CORRECTED (this fix wave). This used to claim
    `provenance._load()` guards only the top-level document shape. That
    stopped being true when Task 5 round 4 (5e22d9eb) moved shape
    normalisation into `_load()`: the bare string is now rebuilt into a blank
    entry with `watch: []` before run_pass ever sees it, so what this pins is
    the OUTCOME through a real store, not run_pass's own isinstance guard.
    That guard is exercised directly by
    test_run_pass_tolerates_a_non_dict_entry_from_an_ungated_store below --
    mutating it away left this test green."""
    store.declare_origin("file:local:legacy.gguf", kind="file", node="local",
                         role="weights", origin=None)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:corrupt"] = "not a dict at all"
    path.write_text(json.dumps(data))

    def _fetch(url, **kwargs):
        raise AssertionError("must not be called")

    result = update_check.run_pass(store, _fetch, tmp_path / "e.jsonl", dedup={})
    assert result["checked"] == 0


def test_a_malformed_sibling_artifact_does_not_block_a_healthy_one(store, tmp_path):
    """Required coverage 3: a corrupt per-artifact entry AND a corrupt watch
    list, sitting alongside a healthy watched artifact in the same store,
    must not prevent the healthy artifact from being checked and its verdict
    recorded in the same pass.

    STALE DOCSTRING CORRECTED (this fix wave): since Task 5 round 4
    (5e22d9eb) the gate normalises both corruptions in `_load()`, so this is
    an end-to-end outcome test, not a test of run_pass's own tolerance. The
    per-artifact try/except it was believed to cover is exercised by
    test_a_record_update_failure_on_one_artifact_does_not_stop_the_pass
    below -- mutating that except away left this test green."""
    watched(store, "oci:local:good", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:bad-entry"] = "not a dict"
    data["oci:local:bad-watch"] = {"watch": [None, 42, "oops"]}
    path.write_text(json.dumps(data))

    result = update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})
    assert store.entry("oci:local:good")["update"]["status"] == updates.AVAILABLE
    assert result["checked"] >= 1


_TAGS_SOURCE = {"id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
                "pinned": "v1.0.0", "order": "semver"}


def test_a_record_update_failure_on_one_artifact_does_not_stop_the_pass(
        store, tmp_path):
    """THE PER-ARTIFACT try/except, exercised at last.

    Mutating `except Exception` to `except ZeroDivisionError` left the whole
    suite green before this test existed: every other "malformed sibling"
    fixture is now absorbed by `provenance._load()`'s gate (Task 5 round 4,
    5e22d9eb), so nothing reached the body of the try at all. The one thing
    that genuinely still can raise inside it is `store.record_update`, which
    re-reads the file under its own lock -- a value that changes shape
    between this loop's snapshot and that fresh read.

    The corrupt artifact sorts FIRST, so a healthy sibling being checked
    afterwards is the proof the pass continued rather than unwound."""
    watched(store, "oci:local:bad", _TAGS_SOURCE)
    watched(store, "oci:local:good", _TAGS_SOURCE)
    real_record = store.record_update

    def _record(artifact_id, results, now=None):
        if artifact_id == "oci:local:bad":
            raise RuntimeError("the file changed shape under us")
        return real_record(artifact_id, results, now)

    store.record_update = _record
    result = update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})

    assert store.entry("oci:local:good")["update"]["status"] == updates.AVAILABLE
    assert store.entry("oci:local:bad")["update"] is None
    assert result["available"] == 1


def test_an_artifact_whose_verdict_could_not_be_recorded_is_not_counted(
        store, tmp_path):
    """`checked` used to be incremented BEFORE the try, so a raising
    `record_update` counted an artifact that got no verdict and no event --
    the count said "I checked this" about the one artifact nobody can see a
    result for. It now counts only what was actually recorded."""
    watched(store, "oci:local:x", _TAGS_SOURCE)

    def _record(artifact_id, results, now=None):
        raise RuntimeError("boom")

    store.record_update = _record
    result = update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})
    assert result == {"checked": 0, "available": 0}


def test_run_pass_tolerates_a_non_dict_entry_from_an_ungated_store(tmp_path):
    """`run_pass` takes anything with `.get()`/`.record_update()`, not only a
    gated ProvenanceStore -- the isinstance guard is what makes that contract
    safe, and `entry.get("watch")` runs OUTSIDE the per-artifact try, so an
    ungated non-dict entry escapes the whole pass rather than one artifact.

    This is the direct test the end-to-end fixtures above stopped being once
    `_load()` started normalising entries."""
    recorded = []

    class _UngatedStore:
        def get(self):
            return {"oci:local:corrupt": "not a dict at all",
                    "oci:local:x": {"watch": [_TAGS_SOURCE]}}

        def record_update(self, artifact_id, results, now=None):
            recorded.append(artifact_id)
            return updates.AVAILABLE

    result = update_check.run_pass(_UngatedStore(), ok_fetch([{"name": "v2.0.0"}]),
                                   tmp_path / "e.jsonl", dedup={})
    assert recorded == ["oci:local:x"]
    assert result == {"checked": 1, "available": 1}


def test_the_dedup_key_cannot_collide_across_an_artifact_and_a_source_id(
        store, tmp_path):
    """The dedup key was `f"{artifact_id}/{result['id']}"`. Weights artifact
    ids legitimately contain "/" (a relpath -- app/provenance_collect.py:110)
    and source ids are free-form operator strings, so
    ("oci:local:a/b", "c") and ("oci:local:a", "b/c") produced the SAME key:
    whichever ran first suppressed the other's update-available event. A
    tuple key removes the class rather than escaping the separator."""
    watched(store, "oci:local:a/b", {**_TAGS_SOURCE, "id": "c"})
    watched(store, "oci:local:a", {**_TAGS_SOURCE, "id": "b/c"})
    path = tmp_path / "events.jsonl"

    update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]), path, dedup={})

    logged = [e for e in events.tail_events(path) if e["kind"] == "update-available"]
    assert sorted(e["detail"]["artifact_id"] for e in logged) == \
        ["oci:local:a", "oci:local:a/b"]


def test_moved_dedup_clears_on_recovery_so_a_later_move_to_the_same_place_relogs(
        store, tmp_path):
    """Bundled Minor: `#moved` must clear whenever a result stops reporting
    `detail.moved`, the same asymmetry fix already applied to `#failed` --
    otherwise move to X, get re-declared/fixed, then move to X again would
    not re-log `origin-moved`."""
    source = {"id": "t", "check": "git_compare",
              "remote": "https://github.com/Kaden-Schutt/hipfire",
              "ref": "master", "pinned": "5d3683a7", "order": None}
    watched(store, "oci:local:x", source)
    path = tmp_path / "events.jsonl"
    dedup = {}

    def _moved(url, *, headers=None, method="GET"):
        return {"status_code": 301,
                "headers": {"Location": "https://api.github.com/repositories/1"},
                "json": None}

    def _fixed(url, *, headers=None, method="GET"):
        return {"status_code": 200, "headers": {},
                "json": {"ahead_by": 0, "behind_by": 0, "status": "identical"}}

    update_check.run_pass(store, _moved, path, dedup=dedup)   # moved -> X
    update_check.run_pass(store, _fixed, path, dedup=dedup)   # re-declared/fixed
    update_check.run_pass(store, _moved, path, dedup=dedup)   # moved again -> SAME X

    kinds = [e["kind"] for e in events.tail_events(path)]
    assert kinds.count("origin-moved") == 2


# --- Task 8: UpdateChecker, its own thread, wired outside arbiter.tick() ---


def _settings(**kwargs):
    base = {"update_interval_s": 3600.0, "update_check_enabled": True}
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_disabled_checker_never_runs_a_pass(store, tmp_path):
    checker = update_check.UpdateChecker(
        settings=_settings(update_check_enabled=False),
        provenance_store=store, events_path=tmp_path / "e.jsonl")
    checker._fetch_factory = lambda: (_ for _ in ()).throw(
        AssertionError("must not build a fetcher"))

    # The raising _fetch_factory alone does NOT bind: tick()'s supervisor
    # `except Exception` catches AssertionError too, so without the
    # `not self._enabled` guard this test still passed while logging
    # `update-check-error`. The discriminator is the EVENT LOG — a disabled
    # checker is a silent no-op, a crashed one is not.
    assert checker.tick() is None
    assert events.tail_events(tmp_path / "e.jsonl") == []


def test_tick_survives_a_pass_that_raises(store, tmp_path):
    checker = update_check.UpdateChecker(
        settings=_settings(), provenance_store=store,
        events_path=tmp_path / "e.jsonl")

    def boom():
        raise RuntimeError("kaboom")

    checker._fetch_factory = boom
    checker.tick()          # supervisor catch: must not propagate

    kinds = [e["kind"] for e in events.tail_events(tmp_path / "e.jsonl")]
    # `update-check-error`, NOT `update-check-failed`: see
    # test_one_event_kind_never_carries_two_detail_shapes below.
    assert "update-check-error" in kinds


def test_one_event_kind_never_carries_two_detail_shapes(store, tmp_path):
    """The defect class this codebase has shipped four times, caught before
    it shipped a fifth: the tick-level supervisor catch emitted `{"error"}`
    under the SAME kind the per-source path emits `{"artifact_id","source",
    "note"}` under (app/update_check.py:245-248). A consumer that keys off
    the kind then has to guess which shape it got. Two facts, two kinds.

    Both still classify as failures in the Events tab -- `-error` and
    `-failed` are both in ui/src/model/eventSeverity.ts's FAILURE_SUFFIXES."""
    path = tmp_path / "e.jsonl"
    watched(store, "oci:local:x", _TAGS_SOURCE)

    def _always_fails(url, **kwargs):
        raise RuntimeError("boom")

    checker = update_check.UpdateChecker(settings=_settings(),
                                         provenance_store=store,
                                         events_path=path)
    checker._fetch_factory = lambda: _always_fails
    checker.tick()                      # per-source failure, pass succeeds

    checker._fetch_factory = lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))
    checker.tick()                      # the pass itself could not run

    by_kind = {}
    for event in events.tail_events(path):
        by_kind.setdefault(event["kind"], []).append(event["detail"])
    assert set(by_kind["update-check-failed"][0]) == {"artifact_id", "source", "note"}
    assert set(by_kind["update-check-error"][0]) == {"error"}


def test_two_ticks_never_run_a_pass_at_the_same_time(store, tmp_path):
    """POST /api/provenance/check runs `tick()` on the request thread
    (app/routers/provenance.py:210) while the checker's own thread may be
    mid-pass. They share `self._dedup` and `self._etags` unsynchronised (so
    an event can double-log) and each builds a FRESH fetch closure, making
    the 40-request budget per-PASS rather than per-hour -- which is what
    app/updates/fetch.py:8-14 says the budget exists to protect against on
    GitHub's 60/hour anonymous per-IP ceiling.

    Deterministic, not timing-sampled: if two passes can be inside the body
    at once the barrier releases; if they are serialised it can only ever
    break."""
    checker = update_check.UpdateChecker(settings=_settings(),
                                         provenance_store=store,
                                         events_path=tmp_path / "e.jsonl")
    barrier = threading.Barrier(2, timeout=0.5)
    overlapped = []

    def _factory():
        try:
            barrier.wait()
            overlapped.append(True)
        except threading.BrokenBarrierError:
            pass
        return ok_fetch([])

    checker._fetch_factory = _factory
    threads = [threading.Thread(target=checker.tick) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(t.is_alive() for t in threads)
    assert overlapped == []


def test_start_and_stop_are_clean(store, tmp_path):
    checker = update_check.UpdateChecker(
        settings=_settings(update_interval_s=0.05), provenance_store=store,
        events_path=tmp_path / "e.jsonl")
    checker._fetch_factory = lambda: (lambda url, **kw: {
        "status_code": 200, "headers": {}, "json": []})
    checker.start()
    checker.stop()
    assert checker._thread is not None
    assert isinstance(checker._thread, threading.Thread)
    assert not checker._thread.is_alive()


def test_the_checker_is_not_reachable_from_the_watcher_tick():
    """U8, asserted structurally: arbiter must not import or call this.

    FIXED FROM THE BRIEF: the brief's version read `pathlib.Path
    ("app/arbiter.py")`, a path relative to the process's CWD -- it only
    resolves correctly when pytest happens to be invoked from the service
    root, and silently passes-for-the-wrong-reason (or raises FileNotFound,
    which is at least loud) otherwise. Anchored on this test file's own
    location instead, so the assertion is correct regardless of the
    invocation directory."""
    service_root = pathlib.Path(__file__).resolve().parents[1]
    source = (service_root / "app" / "arbiter.py").read_text()
    assert "update_check" not in source
    assert "UpdateChecker" not in source


# --- additional coverage for the "Specific risks to handle deliberately" ---
# --- section of task-8-brief.md, none of which the brief's own 4 tests ----
# --- above actually exercise. -----------------------------------------------


def test_stop_without_start_is_safe_and_does_not_hang(store, tmp_path):
    """stop() must tolerate never having had start() called at all -- not
    just the disabled-settings case, the general one (e.g. a caller that
    constructs a checker and tears it down before ever starting it)."""
    checker = update_check.UpdateChecker(
        settings=_settings(), provenance_store=store,
        events_path=tmp_path / "e.jsonl")
    checker.stop()          # must not raise, hang, or require start() first
    assert checker._thread is None


def test_disabled_checker_start_spawns_no_thread(store, tmp_path):
    """With update_check_enabled=False, start() itself must be a no-op --
    not merely tick(), which the brief's own test already covers."""
    checker = update_check.UpdateChecker(
        settings=_settings(update_check_enabled=False), provenance_store=store,
        events_path=tmp_path / "e.jsonl")
    checker.start()
    assert checker._thread is None
    checker.stop()          # still must be safe/clean afterwards


def test_tick_with_no_provenance_store_is_a_clean_noop(tmp_path):
    """Confirms the checker behaves sanely when constructed without a
    provenance store (app.main's `_build_update_checker` uses
    `deck.get("provenance_store")`, which is None if that key is ever
    absent) -- tick() must no-op cleanly, the same as the disabled case,
    and must never attempt to build a fetcher."""
    checker = update_check.UpdateChecker(
        settings=_settings(), provenance_store=None,
        events_path=tmp_path / "e.jsonl")
    checker._fetch_factory = lambda: (_ for _ in ()).throw(
        AssertionError("must not build a fetcher without a store"))
    assert checker.tick() is None


def test_tick_survives_run_pass_itself_raising(store, tmp_path, monkeypatch):
    """The brief's own test only proves a raising `_fetch_factory` is
    caught. `run_pass` itself can also raise outside its own per-artifact
    try/except -- `store.get()` is called before that loop's guard, e.g. on
    a corrupt on-disk shape -- and tick() is the supervisor's last line of
    defense against that too."""
    checker = update_check.UpdateChecker(
        settings=_settings(), provenance_store=store,
        events_path=tmp_path / "e.jsonl")
    checker._fetch_factory = lambda: (lambda url, **kw: {
        "status_code": 200, "headers": {}, "json": []})

    def _boom():
        raise RuntimeError("store broke")

    monkeypatch.setattr(store, "get", _boom)

    assert checker.tick() is None
    kinds = [e["kind"] for e in events.tail_events(tmp_path / "e.jsonl")]
    assert "update-check-error" in kinds


def test_fetch_factory_reuses_one_http_client_across_ticks(store, tmp_path, monkeypatch):
    """app/updates/fetch.py's own module docstring explicitly directs Task 8
    to build ONE client and reuse it across passes on the six-hour cadence,
    rather than have `make_fetch` open a fresh one every tick. Verified by
    spying on `update_check.make_fetch` rather than opening real sockets."""
    calls = []

    def _fake_make_fetch(*, timeout_s=8.0, budget=40, etags=None, client=None):
        calls.append(client)
        return lambda url, **kw: {"status_code": 200, "headers": {}, "json": []}

    monkeypatch.setattr(update_check, "make_fetch", _fake_make_fetch)

    checker = update_check.UpdateChecker(
        settings=_settings(), provenance_store=store,
        events_path=tmp_path / "e.jsonl")
    checker.tick()
    checker.tick()

    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[0] is calls[1]


def test_a_failing_artifact_logs_an_event_and_the_pass_continues(store, tmp_path):
    """[max-review c6] The per-artifact catch was fully silent, so an
    artifact could fail every pass forever with nothing to show for it. It
    must stay ISOLATED (the other artifacts still get verdicts) but become
    loud, naming which artifact failed.

    The kind is deliberately NOT `update-check-error`: that one is the
    whole-pass supervisor's and carries {"error"} alone. Two detail shapes
    under one kind is the vocabulary defect this module documents at that
    catch, so this asserts the distinct kind explicitly.
    """
    watched(store, "oci:local:broken", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/a/b",
        "pinned": "v1.0.0", "order": "semver"})
    watched(store, "oci:local:fine", {
        "id": "t", "check": "git_tags", "remote": "https://github.com/c/d",
        "pinned": "v1.0.0", "order": "semver"})

    # Raised at the DISPATCH layer, not from `fetch`: a fetch that raises is
    # caught inside dispatch and returned as a verdict, so it never reaches
    # the per-artifact catch this test is about.
    real_dispatch = update_check.dispatch

    def exploding_dispatch(source, fetch):
        if "/a/b" in source.get("remote", ""):
            raise RuntimeError("upstream exploded")
        return real_dispatch(source, fetch)

    events_path = tmp_path / "events.jsonl"
    with mock.patch.object(update_check, "dispatch", exploding_dispatch):
        result = update_check.run_pass(store, ok_fetch([{"name": "v2.0.0"}]),
                                       events_path, dedup={})

    # Isolation held: the healthy artifact still got its verdict.
    assert result["checked"] == 1
    assert store.entry("oci:local:fine")["update"]["status"] == updates.AVAILABLE

    entries = [json.loads(line) for line in
               events_path.read_text().splitlines() if line.strip()]
    errors = [e for e in entries if e["kind"] == "update-check-artifact-error"]
    assert len(errors) == 1
    assert errors[0]["detail"]["artifact_id"] == "oci:local:broken"
    assert "upstream exploded" in errors[0]["detail"]["error"]
