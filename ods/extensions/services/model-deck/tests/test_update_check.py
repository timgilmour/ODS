"""The update-check pass: dispatch to a checker, record a verdict, log only
on transition. Nothing here actuates -- see app/update_check.py's module
docstring for why that invariant holds structurally, not by convention.
"""

import json

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
    assert "origin-moved" in kinds


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
