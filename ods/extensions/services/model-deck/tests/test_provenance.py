"""The provenance store: atomic current+desired state, drift, gaps."""

import copy
import inspect
import json

import pytest

from app import origins, provenance, provenance_history, updates
from app.provenance import ProvenanceStore

T0 = "2026-08-01T00:00:00+00:00"
T1 = "2026-08-02T00:00:00+00:00"


def _store(tmp_path):
    return ProvenanceStore(tmp_path / "provenance.json", tmp_path / "history.jsonl")


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path)


def _identity(version="sha256:a", label="ds4-spark:v0.5.6"):
    return {"version": version, "label": label, "detail": {},
            "verification": origins.EXACT}


# --- persistence ----------------------------------------------------------

def test_missing_file_reads_as_empty(tmp_path):
    assert _store(tmp_path).get() == {}


def test_observe_persists_and_reads_back(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:ods-hipfire", kind="oci", node="local",
                  role="engine", current=_identity(), now=T0)
    entry = _store(tmp_path).entry("oci:local:ods-hipfire")
    assert entry["current"]["version"] == "sha256:a"
    assert entry["current"]["source"] == "derived"
    assert entry["current"]["verification"] == origins.EXACT
    assert entry["origin"] is None       # never inferred (D8)
    assert entry["desired"] is None


def test_a_corrupt_file_is_renamed_aside_not_silently_emptied(tmp_path):
    # D13: unlike policy/registry/catalog, this file holds operator-declared
    # origins that exist nowhere else. Self-healing to empty would destroy them.
    path = tmp_path / "provenance.json"
    path.write_text("{ this is not json")
    store = ProvenanceStore(path, tmp_path / "history.jsonl")
    assert store.get() == {}
    corpses = list(tmp_path.glob("provenance.json.corrupt-*"))
    assert len(corpses) == 1
    assert corpses[0].read_text() == "{ this is not json"


def test_a_non_object_document_is_also_quarantined(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(["not", "a", "mapping"]))
    store = ProvenanceStore(path, tmp_path / "history.jsonl")
    assert store.get() == {}
    assert list(tmp_path.glob("provenance.json.corrupt-*"))


def test_bad_artifact_ids_are_refused(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(origins.BadArtifactId):
        store.observe("nonsense", kind="oci", node="local", role="engine",
                      current=_identity(), now=T0)


def test_an_id_that_disagrees_with_its_kind_or_node_is_refused(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(origins.BadArtifactId):
        store.observe("oci:local:x", kind="file", node="local", role="weights",
                      current=_identity(), now=T0)
    with pytest.raises(origins.BadArtifactId):
        store.observe("oci:local:x", kind="oci", node="sparky", role="engine",
                      current=_identity(), now=T0)


def test_an_unknown_role_is_refused(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.observe("oci:local:x", kind="oci", node="local", role="widget",
                      current=_identity(), now=T0)


# --- history integration --------------------------------------------------

def test_a_version_change_writes_one_history_record(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity("sha256:a", "v0.5.3"), now=T0)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity("sha256:b", "v0.5.6"), now=T1)
    records = provenance_history.history_for(tmp_path / "history.jsonl", "oci:local:x")
    assert [r["ts"] for r in records] == [T0, T1]
    assert records[1]["from"]["version"] == "sha256:a"
    assert records[1]["to"]["version"] == "sha256:b"


def test_the_first_observation_records_a_null_from(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    records = provenance_history.history_for(tmp_path / "history.jsonl", "oci:local:x")
    assert records[0]["from"] is None


def test_an_unchanged_observation_writes_no_history(tmp_path):
    store = _store(tmp_path)
    for now in (T0, T1):
        store.observe("oci:local:x", kind="oci", node="local", role="engine",
                      current=_identity(), now=now)
    assert len(provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x")) == 1


def test_an_unchanged_observation_still_refreshes_verified_at(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    entry = store.observe("oci:local:x", kind="oci", node="local", role="engine",
                          current=_identity(), now=T1)
    assert entry["current"]["observed_at"] == T0     # the version is that old
    assert entry["current"]["verified_at"] == T1     # but we just checked it


# --- retention ------------------------------------------------------------

def test_mark_unavailable_keeps_the_last_known_current(tmp_path):
    # app.catalog's retention rule: an unplugged drive must not erase history.
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.mark_unavailable("oci:local:x", now=T1)
    entry = store.entry("oci:local:x")
    assert entry["current"]["version"] == "sha256:a"
    assert entry["current"]["verification"] == origins.UNAVAILABLE


def test_mark_unavailable_on_an_unknown_artifact_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    store.mark_unavailable("oci:local:ghost", now=T0)
    assert store.get() == {}


# --- declared origin ------------------------------------------------------

def test_declare_origin_preserves_a_derived_current(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    entry = store.declare_origin(
        "oci:local:x", kind="oci", node="local", role="engine",
        origin={"registry": None, "repository": "x", "reference": "x:1",
                "build": None, "archive": None},
        update_path="watch releases", notes="no source repo", now=T1)
    assert entry["origin"]["repository"] == "x"
    assert entry["current"]["version"] == "sha256:a"
    assert entry["update_path"] == "watch releases"
    assert entry["notes"] == "no source repo"


def test_declare_origin_works_before_anything_is_observed(tmp_path):
    store = _store(tmp_path)
    entry = store.declare_origin(
        "git:sparky:~/ds4", kind="git", node="sparky", role="source",
        origin={"remote": "r", "ref": "v0.5.6", "ref_kind": "tag",
                "default_branch": "main", "diverges_from_default": True}, now=T0)
    assert entry["current"]["verification"] == origins.UNKNOWN
    assert entry["current"]["version"] is None


def test_declare_origin_writes_a_history_record(tmp_path):
    store = _store(tmp_path)
    store.declare_origin("git:sparky:~/ds4", kind="git", node="sparky",
                         role="source", origin={"remote": "r"}, now=T0)
    records = provenance_history.history_for(tmp_path / "history.jsonl",
                                             "git:sparky:~/ds4")
    assert records[0]["field"] == "origin"
    assert records[0]["to"] == {"remote": "r"}


# --- desired --------------------------------------------------------------

def test_set_and_clear_desired(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.set_desired("oci:local:x", version="sha256:b", label="v0.5.6", now=T1)
    assert store.entry("oci:local:x")["desired"]["version"] == "sha256:b"
    store.clear_desired("oci:local:x", now=T1)
    assert store.entry("oci:local:x")["desired"] is None


def test_set_desired_on_an_unknown_artifact_is_refused(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.set_desired("oci:local:ghost", version="v", label=None, now=T0)


def test_clearing_an_already_absent_desired_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.clear_desired("oci:local:x", now=T1)
    assert not [r for r in provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x") if r["field"] == "desired"]


# --- deep verify ----------------------------------------------------------

def test_record_deep_verify_promotes_a_weights_entry_to_exact(tmp_path):
    store = _store(tmp_path)
    store.observe("file:local:m.gguf", kind="file", node="local", role="weights",
                  current={"version": None, "label": "m.gguf",
                           "detail": {"size_bytes": 1, "mtime": 2.0},
                           "verification": origins.CONSISTENT}, now=T0)
    entry = store.record_deep_verify("file:local:m.gguf", "abc123", now=T1)
    assert entry["current"]["version"] == "abc123"
    assert entry["current"]["verification"] == origins.EXACT


def test_record_deep_verify_on_an_unknown_artifact_is_refused(tmp_path):
    with pytest.raises(KeyError):
        _store(tmp_path).record_deep_verify("file:local:ghost", "abc", now=T0)


# --- delete ---------------------------------------------------------------

def test_delete_removes_the_entry_but_not_its_history(tmp_path):
    store = _store(tmp_path)
    store.observe("oci:local:x", kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.delete("oci:local:x")
    assert store.entry("oci:local:x") is None
    assert provenance_history.history_for(tmp_path / "history.jsonl", "oci:local:x")


# --- pure describe / gaps -------------------------------------------------

def _entry(current_version="sha256:a", desired=None, verified_at=T0):
    return {"artifact_id": "oci:local:x", "kind": "oci", "node": "local",
            "role": "engine", "origin": None,
            "current": {"version": current_version, "label": "v0.5.3", "detail": {},
                        "source": "derived", "observed_at": T0,
                        "verification": origins.EXACT, "verified_at": verified_at},
            "desired": desired, "update_path": None, "notes": None}


def test_describe_reports_version_drift_against_desired():
    data = {"oci:local:x": _entry(desired={
        "version": "sha256:b", "label": "v0.5.6", "set_by": "operator", "set_at": T0})}
    assert provenance.describe(data, now=T0, stale_s=3600)[0]["version_drift"] is True


def test_matching_versions_are_not_drift():
    data = {"oci:local:x": _entry(desired={
        "version": "sha256:a", "label": "v0.5.3", "set_by": "operator", "set_at": T0})}
    assert provenance.describe(data, now=T0, stale_s=3600)[0]["version_drift"] is False


@pytest.mark.parametrize("current_version,desired", [
    ("sha256:a", None),        # no opinion -> not drift
    (None, {"version": "sha256:b", "label": None, "set_by": "operator", "set_at": T0}),
])
def test_absence_is_never_drift(current_version, desired):
    # app.facts' rule: "cannot check" must never render as "mismatch".
    data = {"oci:local:x": _entry(current_version=current_version, desired=desired)}
    assert provenance.describe(data, now=T0, stale_s=3600)[0]["version_drift"] is False


def test_describe_computes_stale_at_read_time_without_storing_it():
    data = {"oci:local:x": _entry()}
    fresh = provenance.describe(data, now=T0, stale_s=3600)[0]
    assert fresh["verification"] == origins.EXACT
    later = provenance.describe(data, now="2026-08-09T00:00:00+00:00", stale_s=3600)[0]
    assert later["verification"] == origins.STALE
    assert data["oci:local:x"]["current"]["verification"] == origins.EXACT  # not mutated


def test_unavailable_is_never_relabelled_stale():
    # "the node is down" outranks "we have not looked lately"; collapsing them
    # would hide the actionable one.
    data = {"oci:local:x": _entry()}
    data["oci:local:x"]["current"]["verification"] = origins.UNAVAILABLE
    described = provenance.describe(data, now="2026-09-01T00:00:00+00:00", stale_s=1)
    assert described[0]["verification"] == origins.UNAVAILABLE


def test_describe_is_in_key_order():
    data = {"oci:local:b": {**_entry(), "artifact_id": "oci:local:b"},
            "oci:local:a": {**_entry(), "artifact_id": "oci:local:a"}}
    assert [e["artifact_id"] for e in provenance.describe(
        data, now=T0, stale_s=3600)] == ["oci:local:a", "oci:local:b"]


def test_gaps_lists_artifacts_with_no_recorded_origin():
    data = {
        "oci:local:a": {"origin": None},
        "oci:local:b": {"origin": {"registry": None, "repository": "b"}},
    }
    assert provenance.gaps(data) == ["oci:local:a"]


def test_describe_leaves_exactly_one_place_to_read_the_verification():
    """Two fields with the same name and different answers is how a UI ends
    up rendering `exact` for something that has gone stale."""
    data = {"oci:local:x": _entry()}
    described = provenance.describe(data, now="2026-08-09T00:00:00+00:00",
                                    stale_s=3600)[0]
    assert described["verification"] == origins.STALE
    assert "verification" not in described["current"]
    # ...and the stored document is untouched.
    assert data["oci:local:x"]["current"]["verification"] == origins.EXACT


def test_describe_defaults_watch_and_update_for_pre_migration_entries():
    # _entry() predates this task and carries no "watch"/"update" keys at
    # all -- exactly the shape of the live provenance.json on two machines.
    data = {"oci:local:x": _entry()}
    described = provenance.describe(data, now=T0, stale_s=3600)[0]
    assert described["watch"] == []
    assert described["update"] is None


def test_describe_passes_watch_and_update_through_unmodified():
    """No transformation of its own: whatever `record_update` wrote is what
    a consumer reads. (Round 4 note: the stored `update` here is the shape
    `record_update` actually writes -- status/sources/checked_at. It used to
    be an artificial `{"status": ...}` with no `sources` key at all, which
    the gate now completes; asserting that a hand-truncated update STAYS
    truncated was testing the absence of the guarantee, not the passthrough
    this test is named for.)"""
    stored_update = {"status": updates.CURRENT, "sources": [{"id": "a"}],
                     "checked_at": T0}
    entry = {**_entry(), "watch": [{"id": "a"}], "update": stored_update}
    data = {"oci:local:x": entry}
    described = provenance.describe(data, now=T0, stale_s=3600)[0]
    assert described["watch"] == [{"id": "a"}]
    assert described["update"] == stored_update


# --- watch / update ---------------------------------------------------------

def test_set_watch_validates_every_source(store):
    with pytest.raises(updates.BadWatch):
        store.set_watch("oci:local:x", [{"id": "a", "check": "nope", "pinned": "p"}])


def test_record_update_stores_the_rollup_and_the_sources(store):
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1.0.0", "order": "semver"},
        {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v2.0.0", "order": "semver"},
    ])
    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1.0.0",
         "latest": "v1.0.0", "detail": {}, "note": None},
        {"id": "b", "status": updates.AVAILABLE, "current": "v2.0.0",
         "latest": "v2.1.0", "detail": {}, "note": None},
    ])
    assert status == updates.AVAILABLE          # worst of the two
    entry = store.entry("oci:local:x")
    assert entry["update"]["status"] == updates.AVAILABLE
    assert len(entry["update"]["sources"]) == 2
    assert entry["update"]["checked_at"] is not None


def test_a_result_for_an_unwatched_source_is_dropped_not_retained(store):
    """Retention is bounded by watch. A verdict whose source no longer exists
    answers a question nobody is asking."""
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1.0.0", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None},
        {"id": "ghost", "status": updates.AVAILABLE, "current": "x",
         "latest": "y", "detail": {}, "note": None},
    ])
    ids = [s["id"] for s in store.entry("oci:local:x")["update"]["sources"]]
    assert ids == ["a"]


def test_shrinking_watch_drops_the_orphaned_verdict(store):
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"},
        {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v2", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None},
        {"id": "b", "status": updates.AVAILABLE, "current": "v2", "latest": "v3",
         "detail": {}, "note": None}])
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    entry = store.entry("oci:local:x")
    assert [s["id"] for s in entry["update"]["sources"]] == ["a"]
    assert entry["update"]["status"] == updates.CURRENT


def test_an_unavailable_source_keeps_its_previous_checked_at(store):
    """A transient outage must not erase yesterday's true answer."""
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.AVAILABLE, "current": "v1", "latest": "v2",
         "detail": {}, "note": None}], now="2026-08-01T00:00:00+00:00")
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.UNAVAILABLE, "current": "v1", "latest": None,
         "detail": {}, "note": "network"}], now="2026-08-02T00:00:00+00:00")

    source = store.entry("oci:local:x")["update"]["sources"][0]
    assert source["status"] == updates.AVAILABLE           # kept
    assert source["latest"] == "v2"
    assert source["checked_at"] == "2026-08-01T00:00:00+00:00"
    assert source["stale_note"] == "network"


def test_updates_available_lists_only_available_artifacts(store):
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.AVAILABLE, "current": "v1", "latest": "v2",
         "detail": {}, "note": None}])
    assert provenance.updates_available(store.get()) == ["oci:local:x"]


def test_updates_available_tolerates_entries_missing_the_update_key():
    # No "update" key at all -- a pre-migration entry, or one that has never
    # had a watch pass run against it.
    data = {"oci:local:x": {"artifact_id": "oci:local:x"}}
    assert provenance.updates_available(data) == []


def test_record_update_on_an_unknown_artifact_is_a_no_op(store):
    status = store.record_update("oci:local:ghost", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    assert status == updates.UNAVAILABLE
    assert store.get() == {}


def test_set_watch_is_a_no_op_when_the_sources_are_unchanged(tmp_path):
    """Task 9 calls this every collector pass with the same computed list.
    An unconditional record would grow the history file forever for a value
    that never moved -- the same non-action clear_desired already applies to
    an already-absent desired version."""
    store = _store(tmp_path)
    sources = [{"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
                "pinned": "v1", "order": "semver"}]
    store.set_watch("oci:local:x", sources, now=T0)
    store.set_watch("oci:local:x", list(sources), now=T1)

    records = [r for r in provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x") if r["field"] == "watch"]
    assert len(records) == 1


def test_set_watch_is_a_no_op_when_the_same_sources_arrive_in_a_different_order(tmp_path):
    """Task 9's collector builds its watch list from a dict-derived merge; if
    element order ever varies between passes an order-sensitive comparison
    would silently defeat the anti-spam guard and history would spam again."""
    store = _store(tmp_path)
    a = {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}
    b = {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v2", "order": "semver"}
    store.set_watch("oci:local:x", [a, b], now=T0)
    store.set_watch("oci:local:x", [b, a], now=T1)   # same sources, reordered

    records = [r for r in provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x") if r["field"] == "watch"]
    assert len(records) == 1


def test_set_watch_records_a_new_history_line_when_sources_actually_change(tmp_path):
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}], now=T0)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v2", "order": "semver"}], now=T1)

    records = [r for r in provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x") if r["field"] == "watch"]
    assert len(records) == 2


def test_set_watch_with_an_empty_list_on_an_unknown_artifact_is_a_true_no_op(tmp_path):
    store = _store(tmp_path)
    store.set_watch("oci:local:ghost", [])
    assert store.get() == {}
    assert provenance_history.history_for(tmp_path / "history.jsonl",
                                          "oci:local:ghost") == []


def test_set_watch_tolerates_a_pre_migration_entry_missing_watch_and_update(tmp_path):
    """Entries written before this task lack `watch`/`update` entirely --
    live production data is in exactly this shape on two machines."""
    store = _store(tmp_path)
    store.declare_origin(
        "oci:local:x", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "x", "reference": "x:1",
                "build": None, "archive": None}, now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    del data["oci:local:x"]["watch"]
    del data["oci:local:x"]["update"]
    path.write_text(json.dumps(data))

    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    entry = store.entry("oci:local:x")
    assert entry["watch"][0]["id"] == "a"
    assert entry["origin"]["repository"] == "x"   # untouched


def test_record_update_tolerates_a_pre_migration_entry_missing_watch(tmp_path):
    store = _store(tmp_path)
    store.declare_origin(
        "oci:local:x", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "x", "reference": "x:1",
                "build": None, "archive": None}, now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    del data["oci:local:x"]["watch"]
    del data["oci:local:x"]["update"]
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    assert status == updates.UNAVAILABLE          # nothing is watched: "a" is an orphan
    assert store.entry("oci:local:x")["update"]["sources"] == []


def test_a_corrupt_stored_status_does_not_corrupt_the_rollup(tmp_path):
    """A stored per-source status outside STATUSES (hand-edited or from a
    future format this code has never seen) must not leak past the rollup
    that reads it back."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])

    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["update"]["sources"][0]["status"] = "not-a-real-status"
    path.write_text(json.dumps(data))

    # Changing the watch list forces set_watch to recompute the rollup off
    # the (now corrupt) stored sources.
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"},
        {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v1", "order": "semver"}])

    status = store.entry("oci:local:x")["update"]["status"]
    assert status in updates.STATUSES
    assert status == updates.UNAVAILABLE


def test_a_first_ever_unavailable_result_is_recorded_not_dropped(store):
    """No prior verdict exists yet, so the unavailable result IS the only
    information there is: it must be stored with a real checked_at, not
    silently dropped and not backdated to a verdict that never existed."""
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.UNAVAILABLE, "current": "v1", "latest": None,
         "detail": {}, "note": "dns failure"}], now=T0)

    assert status == updates.UNAVAILABLE
    source = store.entry("oci:local:x")["update"]["sources"][0]
    assert source["status"] == updates.UNAVAILABLE
    assert source["checked_at"] == T0
    assert source["note"] == "dns failure"
    assert source["stale_note"] is None


def test_record_update_skips_a_stored_watch_item_missing_an_id(tmp_path):
    """A corrupted/hand-edited watch entry with no id must be dropped, never
    treated as a source literally named None -- which would let a result
    that ALSO lacks an id (also None) spuriously match it."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"].append(
        {"check": "git_tags", "remote": "https://github.com/x/y",
         "pinned": "v1", "order": "semver"})            # no "id" at all
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None},
        {"status": updates.AVAILABLE, "current": "x", "latest": "y",
         "detail": {}, "note": None},                    # also no "id"
    ])
    ids = [s["id"] for s in store.entry("oci:local:x")["update"]["sources"]]
    assert ids == ["a"]
    assert status == updates.CURRENT


def test_record_update_ignores_a_prior_verdict_missing_an_id(tmp_path):
    """A corrupted prior verdict with no id must not crash the merge, and
    must not be treated as the id-less incoming result's prior either."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    del data["oci:local:x"]["update"]["sources"][0]["id"]
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.UNAVAILABLE, "current": "v1", "latest": None,
         "detail": {}, "note": "network"}])
    assert status == updates.UNAVAILABLE
    source = store.entry("oci:local:x")["update"]["sources"][0]
    assert source["id"] == "a"
    assert source["status"] == updates.UNAVAILABLE   # no id-less prior to preserve


# --- fix round 2: non-dict elements in stored (untrusted) lists ------------

def test_set_watch_survives_a_non_dict_element_in_the_stored_watch_list(tmp_path):
    """`before` comes straight off disk and gets none of validate_watch's
    guarantees -- a hand-edited watch list can hold a bare `None` or string.
    _by_id must sort it deterministically rather than crashing on `.get`."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"] = [None, "garbage"]
    path.write_text(json.dumps(data))

    new_sources = [{"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
                    "pinned": "v1", "order": "semver"}]
    store.set_watch("oci:local:x", new_sources)          # must not raise

    entry = store.entry("oci:local:x")
    assert entry["watch"] == new_sources


def test_set_watch_order_insensitive_no_op_still_works_for_well_formed_input(tmp_path):
    """Regression guard for last round's fix: making _by_id tolerate
    non-dict elements must not break the ordinary case -- a genuinely
    unchanged, well-formed list (just reordered) must still no-op."""
    store = _store(tmp_path)
    sources = [{"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
                "pinned": "v1", "order": "semver"},
               {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
                "pinned": "v1", "order": "semver"}]
    store.set_watch("oci:local:x", sources, now=T0)
    store.set_watch("oci:local:x", list(reversed(sources)), now=T1)

    records = [r for r in provenance_history.history_for(
        tmp_path / "history.jsonl", "oci:local:x") if r["field"] == "watch"]
    assert len(records) == 1


def test_set_watch_retention_trim_survives_a_non_dict_element_in_stored_update_sources(tmp_path):
    """Same defect class, different stored list: update.sources read back
    during set_watch's retention trim can also be corrupted."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["update"]["sources"] = [None, "garbage"]
    path.write_text(json.dumps(data))

    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"},
        {"id": "b", "check": "git_tags", "remote": "https://github.com/c/d",
         "pinned": "v1", "order": "semver"}])                # must not raise

    assert store.entry("oci:local:x")["update"]["sources"] == []


def test_record_update_survives_a_non_dict_element_in_stored_watch(tmp_path):
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"].append(None)
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])                        # must not raise
    assert status == updates.CURRENT


def test_record_update_survives_a_non_dict_element_in_stored_update_sources(tmp_path):
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["update"]["sources"].append(None)
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.UNAVAILABLE, "current": "v1", "latest": None,
         "detail": {}, "note": "network"}])                   # must not raise
    # The garbage element is excluded from `previous`, so "a"'s real prior
    # (CURRENT) is still found and kept -- the unavailable result yields to it.
    assert status == updates.CURRENT
    ids = [s["id"] for s in store.entry("oci:local:x")["update"]["sources"]]
    assert ids == ["a"]


# --- fix round 3: whole-container corruption (non-dict `update`, ---------
# --- non-list `watch`/`update.sources`) ------------------------------------

@pytest.mark.parametrize("garbage_update", ["corrupted", [1, 2, 3], 42, True])
def test_record_update_survives_a_non_dict_stored_update_field(tmp_path, garbage_update):
    """`update` itself -- not just its "sources" list -- can be hand-edited
    into any JSON scalar or a list."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["update"] = garbage_update
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])                        # must not raise
    assert status == updates.CURRENT
    assert store.entry("oci:local:x")["update"]["sources"][0]["id"] == "a"


@pytest.mark.parametrize("garbage_watch", [5, True])
def test_set_watch_survives_a_truthy_non_list_stored_watch(tmp_path, garbage_watch):
    """The deeper pre-existing shape: a truthy non-list `watch` (`5`,
    `true`) must not raise TypeError from iterating it before any
    per-element check runs."""
    store = _store(tmp_path)
    store.declare_origin(
        "oci:local:x", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "x", "reference": "x:1",
                "build": None, "archive": None}, now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"] = garbage_watch
    path.write_text(json.dumps(data))

    new_sources = [{"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
                    "pinned": "v1", "order": "semver"}]
    store.set_watch("oci:local:x", new_sources)                # must not raise
    assert store.entry("oci:local:x")["watch"] == new_sources


@pytest.mark.parametrize("garbage_watch", [5, True])
def test_record_update_survives_a_truthy_non_list_stored_watch(tmp_path, garbage_watch):
    store = _store(tmp_path)
    store.declare_origin(
        "oci:local:x", kind="oci", node="local", role="engine",
        origin={"registry": "ghcr.io", "repository": "x", "reference": "x:1",
                "build": None, "archive": None}, now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["watch"] = garbage_watch
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])                        # must not raise
    assert status == updates.UNAVAILABLE          # nothing usable was watched
    assert store.entry("oci:local:x")["update"]["sources"] == []


def test_record_update_survives_a_non_list_stored_update_sources(tmp_path):
    """`update.sources` itself can be hand-edited to a non-list, not just
    contain non-dict elements."""
    store = _store(tmp_path)
    store.set_watch("oci:local:x", [
        {"id": "a", "check": "git_tags", "remote": "https://github.com/a/b",
         "pinned": "v1", "order": "semver"}])
    store.record_update("oci:local:x", [
        {"id": "a", "status": updates.CURRENT, "current": "v1", "latest": "v1",
         "detail": {}, "note": None}])
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:x"]["update"]["sources"] = "not-a-list"
    path.write_text(json.dumps(data))

    status = store.record_update("oci:local:x", [
        {"id": "a", "status": updates.UNAVAILABLE, "current": "v1", "latest": None,
         "detail": {}, "note": "network"}])                   # must not raise
    # The corrupted container means "a"'s real prior (CURRENT) is unreadable,
    # so the fresh unavailable result is recorded as-is -- honest, not a crash.
    assert status == updates.UNAVAILABLE
    assert store.entry("oci:local:x")["update"]["sources"][0]["id"] == "a"


def test_describe_survives_corrupted_watch_and_update_fields():
    """describe() reads watch/update through the same gate -- a corrupted
    container must not leak a raw scalar into an API response, and must not
    raise while building one."""
    data = {"oci:local:x": {**_entry(), "watch": "corrupted", "update": [1, 2]}}
    described = provenance.describe(data, now=T0, stale_s=3600)[0]
    assert described["watch"] == []
    assert described["update"] is None


def test_updates_available_survives_a_non_dict_entry():
    """A whole entry (not just a nested field) can be hand-edited into a
    non-dict; updates_available must skip it, not raise."""
    data = {"oci:local:x": "corrupted",
            "oci:local:y": {"update": {"status": updates.AVAILABLE}}}
    assert provenance.updates_available(data) == ["oci:local:y"]


# --- fix round 4: the ENTRY itself, and the gate that makes the next -------
# --- level structurally impossible -----------------------------------------

_AID = "oci:local:gated"
_NEIGHBOUR = "oci:local:healthy"
_SOURCE = {"id": "s", "check": "git_tags", "remote": "https://github.com/a/b",
           "pinned": "v1", "order": "semver"}
_RESULT = {"id": "s", "status": updates.AVAILABLE, "current": "v1",
           "latest": "v2", "detail": {}, "note": None}

# Every JSON shape except `null`, which several fields legitimately hold and
# which the old `(x or {})` idiom already handled. The last one is an object
# with nothing this module recognises in it -- a hand-edit, or a field from a
# version that has not been written yet.
_GARBAGE = ("corrupted", 42, True, ["not", "an", "object"], {"unexpected": True})

# The subset that is not an object at all: the shape three earlier rounds left
# reachable at the ENTRY level.
_NOT_AN_OBJECT = ("corrupted", 42, True, ["not", "an", "entry"])


def _origin(repository="a/b"):
    return {"registry": "ghcr.io", "repository": repository, "reference": None,
            "build": None, "archive": None}


@pytest.mark.parametrize("garbage", _NOT_AN_OBJECT)
def test_a_whole_entry_that_is_not_an_object_reads_as_an_empty_entry(tmp_path, garbage):
    """Required coverage 1, read side. `_load()` validates only the top-level
    document (D13); a hand-edited file can hold a bare scalar where an entry
    belongs. Every read path must degrade to "nothing recorded" -- and the id
    must survive, because an unidentifiable row in an API response is its own
    bug."""
    (tmp_path / "provenance.json").write_text(json.dumps({_AID: garbage}))
    data = _store(tmp_path).get()

    assert provenance.gaps(data) == [_AID]        # no readable origin: it IS a gap
    assert provenance.updates_available(data) == []
    described = provenance.describe(data, now=T0, stale_s=3600)
    assert [e["artifact_id"] for e in described] == [_AID]
    assert described[0]["watch"] == []
    assert described[0]["update"] is None
    assert described[0]["version_drift"] is False
    assert described[0]["verification"] == origins.UNKNOWN


@pytest.mark.parametrize("garbage", _NOT_AN_OBJECT)
def test_both_write_paths_survive_a_whole_entry_that_is_not_an_object(tmp_path, garbage):
    """Required coverage 1, write side: `set_watch` and `record_update` both
    read the entry with a bare `.get()` before this round, and both raised
    `AttributeError: 'str' object has no attribute 'get'` on this file."""
    (tmp_path / "provenance.json").write_text(json.dumps({_AID: garbage}))
    store = _store(tmp_path)

    store.set_watch(_AID, [_SOURCE], now=T0)
    assert store.record_update(_AID, [_RESULT], now=T1) == updates.AVAILABLE

    entry = store.entry(_AID)
    assert entry["artifact_id"] == _AID
    assert entry["kind"] == "oci" and entry["node"] == "local"
    assert entry["watch"] == [_SOURCE]
    assert entry["update"]["status"] == updates.AVAILABLE


def test_a_corrupt_entry_never_blocks_its_well_formed_neighbours(tmp_path):
    """Required coverage 2 -- the one that matters most. Three differently
    corrupted entries share the document with a healthy one; the healthy one
    must still read AND write exactly as if they were not there, and the
    corrupt ones must still be listed as gaps rather than disappearing."""
    store = _store(tmp_path)
    store.observe(_NEIGHBOUR, kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.declare_origin(_NEIGHBOUR, kind="oci", node="local", role="engine",
                         origin=_origin(), now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data["oci:local:scalar"] = "corrupted"
    data["oci:local:list"] = ["not", "an", "entry"]
    data["oci:local:number"] = 42
    path.write_text(json.dumps(data))

    store.set_watch(_NEIGHBOUR, [_SOURCE], now=T1)
    assert store.record_update(_NEIGHBOUR, [_RESULT], now=T1) == updates.AVAILABLE

    entry = store.entry(_NEIGHBOUR)
    assert entry["origin"] == _origin()                  # untouched
    assert entry["current"]["version"] == "sha256:a"     # untouched
    assert [s["id"] for s in entry["watch"]] == ["s"]
    assert entry["update"]["status"] == updates.AVAILABLE

    data = store.get()
    described = {e["artifact_id"]: e for e in provenance.describe(
        data, now=T1, stale_s=3600)}
    assert set(described) == {_NEIGHBOUR, "oci:local:scalar", "oci:local:list",
                              "oci:local:number"}
    assert described[_NEIGHBOUR]["update"]["status"] == updates.AVAILABLE
    assert provenance.updates_available(data) == [_NEIGHBOUR]
    assert provenance.gaps(data) == ["oci:local:list", "oci:local:number",
                                     "oci:local:scalar"]


def test_a_corrupt_entry_is_repaired_in_place_not_deleted(tmp_path):
    """The gate normalizes; it never drops. `delete()` stays the only way an
    entry leaves the store, so a corrupt entry survives a neighbour's write
    as a readable empty entry rather than vanishing from the ledger."""
    store = _store(tmp_path)
    store.declare_origin(_NEIGHBOUR, kind="oci", node="local", role="engine",
                         origin=_origin(), now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data[_AID] = "corrupted"
    path.write_text(json.dumps(data))

    store.set_desired(_NEIGHBOUR, version="sha256:b", label="v2", now=T1)

    on_disk = json.loads(path.read_text())
    assert set(on_disk) == {_NEIGHBOUR, _AID}
    assert on_disk[_AID]["artifact_id"] == _AID
    assert on_disk[_AID]["origin"] is None


# --- the structural guarantees themselves ----------------------------------

def _store_calls(store):
    """Every public ProvenanceStore method, with arguments that reach the
    stored entry for `_AID`. Keyed by name so the matrix below can assert the
    table is COMPLETE: a future method with no entry here fails the test
    rather than silently escaping the corruption matrix."""
    return {
        "get": lambda: store.get(),
        "entry": lambda: store.entry(_AID),
        "observe": lambda: store.observe(_AID, kind="oci", node="local",
                                         role="engine", current=_identity(),
                                         now=T1),
        "mark_unavailable": lambda: store.mark_unavailable(_AID, now=T1),
        "declare_origin": lambda: store.declare_origin(
            _AID, kind="oci", node="local", role="engine", origin=_origin(),
            update_path="docker pull", notes="hand declared", now=T1),
        "set_desired": lambda: store.set_desired(_AID, version="sha256:b",
                                                 label="v2", now=T1),
        "clear_desired": lambda: store.clear_desired(_AID, now=T1),
        "set_watch": lambda: store.set_watch(_AID, [_SOURCE], now=T1),
        "record_update": lambda: store.record_update(_AID, [_RESULT], now=T1),
        "record_deep_verify": lambda: store.record_deep_verify(
            _AID, "sha256:deadbeef", now=T1),
        "delete": lambda: store.delete(_AID),
    }


def _document_readers():
    """Every public module-level function that takes a stored document as its
    first argument, with the rest of its arguments. Same completeness trick as
    `_store_calls`: a new pure reader that forgets the gate fails the test."""
    return {"describe": {"now": T1, "stale_s": 3600},
            "gaps": {}, "updates_available": {}}


def _populated(tmp_path):
    """A store whose entry exercises every field the module can write."""
    store = _store(tmp_path)
    store.observe(_AID, kind="oci", node="local", role="engine",
                  current=_identity(), now=T0)
    store.declare_origin(_AID, kind="oci", node="local", role="engine",
                         origin=_origin(), update_path="docker pull",
                         notes="from the sparky-vllm README", now=T0)
    store.set_desired(_AID, version="sha256:b", label="v0.5.6", now=T0)
    store.set_watch(_AID, [_SOURCE], now=T0)
    store.record_update(_AID, [_RESULT], now=T0)
    store.observe(_NEIGHBOUR, kind="oci", node="local", role="engine",
                  current=_identity(version="sha256:n", label="v1"), now=T0)
    return store


def _paths(value, prefix=()):
    """Every addressable position inside `value`, itself included."""
    yield prefix
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _paths(item, (*prefix, index))


def _with_garbage_at(document, path, garbage):
    document = copy.deepcopy(document)
    target = document[_AID]
    if not path:
        document[_AID] = garbage
        return document
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = garbage
    return document


def test_the_corruption_matrix_covers_every_public_store_method(tmp_path):
    """THE COMPLETENESS GUARD. A method added to ProvenanceStore without a
    line in `_store_calls` fails here, which is what stops a fifth round of
    "one more un-gated read" from reaching production untested."""
    assert set(_store_calls(_store(tmp_path))) == {
        name for name in dir(ProvenanceStore) if not name.startswith("_")}


def test_the_document_reader_table_covers_every_public_pure_helper():
    """Same guard for the module-level read side: `describe`/`gaps`/
    `updates_available` take a document straight from a caller, so each has
    to gate its own argument -- and a fourth one has to be added here."""
    found = {name for name, fn in vars(provenance).items()
             if inspect.isfunction(fn) and not name.startswith("_")
             and fn.__module__ == provenance.__name__
             and next(iter(inspect.signature(fn).parameters), None) == "data"}
    assert found == set(_document_readers())


@pytest.mark.parametrize("garbage", _GARBAGE)
def test_no_read_or_write_path_raises_on_any_corrupted_position(tmp_path, garbage):
    """THE MATRIX. Every addressable position in a fully-populated entry --
    the entry itself, every field, every list element, every nested field --
    replaced by garbage in turn, against every public method of the store and
    every public module-level reader.

    It is driven by a REAL populated entry rather than a hand-written list of
    positions, so a field added later (nested or not) is corrupted by this
    test the day it is first written, without anyone remembering to."""
    path = tmp_path / "provenance.json"
    _populated(tmp_path)
    healthy = json.loads(path.read_text())

    for position in _paths(healthy[_AID]):
        for name, call in _store_calls(_store(tmp_path)).items():
            path.write_text(json.dumps(_with_garbage_at(healthy, position, garbage)))
            try:
                call()
            except Exception as exc:   # noqa: BLE001 -- re-raised with a locator
                raise AssertionError(
                    f"{name}() raised on {list(position)} = {garbage!r}: {exc!r}"
                ) from exc

        path.write_text(json.dumps(_with_garbage_at(healthy, position, garbage)))
        data = _store(tmp_path).get()
        for name, kwargs in _document_readers().items():
            try:
                getattr(provenance, name)(data, **kwargs)
            except Exception as exc:   # noqa: BLE001 -- re-raised with a locator
                raise AssertionError(
                    f"{name}() raised on {list(position)} = {garbage!r}: {exc!r}"
                ) from exc


@pytest.mark.parametrize("garbage", _GARBAGE)
def test_the_healthy_neighbour_is_untouched_by_every_corrupted_position(
        tmp_path, garbage):
    """Required coverage 2, generalized over the whole matrix: whatever is
    corrupted in one entry, the entry beside it still reads back exactly."""
    path = tmp_path / "provenance.json"
    _populated(tmp_path)
    healthy = json.loads(path.read_text())
    expected = healthy[_NEIGHBOUR]

    for position in _paths(healthy[_AID]):
        path.write_text(json.dumps(_with_garbage_at(healthy, position, garbage)))
        store = _store(tmp_path)
        assert store.entry(_NEIGHBOUR) == expected, position
        store.set_watch(_NEIGHBOUR, [_SOURCE], now=T1)
        assert store.record_update(_NEIGHBOUR, [_RESULT], now=T1) == updates.AVAILABLE
        assert store.entry(_NEIGHBOUR)["current"]["version"] == "sha256:n"


def test_only_load_reads_the_stored_document(tmp_path):
    """WHY THE GATE HOLDS: `_load()` is the single door stored data comes
    through, so gating it gates every method that will ever exist. This test
    fails if a future method reads the file itself and walks around the
    gate."""
    for name, member in vars(ProvenanceStore).items():
        if name == "_load" or not inspect.isfunction(member):
            continue
        source = inspect.getsource(member)
        assert "read_text" not in source, name
        assert "json.loads" not in source, name


def test_the_shape_table_declares_every_container_field_of_an_entry():
    """The gate's field table is keyed off `_blank_entry`'s own vocabulary.
    A future entry field whose default is a container must be declared in it,
    or a stored value of the wrong type reaches a caller un-normalized."""
    blank = provenance._blank_entry("oci:local:x", "oci", "local", "other")
    containers = {k for k, v in blank.items() if isinstance(v, (dict, list))}
    assert containers <= set(provenance._ENTRY_SHAPE)
    assert set(provenance._ENTRY_SHAPE) <= set(blank)


def test_load_gates_every_entry_including_unparseable_ids():
    """The gate cannot assume the KEY is a well-formed artifact id either --
    the file is hand-editable all the way up."""
    document = {"not-an-artifact-id": "corrupted",
                "oci:local:x": {"watch": 5, "update": "x", "current": 7,
                                "desired": True, "origin": ["nope"]}}
    gated = provenance._stored_document(document)

    assert set(gated) == set(document)
    assert gated["not-an-artifact-id"]["kind"] is None      # never guessed
    assert gated["oci:local:x"]["kind"] == "oci"            # derived from the key
    assert gated["oci:local:x"]["watch"] == []
    assert gated["oci:local:x"]["update"] is None
    assert gated["oci:local:x"]["current"] == provenance._empty_current()
    assert gated["oci:local:x"]["desired"] is None
    assert gated["oci:local:x"]["origin"] is None


def test_the_gate_keeps_every_readable_value_it_is_given():
    """NOT TOO AGGRESSIVE. Only a value whose stored shape cannot be read at
    all is replaced; everything else -- including keys this module has never
    heard of -- survives byte for byte."""
    stored = {"artifact_id": "oci:local:x", "kind": "oci", "node": "local",
              "role": "engine", "origin": _origin(), "current": _identity(),
              "desired": {"version": "v2"}, "update_path": "docker pull",
              "notes": "n", "watch": [_SOURCE],
              "update": {"status": updates.CURRENT, "sources": [_RESULT],
                         "checked_at": T0},
              "a_field_from_a_later_version": {"kept": True}}
    gated = provenance._stored_document({"oci:local:x": stored})["oci:local:x"]

    for field, value in stored.items():
        if field == "current":
            continue           # merged onto the empty-current defaults, below
        assert gated[field] == value, field
    assert gated["current"] == {**provenance._empty_current(), **_identity()}


def test_a_verdict_that_cannot_be_keyed_is_dropped_but_a_declared_source_is_kept():
    """DECLARED vs DERIVED. An id is what `record_update` builds a set and a
    dict from, and an id that is itself a container is unhashable --
    `{s["id"] for s in ...}` raises TypeError on it, not AttributeError.
    Derived verdicts with no usable id are dropped at the gate (the next
    pass rewrites them anyway); a DECLARED watch source is kept, because
    this file is its only home and every write re-serialises the document --
    dropping it here would erase an operator's typo instead of showing it."""
    document = {"oci:local:x": {"watch": [{"id": ["a"]}, {"id": "b"},
                                          {"no": "id"}],
                                "update": {"sources": [{"id": {"a": 1}},
                                                       {"id": "b"}]}}}
    gated = provenance._stored_document(document)["oci:local:x"]

    assert gated["watch"] == [{"id": ["a"]}, {"id": "b"}, {"no": "id"}]
    assert [s["id"] for s in gated["update"]["sources"]] == ["b"]


def test_an_unkeyable_declared_source_still_cannot_reach_a_set_comprehension(tmp_path):
    """The other half of that asymmetry: keeping the source visible must not
    put an unhashable id in front of `record_update`'s `live` set."""
    store = _store(tmp_path)
    store.set_watch(_AID, [_SOURCE], now=T0)
    path = tmp_path / "provenance.json"
    data = json.loads(path.read_text())
    data[_AID]["watch"] += [{"id": ["unhashable"]}, {"no": "id at all"}]
    path.write_text(json.dumps(data))

    assert store.record_update(_AID, [_RESULT], now=T1) == updates.AVAILABLE
    entry = store.entry(_AID)
    assert [s["id"] for s in entry["update"]["sources"]] == ["s"]
    assert len(entry["watch"]) == 3            # the operator's own text, kept
