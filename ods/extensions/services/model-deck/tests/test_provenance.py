"""The provenance store: atomic current+desired state, drift, gaps."""

import json

import pytest

from app import origins, provenance, provenance_history
from app.provenance import ProvenanceStore

T0 = "2026-08-01T00:00:00+00:00"
T1 = "2026-08-02T00:00:00+00:00"


def _store(tmp_path):
    return ProvenanceStore(tmp_path / "provenance.json", tmp_path / "history.jsonl")


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
