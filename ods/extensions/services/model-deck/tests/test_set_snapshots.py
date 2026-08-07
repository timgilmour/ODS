"""Sets must be fully reproducible (Tim, 2026-08-04; whole-store ruling
2026-08-07): a saved set carries the entire settings store. The failure
mode reproducibility buys is staleness, so opening a set diffs its
snapshot against the live store and adopting is explicit."""

from app.sets import diff_snapshot


def _store(**engines_args):
    return {"engines": {k: {"args": v, "updated_ts": {"args": "2026-08-07T00:00:00Z"}}
                        for k, v in engines_args.items()},
            "models": {}, "engine_models": {}}


def test_identical_stores_produce_no_diff():
    a = _store(**{"sparky/vllm": {"x": "1"}})

    assert diff_snapshot(a, _store(**{"sparky/vllm": {"x": "1"}})) == {
        "changed": [], "added": [], "removed": []}


def test_updated_ts_and_notes_never_count_as_drift():
    a = _store(**{"sparky/vllm": {"x": "1"}})
    b = _store(**{"sparky/vllm": {"x": "1"}})
    b["engines"]["sparky/vllm"]["updated_ts"]["args"] = "2026-08-07T09:99:99Z"
    b["engines"]["sparky/vllm"]["notes"] = {"args": "later note"}

    assert diff_snapshot(a, b) == {"changed": [], "added": [], "removed": []}


def test_changed_value_reports_both_sides_with_qualified_key():
    diff = diff_snapshot(_store(**{"sparky/vllm": {"x": "1"}}),
                         _store(**{"sparky/vllm": {"x": "2"}}))

    assert diff["changed"] == [{"scope": "engines/sparky/vllm",
                                "key": "args:x", "snapshot": "1", "current": "2"}]


def test_added_and_removed_scopes():
    diff = diff_snapshot(_store(**{"a/e": {"x": "1"}}), _store(**{"b/e": {"y": "2"}}))

    assert diff["added"] == [{"scope": "engines/b/e", "key": "args:y"}]
    assert diff["removed"] == [{"scope": "engines/a/e", "key": "args:x"}]


def test_none_snapshot_diffs_empty():
    """An old set (settings_snapshot=None) has nothing to diff against --
    diff_snapshot itself stays silent (has_snapshot lives at the route)."""
    assert diff_snapshot(None, _store(**{"sparky/vllm": {"x": "1"}})) == {
        "changed": [], "added": [], "removed": []}
