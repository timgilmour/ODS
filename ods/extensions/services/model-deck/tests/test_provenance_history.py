"""Durable provenance history: append-only, fsync'd, torn-tail tolerant."""

import json

from app import provenance_history as history


def _rec(ts, artifact_id="oci:local:x", to=None):
    return {"ts": ts, "artifact_id": artifact_id, "field": "current",
            "from": None, "to": to or {"version": ts}, "cause": "observed",
            "actor": "watcher"}


def test_append_then_read_round_trips(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    history.append(path, _rec("2026-08-02T00:00:00+00:00"))
    assert [r["ts"] for r in history.read_all(path)] == [
        "2026-08-01T00:00:00+00:00", "2026-08-02T00:00:00+00:00"]


def test_missing_file_reads_as_empty(tmp_path):
    assert history.read_all(tmp_path / "nope.jsonl") == []


def test_append_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "er" / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    assert len(history.read_all(path)) == 1


def test_a_torn_tail_costs_exactly_one_record(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    history.append(path, _rec("2026-08-02T00:00:00+00:00"))
    text = path.read_text()
    path.write_text(text[:-12])          # simulate a crash mid-write
    assert [r["ts"] for r in history.read_all(path)] == ["2026-08-01T00:00:00+00:00"]


def test_a_torn_tail_does_not_corrupt_the_next_append(tmp_path):
    # Without newline repair, the next append concatenates onto the partial
    # line and destroys TWO records instead of one.
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    path.write_text(path.read_text().rstrip("\n")[:-5])
    history.append(path, _rec("2026-08-03T00:00:00+00:00"))
    assert [r["ts"] for r in history.read_all(path)] == ["2026-08-03T00:00:00+00:00"]


def test_append_fsyncs(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(history.os, "fsync", lambda fd: synced.append(fd))
    history.append(tmp_path / "h.jsonl", _rec("2026-08-01T00:00:00+00:00"))
    assert synced


def test_history_for_filters_by_artifact_and_keeps_order(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00", "oci:local:a"))
    history.append(path, _rec("2026-08-02T00:00:00+00:00", "oci:local:b"))
    history.append(path, _rec("2026-08-03T00:00:00+00:00", "oci:local:a"))
    assert [r["ts"] for r in history.history_for(path, "oci:local:a")] == [
        "2026-08-01T00:00:00+00:00", "2026-08-03T00:00:00+00:00"]


def test_state_at_returns_the_last_block_at_or_before_the_timestamp(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00", to={"version": "v0.5.3"}))
    history.append(path, _rec("2026-08-05T00:00:00+00:00", to={"version": "v0.5.6"}))
    at = history.state_at(path, "oci:local:x", "2026-08-04T00:00:00+00:00")
    assert at == {"version": "v0.5.3"}


def test_state_at_before_any_record_is_none(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-05T00:00:00+00:00"))
    assert history.state_at(path, "oci:local:x", "2026-08-01T00:00:00+00:00") is None


def test_state_at_ignores_other_fields(tmp_path):
    path = tmp_path / "h.jsonl"
    rec = _rec("2026-08-01T00:00:00+00:00", to={"version": "desired-only"})
    rec["field"] = "desired"
    history.append(path, rec)
    assert history.state_at(path, "oci:local:x", "2026-08-09T00:00:00+00:00") is None


def test_unparseable_lines_are_skipped_not_raised(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    with path.open("a") as f:
        f.write("{not json}\n")
    history.append(path, _rec("2026-08-02T00:00:00+00:00"))
    assert len(history.read_all(path)) == 2


def test_records_are_written_one_per_line(tmp_path):
    path = tmp_path / "h.jsonl"
    history.append(path, _rec("2026-08-01T00:00:00+00:00"))
    history.append(path, _rec("2026-08-02T00:00:00+00:00"))
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all(json.loads(ln)["artifact_id"] == "oci:local:x" for ln in lines)
