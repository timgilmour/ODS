"""Tests for app.events — the append-only JSONL event log.

log_event() appends one line per call: {"ts": <UTC ISO8601 w/ tz>, "kind":
kind, "detail": detail}. tail_events() reads the last n parsed entries back,
oldest->newest, skipping any line that fails to parse as JSON; a missing
file reads as an empty list rather than raising, matching the "display-only,
self-healing" quality bar used elsewhere in this extension (registry.py,
policy.py).
"""

import json
from datetime import datetime

from app.events import log_event, tail_events


def test_log_event_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "does" / "not" / "exist" / "events.jsonl"

    log_event(path, "tenant_loaded", {"tenant": "lemonade"})

    assert path.is_file()


def test_log_event_appends_one_json_line_with_ts_kind_detail(tmp_path):
    path = tmp_path / "events.jsonl"

    log_event(path, "tenant_loaded", {"tenant": "lemonade", "model": "foo.gguf"})

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "tenant_loaded"
    assert record["detail"] == {"tenant": "lemonade", "model": "foo.gguf"}
    assert "ts" in record


def test_log_event_ts_is_utc_iso8601_with_timezone(tmp_path):
    path = tmp_path / "events.jsonl"

    log_event(path, "tenant_loaded", {})

    record = json.loads(path.read_text().splitlines()[0])
    parsed = datetime.fromisoformat(record["ts"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_log_event_multiple_calls_append_not_overwrite(tmp_path):
    path = tmp_path / "events.jsonl"

    log_event(path, "a", {"n": 1})
    log_event(path, "b", {"n": 2})
    log_event(path, "c", {"n": 3})

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["kind"] for line in lines] == ["a", "b", "c"]


def test_tail_events_returns_oldest_to_newest(tmp_path):
    path = tmp_path / "events.jsonl"
    for i in range(5):
        log_event(path, f"kind{i}", {"n": i})

    result = tail_events(path)

    assert [entry["kind"] for entry in result] == [f"kind{i}" for i in range(5)]


def test_tail_events_respects_n_limit(tmp_path):
    path = tmp_path / "events.jsonl"
    for i in range(10):
        log_event(path, f"kind{i}", {"n": i})

    result = tail_events(path, n=3)

    assert [entry["kind"] for entry in result] == ["kind7", "kind8", "kind9"]


def test_tail_events_skips_unparseable_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    log_event(path, "good1", {})
    with path.open("a") as f:
        f.write("{not valid json\n")
        f.write("\n")  # blank line
    log_event(path, "good2", {})

    result = tail_events(path)

    assert [entry["kind"] for entry in result] == ["good1", "good2"]


def test_tail_events_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "does-not-exist.jsonl"

    assert tail_events(path) == []


def test_tail_events_default_n_is_100(tmp_path):
    path = tmp_path / "events.jsonl"
    for i in range(150):
        log_event(path, f"kind{i}", {"n": i})

    result = tail_events(path)

    assert len(result) == 100
    assert result[0]["kind"] == "kind50"
    assert result[-1]["kind"] == "kind149"
