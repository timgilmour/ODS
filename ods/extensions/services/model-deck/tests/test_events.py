"""Tests for app.events — the rotating JSONL event log.

log_event() appends one line per call: {"ts": <UTC ISO8601 w/ tz>, "kind":
kind, "detail": detail}, and rotates once the file passes _MAX_LOG_BYTES
(rewritten to its newest _TRIM_TO_LINES lines — the log is display-only, so
old lines are expendable). tail_events() reads the last n parsed entries back,
oldest->newest, skipping any line that fails to parse as JSON; a missing
file reads as an empty list rather than raising, matching the "display-only,
self-healing" quality bar used elsewhere in this extension (registry.py,
policy.py).
"""

import json
from datetime import datetime

from app.events import _MAX_LOG_BYTES, _TRIM_TO_LINES, log_event, tail_events


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


def test_log_event_trims_an_oversized_log(tmp_path):
    """[max-review c7] The log was append-only forever, and tail_events reads
    the WHOLE file to return its last n — so a long-lived deck paid a growing
    read on every Events poll. Rotation keeps the newest lines and drops the
    rest; the log is display-only, so old lines are expendable in a way a
    state file's contents never are.
    """
    path = tmp_path / "events.jsonl"
    pad = "x" * 1000
    for i in range(6000):                     # ~6 MB, past _MAX_LOG_BYTES
        log_event(path, "k", {"i": i, "pad": pad})

    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    # The bound is the BYTE cap, not the line count. Trimming to
    # _TRIM_TO_LINES does not pin the file there — writing continues after a
    # trim, so the log climbs back toward _MAX_LOG_BYTES and trims again.
    # Asserting `len(lines) <= _TRIM_TO_LINES` would only hold if a trim
    # happened to land on the very last write.
    assert path.stat().st_size <= _MAX_LOG_BYTES + 2000  # + one record's slack
    # ...and a trim demonstrably HAPPENED: all 6000 would otherwise be here.
    assert len(lines) < 6000
    assert len(lines) >= _TRIM_TO_LINES
    # The NEWEST survive — a trim that kept the head would silently freeze
    # the Events tab at the deck's first few thousand events.
    assert json.loads(lines[-1])["detail"]["i"] == 5999
    assert tail_events(path, 1)[0]["detail"]["i"] == 5999


def test_trim_keeps_the_log_parseable(tmp_path):
    """A trim that split a line would poison every later tail_events read
    (they skip unparseable lines, so the damage would be silent)."""
    path = tmp_path / "events.jsonl"
    pad = "y" * 1000
    for i in range(6000):
        log_event(path, "k", {"i": i, "pad": pad})

    text = path.read_text()
    assert text.endswith("\n")
    for line in text.splitlines():
        json.loads(line)          # raises if the trim corrupted anything


def test_small_log_is_left_alone(tmp_path):
    """Rotation must not touch a log below the cap — the common case."""
    path = tmp_path / "events.jsonl"
    for i in range(50):
        log_event(path, "k", {"i": i})

    assert len(path.read_text().splitlines()) == 50
    assert tail_events(path, 100)[0]["detail"]["i"] == 0


def test_concurrent_log_event_never_crashes_on_rotation(tmp_path):
    """The trim writes ONE fixed `.trim-tmp` path per log, and log_event is
    reachable from essentially every thread the deck runs (arbiter watcher,
    storage watcher, mover, update-checker, and all sync HTTP routes on
    FastAPI's threadpool). Unlocked, two concurrent trims race that path and
    the loser's os.replace raises FileNotFoundError into whichever thread it
    was — a 500 on a route, or a dead watcher pass.

    Sized so several rotations happen while both threads are writing.
    """
    import threading

    path = tmp_path / "events.jsonl"
    pad = "z" * 1000
    start = threading.Barrier(2)
    errors = []

    def writer(tag):
        try:
            start.wait(timeout=5)
            for i in range(4000):
                log_event(path, "k", {"who": tag, "i": i, "pad": pad})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    # Still a well-formed log afterwards — a torn trim would strand a partial
    # line, and tail_events silently skips those, so assert on the raw text.
    for line in path.read_text().splitlines():
        json.loads(line)


def test_large_records_do_not_thrash_the_trim(tmp_path, monkeypatch):
    """[T10 review] The trigger is bytes; a LINE-only trim target has no
    hysteresis above _MAX_LOG_BYTES/_TRIM_TO_LINES (2500 B/line) — the file
    stays oversized after each trim, so every subsequent append rewrites the
    whole multi-MB log.

    Reachable, not theoretical: engines raise EngineError(resp.text) carrying
    a raw HTTP body, and the arbiter logs `load-failed` with str(exc)
    UN-deduped every 2 s tick, so one degraded backend thrashes the log for
    the duration of an outage.

    Fixture is deliberately AWAY from the other rotation tests' 1000-byte
    payloads — those sit under the ratio, which is exactly why the suite
    could not see this.
    """
    import app.events as events_mod

    path = tmp_path / "events.jsonl"
    pad = "q" * 3000                       # ~3 KB/line, above the 2500 ratio

    trims = []
    real_trim = events_mod._trim
    monkeypatch.setattr(events_mod, "_trim",
                        lambda p: (trims.append(1), real_trim(p))[1])

    while not trims:                       # write until the first rotation
        log_event(path, "load-failed", {"error": pad})
    first = len(trims)

    for _ in range(20):                    # 20 more appends after that trim
        log_event(path, "load-failed", {"error": pad})

    # Thrashing would trim on essentially every one of those 20 appends.
    assert len(trims) - first <= 1
    assert path.stat().st_size <= _MAX_LOG_BYTES
