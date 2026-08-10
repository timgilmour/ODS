"""Tests for app.characteristics — the machine-owned derived fact cache.

Every field carries where it came from and when. Provenance is not
decoration: an operator looking at 'context: 262144' must be able to tell
whether that was read from the checkpoint, read from a live engine, or
asserted by a human, because those three have very different reliability.
"""

import json

import pytest

from app.characteristics import CharacteristicsStore


def test_get_empty_when_missing(tmp_path):
    assert CharacteristicsStore(tmp_path / "c.json").get() == {}


def test_get_empty_when_corrupt(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("]not json[")

    assert CharacteristicsStore(path).get() == {}


def test_put_fields_stamps_nothing_it_was_not_given(tmp_path):
    """The store persists what the deriver computed; it does not invent
    provenance. A deriver that forgets to stamp is a bug we want visible."""
    store = CharacteristicsStore(tmp_path / "c.json")

    store.put_fields("model/Qwen3.6-35B-A3B-heretic-NVFP4", {
        "quant_method": {"value": "compressed-tensors",
                         "source": "config.json",
                         "derived_ts": "2026-08-04T00:00:00+00:00"},
    })

    entry = store.entry("model/Qwen3.6-35B-A3B-heretic-NVFP4")
    assert entry["quant_method"] == {
        "value": "compressed-tensors",
        "source": "config.json",
        "derived_ts": "2026-08-04T00:00:00+00:00",
    }


def test_put_fields_merges_rather_than_replaces(tmp_path):
    """Two derivers write different fields of the same entry (checkpoint
    facts and live facts). The second must not erase the first."""
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("model/m", {"a": {"value": 1, "source": "config.json", "derived_ts": "t"}})

    store.put_fields("model/m", {"b": {"value": 2, "source": "/v1/models", "derived_ts": "t"}})

    assert set(store.entry("model/m")) == {"a", "b"}


def test_put_fields_overwrites_the_same_field(tmp_path):
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("model/m", {"a": {"value": 1, "source": "s", "derived_ts": "t1"}})

    store.put_fields("model/m", {"a": {"value": 9, "source": "s", "derived_ts": "t2"}})

    assert store.entry("model/m")["a"]["value"] == 9
    assert store.entry("model/m")["a"]["derived_ts"] == "t2"


def test_entry_of_unknown_key_is_empty(tmp_path):
    assert CharacteristicsStore(tmp_path / "c.json").entry("nope") == {}


def test_forget_removes_entry(tmp_path):
    store = CharacteristicsStore(tmp_path / "c.json")
    store.put_fields("model/m", {"a": {"value": 1, "source": "s", "derived_ts": "t"}})

    store.forget("model/m")

    assert store.get() == {}


def test_write_is_atomic(tmp_path):
    store = CharacteristicsStore(tmp_path / "c.json")

    store.put_fields("model/m", {"a": {"value": 1, "source": "s", "derived_ts": "t"}})

    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_rejects_a_field_missing_provenance(tmp_path):
    """Cheap guard against a deriver silently writing bare values."""
    store = CharacteristicsStore(tmp_path / "c.json")

    try:
        store.put_fields("model/m", {"a": {"value": 1}})
    except ValueError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rejects_a_non_dict_field(tmp_path):
    """``k not in field`` does substring matching on a str, so a caller that
    passes a string instead of a field record can slip past the provenance
    check entirely (e.g. "value" in "source,value,derived_ts" is True). Guard
    the type explicitly rather than relying on `in`'s duck typing."""
    store = CharacteristicsStore(tmp_path / "c.json")

    with pytest.raises(ValueError, match="a"):
        store.put_fields("model/m", {"a": "source,value,derived_ts"})


def test_stored_json_is_human_readable(tmp_path):
    """These files get read by a human during an incident."""
    path = tmp_path / "c.json"
    CharacteristicsStore(path).put_fields(
        "model/m", {"a": {"value": 1, "source": "s", "derived_ts": "t"}})

    assert "\n" in path.read_text()
    assert json.loads(path.read_text())


def test_put_fields_concurrent_writers_lose_nothing(tmp_path):
    """Unlocked load-modify-save [max-review c4]: the watcher thread's derive
    pass and HTTP request threads share ONE CharacteristicsStore, and
    put_fields read the file, mutated, and wrote it back with nothing
    serializing the three steps. Two threads writing DIFFERENT keys must both
    land. Barrier-started so the interleave is real rather than incidental.

    Unlocked, this fails two ways at once: a stale read swallows the other
    thread's write, AND _save's atomic replace races itself — both threads
    write the SAME `.tmp` path, so one os.replace hits a file the other
    already moved (FileNotFoundError out of a background derive pass). Hence
    asserting both that no thread raised and that every field landed.
    IntentStore and SettingsStore already lock; this was the odd one out.
    """
    import threading

    store = CharacteristicsStore(tmp_path / "characteristics.json")
    start = threading.Barrier(2)
    errors = []

    def writer(key, value):
        try:
            start.wait(timeout=5)
            for i in range(60):
                store.put_fields(key, {f"f{i}": {"value": value, "source": "t",
                                                 "derived_ts": "2026-08-10"}})
        except Exception as exc:  # surfaced below; a raise here would be silent
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k, k)) for k in ("alpha", "beta")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    data = store.get()
    assert len(data.get("alpha", {})) == 60
    assert len(data.get("beta", {})) == 60
