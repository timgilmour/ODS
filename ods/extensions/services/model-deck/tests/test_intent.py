"""Tests for app.intent — the durable desired-state store.

intent.json is a flat mapping of {"<node>/<resource>": record}. Unlike
policy.json there are NO known-key defaults: the keys are discovered at
runtime, which is what makes new resources (a second node, a new engine)
work without a code change. A missing or corrupt file reads as empty and
self-heals, matching PolicyStore's quality bar.
"""

import json

from app.intent import IntentStore


def test_get_empty_when_file_missing(tmp_path):
    store = IntentStore(tmp_path / "intent.json")

    assert store.get() == {}


def test_get_empty_when_file_corrupt(tmp_path):
    path = tmp_path / "intent.json"
    path.write_text("{not json")
    store = IntentStore(path)

    assert store.get() == {}


def test_record_persists_full_record(tmp_path):
    path = tmp_path / "intent.json"
    store = IntentStore(path)

    store.record("local/hipfire", state="loaded", model="gpt-oss-120b",
                 engine="hipfire", now="2026-08-04T00:00:00+00:00")

    assert json.loads(path.read_text()) == {
        "local/hipfire": {
            "state": "loaded",
            "model": "gpt-oss-120b",
            "engine": "hipfire",
            "updated_ts": "2026-08-04T00:00:00+00:00",
            "last_healthy_ts": None,
            "failures": 0,
            "quarantined": False,
        }
    }


def test_record_unloaded_is_intent_not_deletion(tmp_path):
    """The #1 invariant: a deliberate unload is a recorded intent, so the
    reconciler can tell 'parked on purpose' from 'died'."""
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/lemonade", state="loaded", model="qwen", engine="lemonade")

    store.record("local/lemonade", state="unloaded", model=None, engine="lemonade")

    assert store.get()["local/lemonade"]["state"] == "unloaded"


def test_record_arbitrary_new_key_accepted(tmp_path):
    """No known-tenant allowlist — a new node/resource just works."""
    store = IntentStore(tmp_path / "intent.json")

    store.record("sparky/slot0", state="loaded", model="heretic", engine="vllm")

    assert "sparky/slot0" in store.get()


def test_record_preserves_last_healthy_but_resets_failure_budget(tmp_path):
    """last_healthy_ts is the resource's own health history and survives a
    re-record; the failure budget does not. A deliberate action is the signal
    that the situation changed, so the resource earns a fresh budget."""
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")
    store.note_healthy("local/hipfire", now="2026-08-04T01:00:00+00:00")
    store.note_failure("local/hipfire")

    store.record("local/hipfire", state="loaded", model="b", engine="hipfire")

    record = store.get()["local/hipfire"]
    assert record["last_healthy_ts"] == "2026-08-04T01:00:00+00:00"
    assert record["failures"] == 0
    assert record["quarantined"] is False


def test_record_releases_a_quarantine(tmp_path):
    """Without this the spent budget is permanent — and once the resource is
    parked it is invisible too, since derive_status only reports quarantine on
    the loaded-intent branch."""
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")
    store.note_failure("local/hipfire")
    store.note_failure("local/hipfire")
    assert store.get()["local/hipfire"]["quarantined"] is True

    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    record = store.get()["local/hipfire"]
    assert record["quarantined"] is False
    assert record["failures"] == 0


def test_note_failure_returns_running_count_and_quarantines_at_two(tmp_path):
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    assert store.note_failure("local/hipfire") == 1
    assert store.get()["local/hipfire"]["quarantined"] is False
    assert store.note_failure("local/hipfire") == 2
    assert store.get()["local/hipfire"]["quarantined"] is True


def test_note_healthy_clears_failures_and_quarantine(tmp_path):
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")
    store.note_failure("local/hipfire")
    store.note_failure("local/hipfire")

    store.note_healthy("local/hipfire", now="2026-08-04T02:00:00+00:00")

    record = store.get()["local/hipfire"]
    assert record["failures"] == 0
    assert record["quarantined"] is False
    assert record["last_healthy_ts"] == "2026-08-04T02:00:00+00:00"


def test_note_failure_on_unknown_key_is_a_noop(tmp_path):
    store = IntentStore(tmp_path / "intent.json")

    assert store.note_failure("ghost/slot0") == 0
    assert store.get() == {}


def test_forget_removes_the_key(tmp_path):
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    store.forget("local/hipfire")

    assert store.get() == {}


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    path = tmp_path / "intent.json"
    store = IntentStore(path)

    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    assert [p.name for p in tmp_path.iterdir()] == ["intent.json"]


def test_parent_dir_created(tmp_path):
    path = tmp_path / "nested" / "deep" / "intent.json"
    store = IntentStore(path)

    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    assert path.exists()
