"""Tests for app.intent — the durable desired-state store.

intent.json is a flat mapping of {"<node>/<resource>": record}. Unlike
policy.json there are NO known-key defaults: the keys are discovered at
runtime, which is what makes new resources (a second node, a new engine)
work without a code change. A missing or corrupt file reads as empty and
self-heals, matching PolicyStore's quality bar.
"""

import json

import pytest

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
            "actor": "operator",  # default — see test_record_actor_defaults_to_operator
            "updated_ts": "2026-08-04T00:00:00+00:00",
            "last_healthy_ts": None,
            "failures": 0,
            "quarantined": False,
        }
    }


def test_record_actor_defaults_to_operator(tmp_path):
    """Deliberate default (task 6 follow-up, max-review Important-1): a
    forgotten ``actor`` kwarg must fail SAFE for app.routers.control's
    pull-through supersession check — "operator" is the value that check
    treats as ABLE to supersede, so an accidental omission still protects an
    operator's later action rather than silently becoming invisible to it."""
    store = IntentStore(tmp_path / "intent.json")

    store.record("local/hipfire", state="loaded", model="a", engine="hipfire")

    assert store.get()["local/hipfire"]["actor"] == "operator"


def test_record_accepts_explicit_actor_deck(tmp_path):
    """The one caller class that overrides the default: app.arbiter's own
    two automatic records (idle-release/contention-eviction unload,
    pending-load retrigger)."""
    store = IntentStore(tmp_path / "intent.json")

    store.record("local/lemonade", state="unloaded", model=None,
                 engine="lemonade", actor="deck")

    assert store.get()["local/lemonade"]["actor"] == "deck"


def test_record_rejects_invalid_actor(tmp_path):
    store = IntentStore(tmp_path / "intent.json")

    with pytest.raises(ValueError):
        store.record("local/lemonade", state="loaded", model="a",
                     engine="lemonade", actor="bogus")

    assert store.get() == {}  # refused before any write, like the state guard


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


def test_health_bookkeeping_never_touches_actor_or_updated_ts(tmp_path):
    """note_failure/note_healthy mutate FIELDS on the existing record; they
    must never rebuild it. A rebuild would default ``actor`` back to
    "operator" (relabeling the arbiter's own deck records, which
    app.routers.control's pull-through supersession check reads) and re-stamp
    ``updated_ts`` (the settings-drift baseline in app.routers.__init__,
    which must advance only at a deliberate load/unload).

    Today they mutate in place, so this pins the property rather than
    reporting a bug — a future refactor to "rebuild the record" is exactly
    what it is here to fail. The fixture is deck-authored with an OLD
    timestamp: both away from what a rebuild would produce.
    """
    store = IntentStore(tmp_path / "intent.json")
    old_ts = "2026-08-09T12:00:00+00:00"
    store.record("local/lemonade", state="loaded", model="m.gguf",
                 engine="lemonade", actor="deck", now=old_ts)

    store.note_failure("local/lemonade")
    record = store.get()["local/lemonade"]
    assert record["actor"] == "deck"
    assert record["updated_ts"] == old_ts

    store.note_healthy("local/lemonade")
    record = store.get()["local/lemonade"]
    assert record["actor"] == "deck"
    assert record["updated_ts"] == old_ts


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


def test_concurrent_record_and_note_healthy_lose_no_deliberate_write(tmp_path):
    """Regression for the 2026-08-06 lost-write: a router-thread record()
    racing watcher-thread note_healthy() calls must never be clobbered by a
    stale in-memory copy. Hammer both from two threads; after every joined
    round the deliberately-recorded state must be what record() last wrote."""
    import threading

    store = IntentStore(tmp_path / "intent.json")
    store.record("local/lemonade", state="loaded", model="m.gguf", engine="lemonade")
    store.record("local/hipfire", state="loaded", model=None, engine="hipfire")

    stop = threading.Event()

    def hammer_note_healthy():
        while not stop.is_set():
            store.note_healthy("local/hipfire")

    t = threading.Thread(target=hammer_note_healthy, daemon=True)
    t.start()
    try:
        for i in range(1000):
            state = "unloaded" if i % 2 else "loaded"
            store.record("local/lemonade", state=state, model=None, engine="lemonade")
            assert store.get()["local/lemonade"]["state"] == state
    finally:
        stop.set()
        t.join(timeout=5)


def test_put_back_restores_a_record_verbatim(tmp_path):
    """``put_back`` is the rollback primitive for a failed actuation: "as if
    it never happened". Unlike ``record()`` it re-stamps NOTHING — every
    field, including the ones record() deliberately resets
    (failures/quarantined) and the ones it re-derives (updated_ts, actor),
    comes back byte-identical.

    The fixture is quarantined with a non-zero failure count and an OLD
    operator-authored timestamp, because those are exactly the fields a
    record()-based rollback silently loses: it would clear a quarantine that
    the failure budget deliberately imposed, putting a crash-looping
    resource back into the restore rotation.
    """
    store = IntentStore(tmp_path / "intent.json")
    old_ts = "2026-08-09T12:00:00+00:00"
    store.record("local/lemonade", state="loaded", model="extra.foo.gguf",
                 engine="lemonade", actor="operator", now=old_ts)
    store.note_failure("local/lemonade")
    store.note_failure("local/lemonade")  # -> FAILURE_BUDGET -> quarantined
    prior = store.get()["local/lemonade"]
    assert prior["quarantined"] is True  # fixture precondition

    # The speculative pre-record an actuation arm writes before acting; it
    # resets failures/quarantined and re-stamps actor/updated_ts.
    store.record("local/lemonade", state="unloaded", model=None,
                 engine="lemonade", actor="deck")
    assert store.get()["local/lemonade"] != prior  # the write really happened

    store.put_back("local/lemonade", prior)

    assert store.get()["local/lemonade"] == prior


def test_put_back_leaves_other_keys_untouched(tmp_path):
    store = IntentStore(tmp_path / "intent.json")
    store.record("local/lemonade", state="loaded", model="m.gguf", engine="lemonade")
    store.record("local/hipfire", state="loaded", model=None, engine="hipfire")
    prior = store.get()["local/lemonade"]
    untouched = store.get()["local/hipfire"]

    store.record("local/lemonade", state="unloaded", model=None, engine="lemonade")
    store.put_back("local/lemonade", prior)

    assert store.get()["local/hipfire"] == untouched


def test_put_back_refuses_a_record_that_is_not_one(tmp_path):
    """Refuse, don't coerce: a caller handing back something that isn't a
    record would otherwise persist a shape every reader then has to defend
    against. state+engine are the fields derive_status/plan_reconcile
    dereference, so they are the minimum bar."""
    store = IntentStore(tmp_path / "intent.json")

    with pytest.raises(ValueError):
        store.put_back("local/lemonade", {"engine": "lemonade"})  # no state
    with pytest.raises(ValueError):
        store.put_back("local/lemonade", {"state": "loaded"})  # no engine
    with pytest.raises(ValueError):
        store.put_back("local/lemonade", {"state": "sideways", "engine": "lemonade"})
    with pytest.raises(ValueError):
        store.put_back("local/lemonade", "not a dict")


def test_load_drops_a_malformed_record_and_keeps_the_others(tmp_path):
    """Boundary gate [T7 review Important-1]. IntentStore was the one sibling
    store without a per-record gate: _load checked only the whole-FILE shape,
    so per-record garbage reached every consumer verbatim — and consumers
    hard-index (app/lifecycle.py and app/reconcile.py both do
    intent["model"]), so one bad record crashed the reconcile pass.

    Gated at _load, per this codebase's boundary rule and PolicyStore's
    precedent: malformed records are DROPPED, not repaired — there is no
    default to heal an intent to, and a dropped record reads as "no intent",
    which is the safe direction (nothing gets restored).
    """
    path = tmp_path / "intent.json"
    path.write_text(json.dumps({
        "local/good": {"state": "loaded", "model": "m.gguf", "engine": "lemonade",
                       "actor": "operator", "updated_ts": "2026-08-09T00:00:00+00:00",
                       "last_healthy_ts": None, "failures": 0, "quarantined": False},
        "local/not-a-dict": "wat",
        "local/bad-state": {"state": "sideways", "model": None, "engine": "lemonade"},
        "local/no-engine": {"state": "loaded", "model": None},
        "local/no-model-key": {"state": "loaded", "engine": "lemonade"},
        "local/bad-actor": {"state": "loaded", "model": None, "engine": "lemonade",
                            "actor": "gremlin"},
    }))
    store = IntentStore(path)

    data = store.get()

    assert set(data) == {"local/good"}


def test_load_keeps_a_legacy_record_with_no_actor(tmp_path):
    """A pre-upgrade intent.json has no `actor` at all. That is legal — every
    reader treats a missing actor as "operator" — so the gate must not drop
    it. Only a PRESENT-but-invalid actor is malformed."""
    path = tmp_path / "intent.json"
    path.write_text(json.dumps({
        "local/legacy": {"state": "loaded", "model": "m.gguf", "engine": "lemonade",
                         "updated_ts": "2026-08-09T00:00:00+00:00"},
    }))

    assert set(IntentStore(path).get()) == {"local/legacy"}


def test_a_model_key_of_none_is_well_formed(tmp_path):
    """model=None is legitimate — hipfire is single-model, so "loaded, no
    opinion which" is a real intent (app/lifecycle.py's `wanted is None`
    branch). The gate checks PRESENCE, never truthiness."""
    path = tmp_path / "intent.json"
    path.write_text(json.dumps({
        "local/hipfire": {"state": "loaded", "model": None, "engine": "hipfire"},
    }))

    assert set(IntentStore(path).get()) == {"local/hipfire"}


def test_put_back_refuses_a_record_with_no_model_key(tmp_path):
    """[T7 review Important-2] state+engine was NOT the right bar: consumers
    hard-index model too (app/lifecycle.py, app/reconcile.py), so a
    record accepted without it persists and KeyErrors the next reconcile
    pass."""
    store = IntentStore(tmp_path / "intent.json")

    with pytest.raises(ValueError):
        store.put_back("local/lemonade", {"state": "loaded", "engine": "lemonade"})


def test_put_back_refuses_an_unknown_actor(tmp_path):
    """record() validates actor; put_back must hold the same contract, or a
    put-back "gremlin" reads as non-operator at app/routers/control.py's
    supersession check — the direction record()'s own comment calls wrong."""
    store = IntentStore(tmp_path / "intent.json")

    with pytest.raises(ValueError):
        store.put_back("local/lemonade", {"state": "loaded", "model": None,
                                          "engine": "lemonade", "actor": "gremlin"})
