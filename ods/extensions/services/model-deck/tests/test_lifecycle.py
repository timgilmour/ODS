"""Tests for app.lifecycle.derive_status — intent x observed -> status.

This is the function that fixes the 2026-08-03 outage class: 'deliberately
parked' and 'dead' produce the same observation, and only intent tells them
apart. Every row of the design doc's table is covered here.
"""

import pytest

from app.lifecycle import STATUSES, derive_status


def _intent(state="loaded", model="m", quarantined=False):
    return {
        "state": state,
        "model": model,
        "engine": "hipfire",
        "updated_ts": "2026-08-04T00:00:00+00:00",
        "last_healthy_ts": None,
        "failures": 0,
        "quarantined": quarantined,
    }


def _observed(reachable=True, loaded=True, model="m", transitioning=False):
    return {"reachable": reachable, "loaded": loaded, "model": model,
            "transitioning": transitioning}


def test_loaded_intent_matching_observation_is_serving():
    result = derive_status(_intent(), _observed())

    assert result["status"] == "serving"


def test_loaded_intent_wrong_model_is_drifted():
    result = derive_status(_intent(model="wanted"), _observed(model="actual"))

    assert result["status"] == "drifted"
    assert "wanted" in result["reason"] and "actual" in result["reason"]


def test_loaded_intent_with_no_named_model_is_serving_whatever_is_up():
    """Single-model engines (hipfire) record model=None because the Deck
    cannot choose their model. That must mean 'anything counts', not
    'permanently drifted' — the naive equality check gets this wrong."""
    result = derive_status(_intent(model=None), _observed(model="gpt-oss-120b"))

    assert result["status"] == "serving"


def test_loaded_intent_not_loaded_is_down():
    """The hipfire-dead-for-26-hours case."""
    result = derive_status(_intent(), _observed(loaded=False, model=None))

    assert result["status"] == "down"


def test_loaded_intent_unreachable_is_unreachable():
    result = derive_status(_intent(), _observed(reachable=False, loaded=False, model=None))

    assert result["status"] == "unreachable"


def test_unloaded_intent_not_loaded_is_parked():
    """The deliberate park. MUST NOT be 'down'."""
    result = derive_status(_intent(state="unloaded", model=None),
                           _observed(loaded=False, model=None))

    assert result["status"] == "parked"


def test_unloaded_intent_but_loaded_is_unexpected():
    result = derive_status(_intent(state="unloaded", model=None), _observed(model="surprise"))

    assert result["status"] == "unexpected"


def test_no_intent_loaded_is_unmanaged():
    result = derive_status(None, _observed(model="mystery"))

    assert result["status"] == "unmanaged"


def test_no_intent_not_loaded_is_idle():
    result = derive_status(None, _observed(loaded=False, model=None))

    assert result["status"] == "idle"


def test_no_intent_unreachable_is_unreachable():
    result = derive_status(None, _observed(reachable=False, loaded=False, model=None))

    assert result["status"] == "unreachable"


def test_transitioning_is_warming_not_down():
    """Mid-boot must never read 'down' — that is what turns one slow start
    into a restart storm."""
    result = derive_status(_intent(), _observed(loaded=False, model=None, transitioning=True))

    assert result["status"] == "warming"


def test_transitioning_outranks_quarantine():
    """Something actively coming up is news; a stale quarantine label is not."""
    result = derive_status(_intent(quarantined=True),
                           _observed(loaded=False, model=None, transitioning=True))

    assert result["status"] == "warming"


def test_quarantine_wins_over_everything():
    result = derive_status(_intent(quarantined=True), _observed(loaded=False, model=None))

    assert result["status"] == "quarantined"


def test_quarantine_does_not_mask_a_healthy_recovery():
    """If it came back on its own, say so — quarantine is about restore
    attempts, and an operator staring at 'quarantined' while the thing is
    actually serving would be a lie."""
    result = derive_status(_intent(quarantined=True), _observed())

    assert result["status"] == "serving"


def test_every_returned_status_is_in_the_vocabulary():
    cases = [
        (_intent(), _observed()),
        (_intent(model="a"), _observed(model="b")),
        (_intent(), _observed(loaded=False, model=None)),
        (_intent(), _observed(reachable=False, loaded=False, model=None)),
        (_intent(state="unloaded", model=None), _observed(loaded=False, model=None)),
        (_intent(state="unloaded", model=None), _observed()),
        (None, _observed()),
        (None, _observed(loaded=False, model=None)),
        (_intent(quarantined=True), _observed(loaded=False, model=None)),
    ]

    for intent, observed in cases:
        assert derive_status(intent, observed)["status"] in STATUSES


def test_reason_is_always_a_nonempty_string():
    assert derive_status(None, _observed())["reason"]
    assert derive_status(_intent(), _observed())["reason"]
