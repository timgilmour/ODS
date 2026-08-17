"""HoldStore — announced absences the reconciler must not restore."""

import pytest

from app.holds import DEFAULT_HOLD_TTL_S, MAX_HOLD_TTL_S, HoldStore


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_unheld_key_is_not_held():
    store = HoldStore(clock=FakeClock())
    assert store.held("local/hipfire") is False


def test_hold_then_held():
    store = HoldStore(clock=FakeClock())
    store.hold("local/hipfire", 60.0)
    assert store.held("local/hipfire") is True


def test_hold_expires_on_its_own():
    clock = FakeClock()
    store = HoldStore(clock=clock)
    store.hold("local/hipfire", 60.0)
    clock.now += 59.0
    assert store.held("local/hipfire") is True
    clock.now += 2.0
    assert store.held("local/hipfire") is False


def test_expiry_is_evaluated_at_read_not_write():
    """No sweeper thread: a hold nobody asks about still expires."""
    clock = FakeClock()
    store = HoldStore(clock=clock)
    store.hold("local/hipfire", 10.0)
    clock.now += 100.0
    assert store.held("local/hipfire") is False
    # And the expired entry is not retained.
    assert store.release("local/hipfire") is False


def test_release_reports_whether_it_removed_anything():
    store = HoldStore(clock=FakeClock())
    assert store.release("local/hipfire") is False
    store.hold("local/hipfire", 60.0)
    assert store.release("local/hipfire") is True
    assert store.held("local/hipfire") is False


def test_holds_are_per_key():
    store = HoldStore(clock=FakeClock())
    store.hold("local/hipfire", 60.0)
    assert store.held("local/lemonade") is False


def test_re_holding_extends_the_deadline():
    clock = FakeClock()
    store = HoldStore(clock=clock)
    store.hold("local/hipfire", 60.0)
    clock.now += 50.0
    store.hold("local/hipfire", 60.0)
    clock.now += 20.0
    assert store.held("local/hipfire") is True


@pytest.mark.parametrize("bad", [0, -1, MAX_HOLD_TTL_S + 1])
def test_refuses_a_ttl_outside_the_band(bad):
    """Refuse, never coerce: a clamped TTL is a guess about intent."""
    store = HoldStore(clock=FakeClock())
    with pytest.raises(ValueError):
        store.hold("local/hipfire", bad)
    assert store.held("local/hipfire") is False


def test_default_ttl_exceeds_a_cold_load():
    """hipfire's manifest budgets health_timeout: 300 for a cold MQ4 load."""
    assert DEFAULT_HOLD_TTL_S > 300.0
    assert DEFAULT_HOLD_TTL_S <= MAX_HOLD_TTL_S
