"""Tests for app.reconcile.plan_reconcile — which statuses cause a restore.

Restore happens for exactly one status ('down') under exactly the right
conditions. Everything else is deliberately inert; each such case below
corresponds to a real way an eager reconciler would have made 2026-08-03
worse instead of better.
"""

from app.reconcile import plan_reconcile


def _statuses(**kwargs):
    return {key: {"status": status, "reason": "r"} for key, status in kwargs.items()}


def _intents(**kwargs):
    return {
        key: {"state": state, "model": "m", "engine": "hipfire",
              "updated_ts": "t", "last_healthy_ts": None,
              "failures": 0, "quarantined": False}
        for key, state in kwargs.items()
    }


def test_down_produces_a_restore():
    actions = plan_reconcile(
        _statuses(**{"local/hipfire": "down"}),
        _intents(**{"local/hipfire": "loaded"}),
        auto_enabled=True,
    )

    assert actions == [{
        "action": "restore", "key": "local/hipfire",
        "engine": "hipfire", "model": "m",
        "reason": "intent is loaded but nothing is loaded",
    }]


def test_parked_is_never_restored():
    """THE invariant. A deliberate unload must survive every tick."""
    actions = plan_reconcile(
        _statuses(**{"local/lemonade": "parked"}),
        _intents(**{"local/lemonade": "unloaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_serving_produces_nothing():
    actions = plan_reconcile(
        _statuses(**{"local/hipfire": "serving"}),
        _intents(**{"local/hipfire": "loaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_drifted_is_reported_not_actuated():
    actions = plan_reconcile(
        _statuses(**{"sparky/slot0": "drifted"}),
        _intents(**{"sparky/slot0": "loaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_unexpected_is_never_auto_unloaded():
    actions = plan_reconcile(
        _statuses(**{"local/lemonade": "unexpected"}),
        _intents(**{"local/lemonade": "unloaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_unmanaged_is_never_actuated():
    actions = plan_reconcile(
        _statuses(**{"local/comfyui": "unmanaged"}), {},
        auto_enabled=True,
    )

    assert actions == []


def test_unreachable_produces_nothing():
    """A node that is off is not a model that fell over."""
    actions = plan_reconcile(
        _statuses(**{"sparky/slot0": "unreachable"}),
        _intents(**{"sparky/slot0": "loaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_quarantined_is_never_retried():
    actions = plan_reconcile(
        _statuses(**{"local/hipfire": "quarantined"}),
        _intents(**{"local/hipfire": "loaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_auto_disabled_suppresses_everything():
    actions = plan_reconcile(
        _statuses(**{"local/hipfire": "down"}),
        _intents(**{"local/hipfire": "loaded"}),
        auto_enabled=False,
    )

    assert actions == []


def test_warming_defers_restores():
    """Right after a node boots, 'not loaded yet' is indistinguishable from
    'died'. Waiting costs seconds; guessing wrong costs a swap.

    This used to be a GLOBAL ``boot_window_active`` flag, which meant one
    spark swap suppressed restore of local hipfire and lemonade for up to 20
    minutes. The suppression now rides on the observation of the booting
    resource itself: observe_spark marks it ``transitioning``, derive_status
    turns that into ``warming``, and warming is simply not actionable — so
    only the resource that is actually booting is protected.
    """
    actions = plan_reconcile(
        _statuses(**{"sparky/slot0": "warming"}),
        _intents(**{"sparky/slot0": "loaded"}),
        auto_enabled=True,
    )

    assert actions == []


def test_a_booting_spark_does_not_suppress_a_local_restore():
    """The defect the global boot window had: one spark swap (which can take
    ~15 min) blocked restore of local hipfire and lemonade for the whole
    window. Suppression is per-resource now."""
    actions = plan_reconcile(
        _statuses(**{"sparky/slot0": "warming", "local/hipfire": "down"}),
        _intents(**{"sparky/slot0": "loaded", "local/hipfire": "loaded"}),
        auto_enabled=True,
    )

    assert [a["key"] for a in actions] == ["local/hipfire"]


def test_missing_intent_for_a_down_key_is_skipped_not_crashed():
    actions = plan_reconcile(
        _statuses(**{"ghost/slot0": "down"}), {},
        auto_enabled=True,
    )

    assert actions == []


def test_multiple_downs_all_planned_in_key_order():
    actions = plan_reconcile(
        _statuses(**{"local/hipfire": "down", "local/lemonade": "down"}),
        _intents(**{"local/hipfire": "loaded", "local/lemonade": "loaded"}),
        auto_enabled=True,
    )

    assert [a["key"] for a in actions] == ["local/hipfire", "local/lemonade"]
