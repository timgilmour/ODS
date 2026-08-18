"""Announced absences — keys the reconciler must not restore right now.

An engine can go away for two very different reasons: it died, or somebody
deliberately took it down. The reconciler cannot tell those apart from an
observation alone, so it treats every absence as a death and restores it.
That is correct for a crash and wrong for an operator's own actuation — see
``bin/ods-host-agent.py``'s dashboard activate path, which recreates the
hipfire container out from under the deck.

A hold is an actuator saying "this absence is mine, expect it". It suppresses
restore for one key and nothing else: it does not park the resource, does not
change intent, and does not touch the failure budget.

DELIBERATELY IN-MEMORY (spec D1). It matches ``_restore_unverified``
(app/arbiter.py) — same lifetime, same reasoning — and it fails open BY
CONSTRUCTION rather than by remembering to expire: a deck restart drops every
hold and the world is reconciled from observation again, which is the
direction we want to fail. The cost is one spurious restore if the deck
restarts inside an actuation window; the alternative is a hold surviving into
a world that no longer justifies it, which is the swap.sh stranding failure
(homelab 5db3275) rebuilt with extra steps.

Every hold is TTL-bounded for the same reason. A crashed actuator must never
be able to silence the reconciler indefinitely.
"""

from __future__ import annotations

import time
from typing import Callable

# A cold MQ4 load on hipfire takes minutes; its manifest budgets
# health_timeout: 300. The default sits above that so a normal activation
# never races its own hold expiring.
DEFAULT_HOLD_TTL_S: float = 360.0
# Ceiling. Past this, an actuator is not announcing an absence, it is
# disabling the reconciler — which is what POST /api/lifecycle/auto is for.
MAX_HOLD_TTL_S: float = 900.0


class HoldStore:
    """Per-key announced absences with monotonic deadlines."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._until: dict[str, float] = {}

    def hold(self, key: str, ttl_s: float = DEFAULT_HOLD_TTL_S) -> float:
        """Suppress restore for `key` for `ttl_s` seconds. Returns the deadline.

        Re-holding extends: an actuator that announces twice (retry, or a
        rollback recreate after a failed activation) means the window is
        still open, not that it should have closed at the first deadline.
        """
        if not 0 < ttl_s <= MAX_HOLD_TTL_S:
            raise ValueError(
                f"ttl_s must be in (0, {MAX_HOLD_TTL_S}]; got {ttl_s!r}")
        until = self._clock() + ttl_s
        self._until[key] = until
        return until

    def held(self, key: str) -> bool:
        """Whether `key`'s absence is currently announced.

        Expiry is evaluated HERE, at read time, and the expired entry is
        dropped — there is no sweeper thread to fail to start.
        """
        until = self._until.get(key)
        if until is None:
            return False
        if self._clock() >= until:
            # pop, not del (Task 2 fix round, ordered ahead of the next task
            # wiring the HTTP router path onto this same object): a
            # concurrent hold()/release() from a request thread could drop
            # this key between the get above and the del, which would raise
            # KeyError. pop(key, None) tolerates it already being gone —
            # same outcome, no race.
            self._until.pop(key, None)
            return False
        return True

    def release(self, key: str) -> bool:
        """Drop `key`'s hold. True if one was actually removed."""
        return self._until.pop(key, None) is not None
