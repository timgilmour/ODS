"""The one actuation lock.

Three things actuate real engine state from three different threads: the
watcher's tick (arbitration + reconcile restores), a set apply (HTTP
thread), and the pull-through completion hook (mover thread, minutes after
its request). Each previously coordinated differently — the tick with a
start-of-tick peek [max-review #7], the hook with nothing [#6]. One lock,
owned by whoever is actuating, replaces all of it:

* apply() HOLDS it for the whole step sequence (blocking acquire — an
  operator's apply waits out an in-flight tick's restores, up to a
  lemonade load's 180 s worst case, rather than interleaving with them);
* the tick's actuation+reconcile phase TRY-acquires — skipping a tick
  while someone else actuates is exactly the old yield semantics, minus
  the race after the peek;
* the pull-through hook HOLDS it around its restart+load.

Observation, derive and provenance passes never touch engine state and
never take this lock.
"""

import threading

LOCK = threading.Lock()


def in_progress() -> bool:
    """True while any actuator holds the lock — a peek, not an acquire."""
    return LOCK.locked()
