"""The one actuation lock.

Three things actuate real engine state from three different threads: the
watcher's tick (arbitration + reconcile restores), a set apply (HTTP
thread), and the pull-through completion hook (mover thread, minutes after
its request). Each previously coordinated differently — the tick with a
start-of-tick peek [max-review #7], the hook with nothing [#6]. One lock,
owned by whoever is actuating, replaces all of it:

* apply() HOLDS it for the whole step sequence (blocking acquire — an
  operator's apply waits out whatever else is actuating first, rather than
  interleaving with it). Worst case is NOT a lemonade load's 180 s
  (app/engines/lemonade.py) — a plan that activates a new default route
  blocks on the host agent's own 600 s read timeout
  (app/engines/hostagent.py:36) before this apply's OWN next step, let
  alone before a caller waiting on this lock, ever runs;
* the tick's actuation+reconcile phase TRY-acquires — skipping a tick
  while someone else actuates is exactly the old yield semantics, minus
  the race after the peek;
* the pull-through hook HOLDS it around its restart+load — worst case
  ~285 s: a lemonade container restart's stop/retry/start (~45 s,
  app/notify.py) + the readiness poll (60 s, `_READY_TIMEOUT_S` in
  app/routers/control.py) + a 180 s lemonade load
  (app/engines/lemonade.py).

Observation, derive and provenance passes never touch engine state and
never take this lock.
"""

import threading

LOCK = threading.Lock()


def in_progress() -> bool:
    """True while any actuator holds the lock — a peek, not an acquire."""
    return LOCK.locked()
