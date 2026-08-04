"""D7-D9: lifecycle reconciliation against the live box.

D9 is the most important regression in the suite: a deliberate unload must
NOT be reloaded by the reconciler. If D9 ever fails, the Deck has started
fighting its operator, and no amount of restore correctness compensates.

D7 is its mirror image and the incident these drills exist for: ods-hipfire
sat Exited(0) for 26 hours on 2026-08-03 looking exactly like a park. The
out-of-band stop here is performed through the deck's OWN socket proxy —
the deck must not be told, because "somebody stopped it behind our back" is
precisely the case under test.

Every case that stops hipfire takes `restore_hipfire`, whose teardown runs
unconditionally (conftest.py). D9 needs it as much as D7 does: its whole
point is that hipfire is left parked for a minute, and pytest cannot be
relied upon to reach the end of the test body.
"""

import time

import pytest

from conftest import HIPFIRE_CONTAINER, wait_hipfire_state

pytestmark = pytest.mark.disruptive

KEY = "local/hipfire"

# A restore is planned within one 2 s tick, but proving it means waiting for
# hipfire to finish loading its 33 GB of weights (~17 s warm, minutes cold).
RESTORE_TIMEOUT = 600.0
# ~30 watcher ticks at the 2 s default interval: long enough that a
# reconciler that fights parks would have done so many times over.
PARK_WATCH_S = 60.0


def _lifecycle(deck, key: str = KEY) -> dict:
    return deck.get("/api/state").json()["lifecycle"][key]


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(2)
    pytest.fail(f"{what} — not met within {timeout_s:.0f}s")


def _restores_for(deck, key: str = KEY) -> list[dict]:
    events = deck.get("/api/events", params={"n": 500}).json()["events"]
    return [e for e in events
            if e["kind"] == "lifecycle-restore" and e["detail"].get("key") == key]


@pytest.fixture
def hipfire_managed(deck, litellm_direct):
    """Preconditions for a hipfire lifecycle drill, plus intent 'loaded'.

    The resume is the deliberate action that RECORDS the intent these cases
    reconcile against; without it hipfire reads 'unmanaged' and nothing in
    the reconciler is reachable. It is idempotent on a running container
    (the proxy answers 304).
    """
    state = deck.get("/api/state").json()["world"]["tenants"]["hipfire"]["state"]
    if state != "running":
        pytest.skip(f"hipfire is {state} — the lifecycle window expects it serving")
    if litellm_direct.default_targets_hipfire():
        pytest.skip("litellm's default route targets hipfire — park is guard-refused, "
                    "and stopping it would break every default caller")

    deck.post("/api/tenants/hipfire/resume", params={"force": "true"},
              timeout=60.0).raise_for_status()
    _wait_for(lambda: _lifecycle(deck)["status"] == "serving", 120,
              "intent recorded but hipfire never read 'serving'")
    return state


def test_d7_reconciler_restores_an_out_of_band_stop(deck, docker, hipfire_managed,
                                                    restore_hipfire):
    """Intent says loaded; stop the container behind the Deck's back; the
    reconciler notices 'down' and brings it back, unaided."""
    before = len(_restores_for(deck))

    docker.stop(HIPFIRE_CONTAINER)
    assert not docker.running(HIPFIRE_CONTAINER)
    # Ground truth first: the Deck must SEE the loss, and must call it 'down'
    # (intent loaded + nothing loaded), never 'parked'. Reading it as a park
    # is the 26-hour outage.
    _wait_for(lambda: _lifecycle(deck)["status"] in ("down", "warming", "serving"),
              60, "Deck never noticed the out-of-band stop")

    _wait_for(lambda: _lifecycle(deck)["status"] == "serving", RESTORE_TIMEOUT,
              "hipfire was not restored to 'serving'")
    assert docker.running(HIPFIRE_CONTAINER)

    after = _restores_for(deck)
    assert len(after) > before, "no lifecycle-restore event for the restored key"

    # The operator escapes, on a key that now really exists (Task 9 routes).
    cleared = deck.post(f"/api/lifecycle/quarantine/{KEY}/clear")
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["quarantined"] is False
    missing = deck.post("/api/lifecycle/quarantine/local/__no_such_key__/clear")
    assert missing.status_code == 404, missing.text


@pytest.mark.skip(reason=(
    "no live-safe failure injection on this box. A restore only counts a "
    "failure when it RAISES: DockerCtl.start() returns 204 for a container "
    "that starts and immediately exits, a renamed or removed container makes "
    "status() raise and therefore derives 'unreachable' (never the actionable "
    "'down'), and the one remaining lever — stopping ods-docker-ctl — is "
    "outside the drill's blast radius and would disable the very restore path "
    "under test. sparky, whose swap failures are the other natural source, is "
    "powered off. The budget itself is covered by unit rows over "
    "IntentStore.note_failure and Watcher._execute_restore; what can be "
    "asserted live is the release route, which test_d7 does."))
def test_d8_quarantine_after_two_failed_restores(deck, docker, restore_hipfire):
    """A resource whose restore keeps raising is tried exactly twice, then
    left alone. Enable by pointing the drill at a box where hipfire's
    restore can be made to fail (see the skip reason)."""
    before = len(_restores_for(deck))

    docker.stop(HIPFIRE_CONTAINER)
    _wait_for(lambda: _lifecycle(deck)["status"] == "quarantined", 300,
              "key never quarantined")

    events = deck.get("/api/events", params={"n": 500}).json()["events"]
    failures = [e for e in events if e["kind"] == "lifecycle-restore-failed"
                and e["detail"].get("key") == KEY]
    assert len(failures) == 2, failures
    assert [e for e in events if e["kind"] == "lifecycle-quarantined"
            and e["detail"].get("key") == KEY]

    time.sleep(30)
    events = deck.get("/api/events", params={"n": 500}).json()["events"]
    after = [e for e in events if e["kind"] == "lifecycle-restore-failed"
             and e["detail"].get("key") == KEY]
    assert len(after) == len(failures), "quarantined key was retried"
    assert len(_restores_for(deck)) == before


def test_d9_a_deliberate_park_stays_parked(deck, docker, hipfire_managed,
                                           restore_hipfire):
    """THE regression. Park hipfire deliberately; it must stay parked across
    ~30 reconcile ticks, and must read 'parked', never 'down'."""
    before = len(_restores_for(deck))

    # force=true skips the conversation-activity guard only; the litellm
    # route guard is never skippable and hipfire_managed already cleared it.
    deck.post("/api/tenants/hipfire/park", params={"force": "true"},
              timeout=60.0).raise_for_status()
    _wait_for(lambda: _lifecycle(deck)["status"] == "parked", 120,
              "park did not derive 'parked'")
    assert not docker.running(HIPFIRE_CONTAINER)

    time.sleep(PARK_WATCH_S)

    status = _lifecycle(deck)
    assert status["status"] == "parked", status
    assert status["intent"]["state"] == "unloaded", status["intent"]
    assert not docker.running(HIPFIRE_CONTAINER), "something restarted a parked engine"
    assert len(_restores_for(deck)) == before, "the reconciler fought a deliberate park"

    # And a park is recoverable by the ordinary route, not just by teardown.
    deck.post("/api/tenants/hipfire/resume", params={"force": "true"},
              timeout=60.0).raise_for_status()
    assert wait_hipfire_state(deck, "running", RESTORE_TIMEOUT)
