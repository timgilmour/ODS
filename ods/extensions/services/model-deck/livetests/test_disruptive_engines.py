"""D12: a declared REMOTE engine's lifecycle round trip, against the live box.

    ./deck-drill --disruptive-only -k engines

up -> healthy -> DELIBERATE UNLOAD -> ~30 reconcile ticks with no restore ->
load -> healthy -> the box left exactly as it was found.

This is D9 (`test_disruptive_lifecycle.py`, "a deliberate park stays parked")
carried to the fourth kind, and it is the more dangerous half of the same
invariant. Locally, a reconciler that fights a park restarts a container on
this box. Remotely it sends a compose `up` to ANOTHER machine — one holding
~62 GiB of weights that take ~3.5-4.5 min to reload (GF4, measured at the
2026-08-16 gate) — so a deck that cannot leave a deliberate unload alone
would spend minutes of somebody else's GPU undoing an operator's decision,
possibly while that box is deliberately being freed for something else.

Why it is disruptive: it takes the engine down and brings it back, which
costs the boot every time. Nothing else on the box is touched.

**Preconditions skip, never fail** (`test_spark_ds4.py`'s `ds4_window`
discipline, and its reason): the live deployment declares no remote engine
until an operator does — declaration is a post-deploy step. It also never
runs from a state it did not find: a busy engine (a render in flight), an
unreachable node, or a node that cannot be asked for verbs at all each skip
with a reason rather than forcing anything.

**Teardown restores what was there BEFORE the case** (the D11 lesson,
`75ff602a`: a drill that restores a hardcoded known-good state moves boxes it
found somewhere else). Loaded before means loaded after; down before means
down after. What teardown deliberately does NOT undo is the intent RECORD —
there is no route that un-records intent short of forgetting the declaration,
and leaving a truthful record of the drill's own last action is the same
posture `restore_hipfire` and `restore_ds4` take. An engine found `unmanaged`
(no intent at all — somebody launched it by hand) therefore ends this case
managed, which is a change this module states out loud rather than hides.

Runtime: expect 6-12 minutes, and budget for more. Each load leg is a cold
boot of the engine; the park watch is a flat minute.
"""

import time

import pytest

from test_safe_engines import declared_remote_engines, engine_key

pytestmark = pytest.mark.disruptive

# GF4: ~3.5-4.5 min cold, container start -> healthy. 600 s is not a padded
# guess — it is `_SGLANG_OMNI_WARM_WINDOW_S` (app/engine_kinds.py), the window
# inside which the deck reads a `down` observation as a boot in flight rather
# than a death. A boot that outlasts it is a real defect (the reconciler
# starts restore-storming it toward quarantine), so this case fails at exactly
# that line rather than waiting longer and calling it fine.
BOOT_TIMEOUT = 600.0
# Bringing a container down is not a boot: seconds, plus one observation TTL.
UNLOAD_TIMEOUT = 180.0
# ~30 watcher ticks at the 2 s default interval — long enough that a
# reconciler with any opinion about a parked engine would have acted many
# times over. Same budget, same reasoning, as D9's PARK_WATCH_S.
#
# IT MUST ALSO EXCEED THE OBSERVATION TTL (app.node_clients'
# REMOTE_OBSERVE_TTL_S, 10 s), and that is not a detail: a remote engine's
# observation is cached, so a watch shorter than the TTL would re-read the
# same pre-watch answer and "nothing came back up" would be the CACHE
# talking. Found by driving this case against a fake engine that revived
# itself mid-watch — at 3 s it passed anyway; at 12 s it failed as it should.
PARK_WATCH_S = 60.0
# The deck derives from a TTL-cached observation (10 s), so a status trails
# the wire by a few seconds.
DERIVE_TIMEOUT = 60.0


def _lifecycle(deck, key: str) -> dict:
    return deck.get("/api/state").json()["lifecycle"][key]


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(2)
    pytest.fail(f"{what} — not met within {timeout_s:.0f}s")


def _restores_for(deck, key: str) -> list[dict]:
    events = deck.get("/api/events", params={"n": 500}).json()["events"]
    return [e for e in events
            if e["kind"] in ("lifecycle-restore", "lifecycle-restore-failed")
            and e["detail"].get("key") == key]


def _verb(deck, node_id: str, resource: str, verb: str) -> None:
    """One engine verb, asserted accepted. 202 (not 200) is the route's own
    answer: the node-agent queues the request for its host-side helper and
    nothing in the deck observes the result, so claiming an outcome here
    would be a lie (app/routers/serving.py's engine_verb)."""
    r = deck.post(f"/api/nodes/{node_id}/engines/{resource}/{verb}", timeout=30.0)
    assert r.status_code == 202, f"{verb} refused: {r.status_code} {r.text}"


def _is_up(deck, key: str) -> bool:
    """Whether the engine is observed RESIDENT — the kind's own `active`
    reading, off the wire: it holds its weights between renders, so both a
    busy and an idle observation are "up" (app/engine_kinds.py's
    _SglangOmniAdapter.active). Absent/None means we failed to look, which is
    not "down"."""
    observed = _lifecycle(deck, key)["observed"]
    return bool(observed["reachable"]) and bool(observed["loaded"])


@pytest.fixture
def engine_window(deck):
    """Preconditions for the round trip; skips rather than forces any of them.

    Returns ``(node_id, resource, key, was_up)`` — the last of which is what
    teardown restores the box to.
    """
    declared = declared_remote_engines(deck)
    if not declared:
        pytest.skip("no engine is declared on any node-agent entry — declaring "
                    "one (POST /api/nodes/{node}/engines) is a post-deploy "
                    "operator step, not part of the seed")

    kinds = {k["kind"]: k for k in deck.get("/api/engine-kinds").json()["kinds"]}
    nodes = {n["id"]: n for n in deck.get("/api/nodes").json()["nodes"]}

    node_id, engine = declared[0]
    resource = engine["resource"]
    key = engine_key(node_id, engine)
    node = nodes[node_id]

    # The three things `client_for` needs to answer with a client at all
    # (app/node_clients.py) — checked here so the case skips with a reason
    # instead of collecting a 503 from the verb route.
    if not node.get("address") or not node["credential_set"]:
        pytest.skip(f"node {node_id!r} has no address or no stored credential — "
                    "the deck cannot ask it for anything")
    if not kinds[engine["kind"]]["remote_capable"]:
        pytest.skip(f"{resource} is declared as {engine['kind']!r}, which has no "
                    "remote constructor — nothing to actuate")

    entry = _lifecycle(deck, key)
    if not entry["observed"]["reachable"]:
        pytest.skip(f"{key} reads {entry['status']!r} — the node is not "
                    "answering, and a drill must not wake a box it found dark")

    was_up = _is_up(deck, key)
    if was_up:
        # The activity check only means anything while the engine is
        # RESIDENT: a down engine has no render to protect. It reads the
        # WORLD tenant, not the lifecycle `observed` record — the latter is
        # app.observe's four-field shape (reachable/loaded/model/
        # transitioning), deliberately kind-blind, and an activity count is
        # this kind's own vocabulary (app/state.py's snapshot_remote carries
        # the adapter's observation verbatim).
        tenant = deck.get("/api/state").json()["world"]["remote_tenants"][key]
        if "busy_requests" not in tenant:
            pytest.skip(f"{key} ({engine['kind']}) reports no activity signal "
                        "this drill knows how to read — refusing to guess "
                        "whether it is safe to take down")
        busy = tenant["busy_requests"]
        if busy is None:
            pytest.skip(f"{key}'s busy indicator is unavailable — the deck "
                        "fails toward alive on that (design §4), and so does "
                        "this drill")
        if busy:
            pytest.skip(f"busy-skip: {key} has {busy} request(s) in flight — a "
                        "song render runs for minutes and this case would kill it")

    return node_id, resource, key, was_up


@pytest.fixture
def restore_engine(deck, engine_window):
    """UNCONDITIONAL teardown: the engine ends the case in the state the case
    found it in, up or down.

    Runs on pass, fail and error alike. It depends on `engine_window`, so a
    skipped precondition never arms it — a skip cannot move the box.

    Restoring through the deck's own verb route rather than sparky's helper
    directly is deliberate, the same reason `restore_hipfire` resumes through
    the deck: that is the action that records intent, so the reconciler is
    left with a record that matches the box instead of one naming a state
    nobody is in.
    """
    node_id, resource, key, was_up = engine_window
    yield

    if _is_up(deck, key) == was_up:
        return
    _verb(deck, node_id, resource, "load" if was_up else "unload")
    _wait_for(lambda: _is_up(deck, key) == was_up,
              BOOT_TIMEOUT if was_up else UNLOAD_TIMEOUT,
              f"TEARDOWN FAILED: {key} is not back to "
              f"{'loaded' if was_up else 'unloaded'}")


def test_d12_remote_engine_round_trip(deck, engine_window, restore_engine,
                                      lemonade_guard):
    """up -> healthy -> deliberate unload -> ~30 ticks parked and untouched ->
    load -> healthy.

    `lemonade_guard`: the two boot legs together can outlast lemonade's 900 s
    idle TTL, so the deck's idle-release may legitimately unload it mid-drill
    and trip `box_bookend`'s left-as-found check — found live on D5
    (2026-07-31) and inherited by every long case since.
    """
    node_id, resource, key, _was_up = engine_window

    # --- leg 1: up, and the deck says so -----------------------------------
    if not _is_up(deck, key):
        _verb(deck, node_id, resource, "load")
        _wait_for(lambda: _is_up(deck, key), BOOT_TIMEOUT,
                  f"{key} never came up (GF4 budget is ~4.5 min; past "
                  f"{BOOT_TIMEOUT:.0f}s the deck reads a boot as a death)")
    else:
        # Already resident, but possibly with no intent recorded (an engine
        # somebody launched by hand reads 'unmanaged'). The load is what makes
        # the unload below a DELIBERATE one rather than a park of something
        # the deck never claimed — and it is idempotent on a running engine
        # (`up` is a compose up).
        _verb(deck, node_id, resource, "load")
    _wait_for(lambda: _lifecycle(deck, key)["status"] == "serving", DERIVE_TIMEOUT,
              f"{key} is up but the deck does not read it as 'serving'")

    restores_before = len(_restores_for(deck, key))

    # --- leg 2: the deliberate unload, and ~30 ticks of being left alone ----
    _verb(deck, node_id, resource, "unload")
    _wait_for(lambda: not _is_up(deck, key), UNLOAD_TIMEOUT,
              f"{key} never went down after a deliberate unload")
    _wait_for(lambda: _lifecycle(deck, key)["status"] == "parked", DERIVE_TIMEOUT,
              f"{key} is down but the deck does not read it as 'parked' — "
              "reading a deliberate unload as 'down' is the 26-hour outage's "
              "mirror image, and restores follow it")

    time.sleep(PARK_WATCH_S)

    entry = _lifecycle(deck, key)
    assert entry["status"] == "parked", entry
    assert entry["intent"]["state"] == "unloaded", entry["intent"]
    assert not _is_up(deck, key), "something brought a parked remote engine back up"
    assert len(_restores_for(deck, key)) == restores_before, \
        "the reconciler fought a deliberate unload on another box"

    # --- leg 3: and it comes back by the ordinary route --------------------
    # This leg is also the park watch's vacuity guard: it proves the verb path
    # and the node were live throughout, so "nothing restored it" above was
    # the D9 invariant holding, not a channel that could not have acted anyway.
    _verb(deck, node_id, resource, "load")
    _wait_for(lambda: _is_up(deck, key), BOOT_TIMEOUT,
              f"{key} did not come back up after the park")
    _wait_for(lambda: _lifecycle(deck, key)["status"] == "serving", DERIVE_TIMEOUT,
              f"{key} is up again but the deck does not read it as 'serving'")
