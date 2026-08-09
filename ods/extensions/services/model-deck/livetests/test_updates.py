"""Safe-tier: proves the shape of a real check, never a version number.

Asserting "the newest ds4 tag is v0.5.6" would rot the moment upstream ships.
What must hold is that a real pass produces a real verdict drawn from the
declared vocabulary (app.updates' CURRENT/AVAILABLE/UNDETERMINED/UNAVAILABLE
-- the last two are read-it-but-cannot-rank-it vs. could-not-read-it-at-all,
never conflated), that every watched artifact carries a `checked_at`, that
`/api/state` surfaces an integer count, and that a check pass -- which reads
upstreams and writes only the provenance ledger -- leaves lifecycle intent
byte-identical.
"""

import pytest

pytestmark = pytest.mark.safe

SAFE_STATUSES = {"current", "available", "undetermined", "unavailable"}

# POST /api/provenance/check runs the pass SYNCHRONOUSLY on the request
# thread (app/routers/provenance.py:210) and each upstream request carries an
# 8 s timeout with a 40-request budget (app/updates/fetch.py:94), so the
# seeded set's ~10 requests can legitimately outlast the `deck` fixture's
# 10 s client timeout (livetests/conftest.py:57) when one upstream is slow.
# A ReadTimeout there is a test ERROR against a degraded-but-correct backend,
# not a finding -- same reasoning as the lemonade_guard load at
# livetests/conftest.py:123 (timeout=240.0).
CHECK_TIMEOUT_S = 120.0

# Fields of an intent record the checker could plausibly have written.
# DELIBERATELY NOT the whole record: `last_healthy_ts` is re-stamped
# unconditionally by IntentStore.note_healthy (app/intent.py:127), which
# app.arbiter.Watcher._reconcile_pass calls on EVERY ~2 s tick for every
# serving placement (app/arbiter.py:807-809). A real check pass takes seconds,
# so comparing whole records would fail deterministically and cry wolf on the
# deck's single-restorer invariant. `updated_ts` is the sentinel that does
# mean somebody wrote intent: only IntentStore.record() stamps it
# (app/intent.py:112), at every deliberate load/unload/park -- the same
# reasoning app/routers/__init__.py:133-148 already applies to settings drift.
_INTENT_ACTUATION_FIELDS = ("state", "model", "engine", "updated_ts")


def _actuation_facts(state: dict) -> dict:
    return {key: {field: (value["intent"] or {}).get(field)
                  for field in _INTENT_ACTUATION_FIELDS}
            for key, value in state["lifecycle"].items()}


def test_check_route_runs_and_reports_counts(deck):
    body = deck.post("/api/provenance/check", timeout=CHECK_TIMEOUT_S).json()
    assert set(body) >= {"checked", "available"}
    assert body["checked"] >= 1, \
        "no artifact has a watch source declared, or update-checking is disabled"


def test_every_watched_artifact_has_a_verdict_from_the_vocabulary(deck):
    for entry in deck.get("/api/provenance").json()["artifacts"]:
        if not entry.get("watch"):
            continue
        update = entry.get("update")
        assert update is not None, f"{entry['artifact_id']} watched but never checked"
        assert update["status"] in SAFE_STATUSES
        assert update["checked_at"]


def test_state_reports_an_updates_count(deck):
    block = deck.get("/api/state").json()["provenance"]
    assert isinstance(block["updates"], int)


def test_the_checker_never_wrote_intent(deck):
    """The single-restorer invariant, asserted from outside: a check pass
    must leave lifecycle intent untouched. app.update_check reads upstreams
    and records a verdict; app.reconcile stays the only actuator. This
    observes that guarantee through the live API rather than grepping it
    out of source, the same spirit as D13's structural (not textual) proof.

    Compares only `_INTENT_ACTUATION_FIELDS`, never the whole record -- see
    that constant for why the arbiter's own liveness stamping makes a
    whole-record comparison a guaranteed false alarm."""
    before = _actuation_facts(deck.get("/api/state").json())
    deck.post("/api/provenance/check", timeout=CHECK_TIMEOUT_S)
    after = _actuation_facts(deck.get("/api/state").json())
    assert before == after
