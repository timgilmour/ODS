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


def test_check_route_runs_and_reports_counts(deck):
    body = deck.post("/api/provenance/check").json()
    assert set(body) >= {"checked", "available"}
    assert body["checked"] >= 1, "no artifact has a watch source declared"


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
    out of source, the same spirit as D13's structural (not textual) proof."""
    before = deck.get("/api/state").json()["lifecycle"]
    deck.post("/api/provenance/check")
    after = deck.get("/api/state").json()["lifecycle"]
    assert {k: v["intent"] for k, v in before.items()} == \
           {k: v["intent"] for k, v in after.items()}
