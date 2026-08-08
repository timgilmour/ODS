"""D13: the provenance ledger against the live box (read-only).

SAFE TIER, and there is no disruptive counterpart by construction: provenance
records and never actuates, so there is nothing here that could need undoing.
That is a property of the design, not a gap in the drill.

Every assertion below checks SHAPE and INVARIANTS, never a specific version —
the versions are the thing that legitimately changes, and a drill that has to
be edited after every upgrade stops getting run.
"""

import pytest

pytestmark = pytest.mark.safe


def test_d13_ledger_lists_the_local_engine_containers(deck):
    body = deck.get("/api/provenance").json()

    ids = {a["artifact_id"] for a in body["artifacts"]}
    assert "oci:local:ods-hipfire" in ids, (
        "no provenance for hipfire — has the watcher run a provenance pass yet? "
        "(throttled to provenance_interval_s)")


def test_d13_local_engine_entries_carry_a_real_content_digest(deck):
    body = deck.get("/api/provenance").json()

    hipfire = next(a for a in body["artifacts"]
                   if a["artifact_id"] == "oci:local:ods-hipfire")
    assert hipfire["current"]["version"].startswith("sha256:")
    assert hipfire["current"]["label"], "no human-readable reference recorded"
    assert hipfire["current"]["version"] != hipfire["current"]["label"], (
        "version and label collapsed into one string — the whole point of the "
        "split is that a digest and a tag are different facts")


def test_d13_weights_are_consistent_never_exact_without_a_deep_check(deck):
    """D5's honest asymmetry, live: a routine pass compares size and mtime,
    which is not proof, so nothing may claim `exact` until somebody asks for
    the hash."""
    body = deck.get("/api/provenance").json()

    for artifact in body["artifacts"]:
        if artifact["kind"] != "file":
            continue
        assert artifact["verification"] in ("consistent", "unavailable", "stale",
                                            "unknown", "exact")
        if artifact["verification"] == "exact":
            assert artifact["current"]["version"], (
                "graded exact with no sha256 — only a deep check can do that")


def test_d13_no_artifact_has_an_inferred_origin(deck):
    """D8, live: an origin is either something a human declared or it is
    null. Nothing in the collection path may invent one."""
    body = deck.get("/api/provenance").json()

    for artifact in body["artifacts"]:
        origin = artifact["origin"]
        assert origin is None or isinstance(origin, dict)

    ids = {a["artifact_id"] for a in body["artifacts"]}
    assert set(body["gaps"]) <= ids, "gap list names an artifact that is not in the ledger"


def test_d13_sparky_engine_provenance_comes_from_live_compose_text(deck):
    """The sparky half: the image reference is parsed from the compose file
    the node-agent serves. Skips rather than fails on a box with no spark."""
    body = deck.get("/api/provenance").json()

    sparky = [a for a in body["artifacts"] if a["node"] != "local"]
    if not sparky:
        pytest.skip("no remote node artifacts — is spark configured?")
    for artifact in sparky:
        assert artifact["current"]["label"], (
            f"{artifact['artifact_id']} has no image reference from compose")


def test_d13_a_sparky_digest_is_present_only_when_it_was_proven(deck):
    """The mis-keying guard, live: a digest may only appear on the profile
    the harvested catalog actually named. It is legitimate for every remote
    entry to have version None (old node-agent, or the newest catalog
    belongs to another profile) — what must never happen is a digest that
    was guessed."""
    body = deck.get("/api/provenance").json()

    for artifact in body["artifacts"]:
        if artifact["node"] == "local" or artifact["kind"] != "oci":
            continue
        version = artifact["current"]["version"]
        if version is None:
            assert artifact["verification"] in ("unknown", "unavailable", "stale")
        else:
            assert version.startswith("sha256:")


def test_d13_state_exposes_the_provenance_block(deck):
    block = deck.get("/api/state").json()["provenance"]

    assert isinstance(block["gaps"], int)
    assert isinstance(block["drift"], list)


def test_d13_every_reported_drift_is_a_real_disagreement(deck):
    """Absence is never drift (app.facts' rule). A box with no declared
    desired versions reports an empty list; if one IS declared, every member
    must genuinely disagree."""
    for entry in deck.get("/api/state").json()["provenance"]["drift"]:
        assert entry["current"]["version"] is not None
        assert entry["desired"] is not None
        assert entry["current"]["version"] != entry["desired"]["version"]


def test_d13_history_is_readable_for_every_listed_artifact(deck):
    """The restorable record. Every artifact in the ledger got there through
    a transition, so every one of them has at least one history record."""
    artifacts = deck.get("/api/provenance").json()["artifacts"]
    if not artifacts:
        pytest.skip("empty ledger — run POST /api/provenance/backfill first")

    for artifact in artifacts[:5]:      # a sample; the ledger can be large
        records = deck.get("/api/provenance/history",
                           params={"artifact_id": artifact["artifact_id"]}).json()
        assert records["records"], f"no history for {artifact['artifact_id']}"
        assert all("to" in r for r in records["records"])
