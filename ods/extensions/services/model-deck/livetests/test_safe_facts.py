"""D10: characteristics and drift against the live box (read-only).

Non-disruptive on purpose: everything here is observable without changing
anything. A drift check that required breaking something to demonstrate
itself would never be run.
"""

import pytest

pytestmark = pytest.mark.safe


def test_d10_facts_have_provenance(deck):
    facts = deck.get("/api/facts").json()

    assert facts, "no facts derived on a box with checkpoints on disk"
    for key, fields in facts.items():
        for name, field in fields.items():
            assert field["origin"] in ("derived", "declared"), f"{key}.{name}"
            assert field["source"], f"{key}.{name} has no source"


def test_d10_identity_is_the_directory_name(deck):
    """The naming rule, verified against reality: every derived identity
    must match its own key, with no alias anywhere."""
    facts = deck.get("/api/facts").json()

    for key, fields in facts.items():
        if not key.startswith("model/") or "identity" not in fields:
            continue
        assert fields["identity"]["value"] == key.split("/", 1)[1]


def test_d10_drift_is_reported_and_nothing_is_corrected(deck):
    before = deck.get("/api/facts").json()

    drift = deck.get("/api/facts/drift").json()

    after = deck.get("/api/facts").json()
    assert before == after, "reading drift must not mutate facts"
    for entries in drift.values():
        for entry in entries:
            assert entry["severity"] in ("crash", "mismatch")
            assert entry["expected_source"]
