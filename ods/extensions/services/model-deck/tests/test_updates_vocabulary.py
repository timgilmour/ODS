import pytest
from app import updates


def test_rollup_takes_the_worst_status():
    # lemonade has two independently drifting sources; one must not mask the other.
    assert updates.rollup([updates.CURRENT, updates.AVAILABLE]) == updates.AVAILABLE
    assert updates.rollup([updates.AVAILABLE, updates.UNDETERMINED]) == updates.UNDETERMINED
    assert updates.rollup([updates.UNDETERMINED, updates.UNAVAILABLE]) == updates.UNAVAILABLE
    assert updates.rollup([updates.CURRENT, updates.CURRENT]) == updates.CURRENT


def test_rollup_of_nothing_is_unavailable():
    # watch: [] -- no origin, nothing to check. Not "current"; we never looked.
    assert updates.rollup([]) == updates.UNAVAILABLE


def test_undetermined_is_not_unavailable():
    # Distinct words on purpose: one is fixable by declaring an order,
    # the other by fixing the network. Collapsing them hides which.
    assert updates.UNDETERMINED != updates.UNAVAILABLE


def test_validate_watch_accepts_a_git_compare_source():
    updates.validate_watch({
        "id": "upstream", "check": "git_compare", "label": "HIPFIRE_REF",
        "remote": "https://github.com/warpfront/hipfire",
        "ref": "master", "pinned": "5d3683a7", "order": None})


def test_validate_watch_rejects_an_unknown_check():
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": "x", "check": "telepathy", "pinned": "a"})


def test_validate_watch_requires_an_order_for_tag_checks():
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": "x", "check": "git_tags",
                                "remote": "https://github.com/a/b",
                                "pinned": "v1.0.0", "order": None})


def test_validate_watch_rejects_an_order_on_non_tag_checks():
    # An order on a digest comparison is meaningless and would imply ranking
    # happens where it does not.
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": "x", "check": "oci_channel",
                                "repository": "a/b", "reference": "slim",
                                "pinned": "sha256:aa", "order": "semver"})


def test_validate_watch_requires_an_id_and_a_pin():
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"check": "git_tags", "pinned": "v1",
                                "order": "semver"})
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": "x", "check": "git_tags", "order": "semver"})
