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


def test_rollup_normalizes_an_unrecognized_status_to_unavailable():
    # provenance.json can be corrupt or hand-edited; rollup must never fail
    # a tick over it, and must never leak a non-vocabulary value out either.
    assert updates.rollup(["frobnicate"]) == updates.UNAVAILABLE


def test_rollup_normalization_is_order_independent():
    # Same tied severity either way -- the result must not depend on which
    # element max() happens to see first.
    assert updates.rollup(["frobnicate", updates.UNAVAILABLE]) == updates.UNAVAILABLE
    assert updates.rollup([updates.UNAVAILABLE, "frobnicate"]) == updates.UNAVAILABLE


def test_rollup_garbage_status_is_not_masked_by_a_healthy_one():
    assert updates.rollup([updates.CURRENT, "frobnicate"]) == updates.UNAVAILABLE


def test_checks_and_statuses_are_pinned():
    # Guard the vocabulary itself: an accidental edit here should fail this
    # test, not surface downstream as a silent behavior change.
    assert updates.CHECKS == ("oci_channel", "oci_tags", "git_compare", "git_tags")
    assert updates.STATUSES == ("current", "available", "undetermined", "unavailable")


def test_rollup_full_chain_is_worst_regardless_of_order():
    all_statuses = [updates.CURRENT, updates.AVAILABLE,
                    updates.UNDETERMINED, updates.UNAVAILABLE]
    assert updates.rollup(all_statuses) == updates.UNAVAILABLE
    assert updates.rollup(list(reversed(all_statuses))) == updates.UNAVAILABLE


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


# --- per-check required fields ---------------------------------------------
#
# `validate_watch` used to check `id`/`check`/`pinned`/`order` and stop, so a
# source no checker could execute was ACCEPTED (HTTP 200, written to disk)
# and turned into a permanent `unavailable` with note "checker raised
# KeyError" -- the KeyError swallowed by dispatch's per-source try
# (app/update_check.py:64-66). Ambiguous input is refused, not coerced.
#
# The table below is derived from what each checker actually reads:
#   oci_channel  app/updates/oci.py:60-62  repository, reference, pinned
#   oci_tags     app/updates/oci.py:87-88  repository, pinned
#   git_compare  app/updates/git.py:101,105,107  remote, pinned, ref
#   git_tags     app/updates/git.py:135,139   remote, pinned
# `registry` is NOT required: app/updates/oci.py:59,86 default it to ghcr.io.

_VALID = {
    "oci_channel": {"id": "c", "check": "oci_channel", "registry": "ghcr.io",
                    "repository": "a/b", "reference": "slim",
                    "pinned": "sha256:aa", "order": None},
    "oci_tags": {"id": "t", "check": "oci_tags", "registry": "ghcr.io",
                 "repository": "a/b", "pinned": "v1.0.0", "order": "semver"},
    "git_compare": {"id": "g", "check": "git_compare",
                    "remote": "https://github.com/a/b", "ref": "main",
                    "pinned": "abc123", "order": None},
    "git_tags": {"id": "gt", "check": "git_tags",
                 "remote": "https://github.com/a/b",
                 "pinned": "v1.0.0", "order": "none"},
}


def test_every_check_kind_declares_what_its_checker_needs():
    """COMPLETENESS GUARD. A fifth check added to CHECKS with no required
    fields declared would silently accept anything again."""
    assert set(updates._REQUIRED) == set(updates.CHECKS)
    assert set(_VALID) == set(updates.CHECKS)


@pytest.mark.parametrize("check", sorted(_VALID))
def test_a_valid_source_of_every_check_kind_is_accepted(check):
    updates.validate_watch(dict(_VALID[check]))


@pytest.mark.parametrize("check,field", [
    (check, field) for check, fields in [
        ("oci_channel", ("repository", "reference", "pinned")),
        ("oci_tags", ("repository", "pinned")),
        ("git_compare", ("remote", "ref", "pinned")),
        ("git_tags", ("remote", "pinned")),
    ] for field in fields])
def test_a_source_missing_a_field_its_checker_indexes_is_refused(check, field):
    source = dict(_VALID[check])
    source.pop(field)
    with pytest.raises(updates.BadWatch):
        updates.validate_watch(source)


def test_the_oci_channel_source_from_the_review_is_refused_not_stored():
    """The exact body the final review confirmed reached disk: accepted with
    200, then permanently `unavailable` because check_channel indexes
    `source["repository"]` (app/updates/oci.py:60)."""
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": "x", "check": "oci_channel",
                                "pinned": "sha256:aa", "order": None})


@pytest.mark.parametrize("bad", [42, ["a/b"], {"a": 1}, None, ""])
def test_a_required_field_that_is_not_a_non_empty_string_is_refused(bad):
    source = {**_VALID["oci_channel"], "repository": bad}
    with pytest.raises(updates.BadWatch):
        updates.validate_watch(source)


def test_registry_stays_optional_because_the_checker_defaults_it():
    """app/updates/oci.py:59,86 -- `source.get("registry") or "ghcr.io"`. A
    field the checker defaults is not a field the operator must supply."""
    source = dict(_VALID["oci_channel"])
    source.pop("registry")
    updates.validate_watch(source)


# --- uniqueness is a property of the LIST, not of one source ----------------

def test_validate_watch_sources_accepts_distinct_ids():
    updates.validate_watch_sources([_VALID["oci_channel"], _VALID["git_tags"]])


def test_a_duplicate_source_id_is_refused():
    """`record_update` merges results into `{s["id"]: s}`
    (app/provenance.py:511), so a second source with the same id silently
    suppresses the first one's verdict. README published `id` as "unique
    within the artifact"; nothing enforced it."""
    with pytest.raises(updates.BadWatch):
        updates.validate_watch_sources([
            dict(_VALID["oci_tags"], id="dupe"),
            dict(_VALID["git_tags"], id="dupe")])


def test_validate_watch_sources_still_validates_each_source():
    with pytest.raises(updates.BadWatch):
        updates.validate_watch_sources([{"id": "x", "check": "oci_channel",
                                         "pinned": "sha256:aa", "order": None}])


def test_validate_watch_rejects_a_non_string_id():
    # Task 9's PUT /watch route takes `id` straight off an untyped request
    # body. Truthiness alone lets a list or object through here, and
    # set_watch's `{s["id"] for s in sources}` then raises an unhandled
    # TypeError (unhashable type) -- a 500, not the clean 422 an operator
    # putting a malformed body should get.
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": ["a"], "check": "git_tags",
                                "pinned": "v1", "order": "semver"})
    with pytest.raises(updates.BadWatch):
        updates.validate_watch({"id": {"a": 1}, "check": "git_tags",
                                "pinned": "v1", "order": "semver"})
