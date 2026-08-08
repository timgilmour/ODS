import pytest
from app.updates import ordering


def test_semver_finds_a_newer_tag():
    r = ordering.rank(["v0.5.4", "v0.5.6", "v0.5.5"], "semver", "v0.5.5")
    assert r["rankable"] is True
    assert r["latest"] == "v0.5.6"
    assert r["newer"] == ["v0.5.6"]
    assert r["unranked"] == []


def test_semver_reports_current_when_pin_is_newest():
    # The real ds4 case, verified live 2026-08-08: v0.5.6 is newest and is what runs.
    r = ordering.rank(["v0.5.6", "v0.5.5", "v0.4.0"], "semver", "v0.5.6")
    assert r["latest"] == "v0.5.6"
    assert r["newer"] == []


def test_semver_excludes_unparseable_tags_instead_of_coercing_them():
    # A synthetic tag set with mixed shapes (the real aeon-vllm tag set is
    # exercised verbatim, under its actual declared order, below). An
    # unparseable tag must be REPORTED, never silently compared -- that is
    # the guess this refuses.
    r = ordering.rank(
        ["v0.22.1-pr44389-spark", "v0.5.6", "pre-audio"], "semver", "v0.5.5")
    assert r["latest"] == "v0.5.6"
    assert sorted(r["unranked"]) == ["pre-audio", "v0.22.1-pr44389-spark"]


def test_date_order_excludes_the_real_aeon_vllm_unparseable_tags():
    # The literal real aeon-vllm-ultimate tag set, run under its actual
    # declared order (date), with a parseable pin.
    r = ordering.rank(
        ["2026-07-27-v0.26.0", "v0.22.1-pr44389-spark", "pre-audio"],
        "date", "2026-06-01-v0.25.0")
    assert r["rankable"] is True
    assert r["latest"] == "2026-07-27-v0.26.0"
    assert r["newer"] == ["2026-07-27-v0.26.0"]
    assert sorted(r["unranked"]) == ["pre-audio", "v0.22.1-pr44389-spark"]


def test_date_order_uses_the_leading_date():
    r = ordering.rank(
        ["2026-07-27-v0.26.0", "2026-06-01-v0.25.0"], "date", "2026-06-01-v0.25.0")
    assert r["latest"] == "2026-07-27-v0.26.0"
    assert r["newer"] == ["2026-07-27-v0.26.0"]


def test_newer_preserves_numeric_order_not_lexical_for_semver():
    # v1.10.0 is numerically newer than v1.2.0, but "1" < "2" lexically, so
    # an alphabetical sort of `newer` would wrongly put v1.10.0 first. The
    # ascending numeric order already established by ranking must survive.
    r = ordering.rank(["v1.2.0", "v1.10.0"], "semver", "v1.0.0")
    assert r["newer"] == ["v1.2.0", "v1.10.0"]


def test_newer_preserves_chronological_order_for_date():
    r = ordering.rank(
        ["2026-01-09-rc1", "2026-01-10-rc2", "2026-02-01-rc3"],
        "date", "2026-01-01-rc0")
    assert r["newer"] == ["2026-01-09-rc1", "2026-01-10-rc2", "2026-02-01-rc3"]


def test_unparseable_pin_under_semver_is_reported_not_ranked():
    # The pin itself fails to parse under the declared order. Ranking must
    # be refused outright, not coerced by inventing a "newer" claim.
    r = ordering.rank(
        ["v1.0.0", "v1.5.0", "not-a-version"], "semver", "not-a-version")
    assert r["rankable"] is False
    assert r["latest"] is None
    assert r["newer"] == ["v1.0.0", "v1.5.0"]
    assert r["unranked"] == ["not-a-version"]


def test_unparseable_pin_under_date_is_reported_not_ranked():
    # The real aeon-vllm-ultimate case: the pin in production IS one of the
    # unparseable tags. Reporting `rankable: True` here would be claiming to
    # have ranked something the module just said it could not read.
    r = ordering.rank(
        ["2026-07-27-v0.26.0", "v0.22.1-pr44389-spark", "pre-audio"],
        "date", "v0.22.1-pr44389-spark")
    assert r["rankable"] is False
    assert r["latest"] is None
    assert r["newer"] == ["2026-07-27-v0.26.0", "pre-audio"]
    assert sorted(r["unranked"]) == ["pre-audio", "v0.22.1-pr44389-spark"]


def test_zero_parseable_tags_under_semver_is_reported_not_ranked():
    # Every tag failed to parse. Reporting `newer: []` here would be a false
    # "you are up to date" produced from having read nothing -- the same
    # class of defect as an unparseable pin.
    r = ordering.rank(["pre-audio", "nightly", "edge"], "semver", "v1.0.0")
    assert r["rankable"] is False
    assert r["latest"] is None
    assert r["newer"] == ["edge", "nightly", "pre-audio"]
    assert sorted(r["unranked"]) == ["edge", "nightly", "pre-audio"]


def test_zero_parseable_tags_under_date_is_reported_not_ranked():
    r = ordering.rank(["pre-audio", "nightly", "edge"], "date", "2026-01-01")
    assert r["rankable"] is False
    assert r["latest"] is None
    assert r["newer"] == ["edge", "nightly", "pre-audio"]
    assert sorted(r["unranked"]) == ["edge", "nightly", "pre-audio"]


def test_an_empty_tag_list_is_a_different_fact_from_unparseable_tags():
    # "The registry returned no tags at all" is not "I could not parse the
    # tags it returned" -- this stays `newer: []`, deliberately, and is
    # handled downstream rather than here.
    r = ordering.rank([], "semver", "v1.0.0")
    assert r["rankable"] is False
    assert r["newer"] == []


def test_none_order_refuses_to_rank_and_lists_what_it_saw():
    # The real comfyui-aeon-spark tag set: channel names, no ordering exists.
    r = ordering.rank(["slim", "full", "latest"], "none", "slim")
    assert r["rankable"] is False
    assert r["latest"] is None
    assert sorted(r["newer"]) == ["full", "latest"]


def test_none_order_reports_nothing_new_when_only_the_pin_exists():
    r = ordering.rank(["slim"], "none", "slim")
    assert r["rankable"] is False
    assert r["newer"] == []


def test_a_pin_absent_from_the_tag_list_is_not_an_error():
    r = ordering.rank(["v1.0.0"], "semver", "v0.9.0")
    assert r["newer"] == ["v1.0.0"]


def test_unknown_order_raises():
    with pytest.raises(ValueError):
        ordering.rank(["v1.0.0"], "lexical", "v1.0.0")
