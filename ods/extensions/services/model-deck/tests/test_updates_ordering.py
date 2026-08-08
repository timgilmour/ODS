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
    # The real aeon-vllm tag set mixes shapes. An unparseable tag must be
    # REPORTED, never silently compared -- that is the guess this refuses.
    r = ordering.rank(
        ["v0.22.1-pr44389-spark", "v0.5.6", "pre-audio"], "semver", "v0.5.5")
    assert r["latest"] == "v0.5.6"
    assert sorted(r["unranked"]) == ["pre-audio", "v0.22.1-pr44389-spark"]


def test_date_order_uses_the_leading_date():
    r = ordering.rank(
        ["2026-07-27-v0.26.0", "2026-06-01-v0.25.0"], "date", "2026-06-01-v0.25.0")
    assert r["latest"] == "2026-07-27-v0.26.0"
    assert r["newer"] == ["2026-07-27-v0.26.0"]


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
