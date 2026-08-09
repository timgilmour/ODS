from app import updates
from app.updates import oci


def fake_fetch(responses):
    """responses: {url_substring: {"status_code":…, "headers":…, "json":…}}"""
    calls = []

    def _fetch(url, *, headers=None, method="GET"):
        calls.append({"url": url, "headers": headers, "method": method})
        for fragment, response in responses.items():
            if fragment in url:
                return response
        raise AssertionError(f"unexpected url {url}")

    _fetch.calls = calls
    return _fetch


TOKEN = {"status_code": 200, "headers": {}, "json": {"token": "t0k"}}


def test_channel_reports_current_when_the_digest_still_matches():
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "aeon-7/comfyui-aeon-spark", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 200,
                            "headers": {"Docker-Content-Digest": "sha256:7fda74d7"},
                            "json": {}},
    })
    r = oci.check_channel(src, fetch)
    assert r["status"] == updates.CURRENT
    assert r["current"] == "sha256:7fda74d7"
    assert r["latest"] == "sha256:7fda74d7"


def test_channel_uses_head_for_the_manifest_request():
    # The manifest request must be a HEAD -- fetching the whole manifest body
    # to read one header would be wasteful, and the reference implementation
    # relies on this. Nothing else in this suite pins the method, so a
    # regression here would pass every other test silently.
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "aeon-7/comfyui-aeon-spark", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 200,
                            "headers": {"Docker-Content-Digest": "sha256:7fda74d7"},
                            "json": {}},
    })
    oci.check_channel(src, fetch)
    manifest_calls = [c for c in fetch.calls if "/manifests/" in c["url"]]
    assert len(manifest_calls) == 1
    assert manifest_calls[0]["method"] == "HEAD"


def test_channel_reports_available_when_the_tag_moved():
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "aeon-7/comfyui-aeon-spark", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 200,
                            "headers": {"Docker-Content-Digest": "sha256:99beef"},
                            "json": {}},
    })
    r = oci.check_channel(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] == "sha256:99beef"


def test_channel_without_a_digest_header_is_unavailable_not_available():
    # A missing header is a failure to READ, not proof the tag moved.
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "a/b", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 200, "headers": {}, "json": {}},
    })
    r = oci.check_channel(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "digest" in r["note"].lower()


def test_channel_401_is_unavailable_with_the_code_recorded():
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "a/b", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 401, "headers": {}, "json": None},
    })
    r = oci.check_channel(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "401" in r["note"]


def test_channel_404_is_unavailable_with_the_code_recorded():
    src = {"id": "slim", "check": "oci_channel", "registry": "ghcr.io",
           "repository": "a/b", "reference": "slim",
           "pinned": "sha256:7fda74d7", "order": None}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/manifests/slim": {"status_code": 404, "headers": {}, "json": None},
    })
    r = oci.check_channel(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "404" in r["note"]


def test_tags_ranks_by_the_declared_order():
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "aeon-7/aeon-vllm-ultimate",
           "pinned": "2026-06-01-v0.25.0", "order": "date"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["2026-06-01-v0.25.0",
                                         "2026-07-27-v0.26.0", "pre-audio"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] == "2026-07-27-v0.26.0"
    assert r["detail"]["unranked"] == ["pre-audio"]


def test_tags_reports_current_when_nothing_ranks_higher():
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["v1.0.0", "v0.9.0"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert r["latest"] == "v1.0.0"
    assert r["detail"]["newer"] == []


def test_tags_with_order_none_is_undetermined():
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "aeon-7/comfyui-aeon-spark",
           "pinned": "slim", "order": "none"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["slim", "full", "latest"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNDETERMINED
    assert r["latest"] is None
    assert sorted(r["detail"]["newer"]) == ["full", "latest"]


def test_tags_undetermined_when_the_pin_itself_is_unparseable_but_siblings_parse():
    # Reached it, ranked the tags it CAN parse, but the pin we hold does not
    # parse under the declared order -- this must not be coerced into a
    # comparison, and it must not be confused with CURRENT just because we
    # happen to know of no tag that beats an unranked pin.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "not-a-version", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["v1.0.0", "v1.2.0"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNDETERMINED
    assert r["latest"] is None
    assert sorted(r["detail"]["newer"]) == ["v1.0.0", "v1.2.0"]


def test_a_non_200_is_unavailable_with_the_code_recorded():
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 404, "headers": {}, "json": None},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "404" in r["note"]


def test_tags_survives_a_malformed_tags_list_body_instead_of_crashing():
    # The fetch contract types `json` as dict | list | None. A registry
    # answering /tags/list with a bare list (not the documented
    # {"tags": [...]} shape) must not crash the checker with AttributeError
    # -- it is a malformed read, distinct in wording from a genuinely empty
    # tag list (see the "distinct notes" tests below).
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": ["v1.0.0", "v1.1.0"]},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" not in r["note"].lower()


def test_tags_survives_a_malformed_token_body_instead_of_crashing():
    # Same contract gap on the /token response: a bare list instead of a
    # dict must degrade to "no token" (unauthenticated request), not crash.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": {"status_code": 200, "headers": {}, "json": ["nope"]},
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["v1.0.0", "v1.1.0"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] == "v1.1.0"


def test_a_non_string_token_value_falls_back_to_unauthenticated():
    """`_token` checked the BODY was a dict but not that `body["token"]` was
    a string, so a numeric or object token f-stringed straight into
    `Authorization: Bearer 42` (app/updates/oci.py:55) -- a wrong header
    instead of no header. Symmetric with the tags-VALUE gap already fixed
    two tests up: validating the container is not validating the value."""
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": {"status_code": 200, "headers": {}, "json": {"token": 42}},
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["v1.0.0", "v1.1.0"]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.AVAILABLE
    tags_call = next(c for c in fetch.calls if "/tags/list" in c["url"])
    assert tags_call["headers"] == {}, "sent a Bearer header built from a non-string"


def test_tags_with_non_string_tag_values_is_unavailable_not_crashing():
    # Reported by the coordinator: {"tags": [1, 2, 3]} previously propagated
    # a bare TypeError out of ordering.rank (regex match against an int).
    # Nothing usable survives a body like this -- it is a failure to READ,
    # the same category as the empty-list and malformed-body cases, not a
    # rankable or even "undetermined" tag set.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": [1, 2, 3]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" not in r["note"].lower()


def test_tags_with_a_bare_string_tags_value_is_not_iterated_character_by_character():
    # Reported by the coordinator: {"tags": "v9.9.9"} does not crash on its
    # own, but a bare string IS a list of its characters to Python -- it
    # must not be iterated into one-character garbage tags.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {"tags": "v9.9.9"}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert r["detail"] == {}
    assert "no tags" not in r["note"].lower()


def test_tags_drops_non_string_elements_and_ranks_the_rest():
    # A partially-malformed list (some elements wrong-typed, some good)
    # should not be thrown away wholesale -- filter to the usable elements
    # and rank those, same as if the bad ones were never sent.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": {"tags": ["v1.0.0", 42, "v1.2.0", None]}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] == "v1.2.0"
    assert r["detail"]["tag_count"] == 2


def test_tags_with_an_empty_tag_list_is_unavailable_not_current():
    # CONTROLLER RULING: an empty tag list is a failure to learn anything
    # about the repository, not evidence of being up to date. The brief's
    # reference logic maps non-rankable + no `newer` to CURRENT, which would
    # report "current" here on the basis of having seen zero tags. That is
    # wrong: no tags at all must be UNAVAILABLE, distinct from tags-present-
    # but-unparseable (UNDETERMINED, covered above).
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {"tags": []}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" in r["note"].lower()
    assert r["latest"] is None


def test_tags_with_a_null_json_body_on_200_is_malformed_not_no_tags():
    # A 200 with no JSON body at all (json: None) is closer to "we could not
    # parse a tag set from this" than to "the registry affirmatively told us
    # tags: []" -- it gets the malformed note, not the "no tags" one.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": None},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" not in r["note"].lower()


def test_tags_missing_key_is_the_same_as_an_empty_list():
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {}},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" in r["note"].lower()


def test_malformed_and_empty_tags_notes_are_distinguishable():
    # Minor from the coordinator: an operator must be able to tell "the
    # registry told us it has nothing" from "we couldn't parse what it
    # sent" by reading `note` alone. Assert the two vocabularies never
    # overlap rather than eyeballing individual strings.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}

    empty = oci.check_tags(src, fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {"tags": []}},
    }))
    malformed_body = oci.check_tags(src, fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": ["a"]},
    }))
    malformed_field = oci.check_tags(src, fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {"tags": "a"}},
    }))
    malformed_elements = oci.check_tags(src, fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {}, "json": {"tags": [1]}},
    }))

    assert empty["note"] != malformed_body["note"]
    assert "no tags" in empty["note"].lower()
    for r in (malformed_body, malformed_field, malformed_elements):
        assert "no tags" not in r["note"].lower()
