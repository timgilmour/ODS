from app import updates
from app.updates import oci


def fake_fetch(responses):
    """responses: {url_substring: {"status_code":…, "headers":…, "json":…}}"""
    calls = []

    def _fetch(url, *, headers=None, method="GET"):
        calls.append(url)
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
    # -- it is a malformed read, folded into "no tags" -> UNAVAILABLE.
    src = {"id": "tags", "check": "oci_tags", "registry": "ghcr.io",
           "repository": "a/b", "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({
        "/token": TOKEN,
        "/tags/list": {"status_code": 200, "headers": {},
                       "json": ["v1.0.0", "v1.1.0"]},
    })
    r = oci.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE


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
