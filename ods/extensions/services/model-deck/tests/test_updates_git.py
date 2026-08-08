from app import updates
from app.updates import git


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


# --- parse_remote -----------------------------------------------------------

def test_parse_remote_reads_owner_and_repo():
    assert git.parse_remote("https://github.com/Entrpi/ds4") == ("Entrpi", "ds4")
    assert git.parse_remote("https://github.com/Entrpi/ds4.git") == ("Entrpi", "ds4")


def test_parse_remote_returns_none_for_non_github():
    assert git.parse_remote("https://gitlab.com/a/b") is None


def test_parse_remote_handles_none_and_empty_string():
    # A caller with a missing/blank remote must not crash the regex match.
    assert git.parse_remote(None) is None
    assert git.parse_remote("") is None


def test_non_github_remote_is_unavailable_not_a_crash():
    src = {"id": "up", "check": "git_compare", "remote": "https://gitlab.com/a/b",
           "ref": "main", "pinned": "abc123", "order": None}
    r = git.check_compare(src, fake_fetch({}))
    assert r["status"] == updates.UNAVAILABLE
    assert "github" in r["note"].lower()


# --- check_compare: basic verdicts ------------------------------------------

def test_compare_reports_available_when_behind():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/warpfront/hipfire",
           "ref": "master", "pinned": "5d3683a7", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"status": "behind", "ahead_by": 0, "behind_by": 12,
                 "commits": [{"sha": "b91f0e22"}]}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["detail"]["behind_by"] == 12
    assert r["latest"] == "b91f0e22"


def test_compare_reports_current_when_identical():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc123", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"status": "identical", "ahead_by": 0, "behind_by": 0,
                 "commits": []}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.CURRENT


def test_a_moved_repository_is_reported_in_the_note():
    # Live on 2026-08-08: Kaden-Schutt/hipfire 301s to warpfront/hipfire,
    # one day after the origin was declared. Upstreams move.
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/Kaden-Schutt/hipfire",
           "ref": "master", "pinned": "5d3683a7", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 301, "headers": {
            "Location": "https://api.github.com/repositories/12345/compare/x...y"},
        "json": None}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert r["detail"]["moved"] is True


def test_rate_limited_is_unavailable_and_says_so():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 403, "headers": {"X-RateLimit-Remaining": "0"},
        "json": None}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "rate limit" in r["note"].lower()


def test_403_without_rate_limit_header_is_ordinary_failure_not_rate_limit():
    # A 403 is not automatically a rate-limit report -- only a confirmed
    # X-RateLimit-Remaining: 0 earns that specific note. Anything else
    # (missing header, or present but nonzero) is an ordinary failure.
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 403, "headers": {"X-RateLimit-Remaining": "42"},
        "json": None}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "rate limit" not in r["note"].lower()
    assert "403" in r["note"]


# --- check_compare: malformed / defensive commit handling -------------------

def test_compare_missing_commits_key_does_not_crash():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"status": "behind", "ahead_by": 0, "behind_by": 5}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] is None


def test_compare_non_list_commits_does_not_crash():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"ahead_by": 0, "behind_by": 3, "commits": {"weird": True}}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] is None


def test_compare_non_dict_last_commit_does_not_crash():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"ahead_by": 0, "behind_by": 2, "commits": ["just-a-sha-string"]}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] is None


def test_compare_non_int_behind_by_does_not_crash():
    # `behind_by` is the verdict-determining field. A non-int value (e.g. a
    # garbled string) must not be trusted into a comparison -- `"12" > 0`
    # raises TypeError in Python 3, and even if it didn't, coercing an
    # untrusted shape into a verdict is exactly the guessing this package
    # refuses to do.
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {},
        "json": {"behind_by": "12", "ahead_by": 0, "commits": []}}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.CURRENT
    assert r["detail"]["behind_by"] == 0


def test_compare_malformed_body_not_a_dict_is_unavailable():
    src = {"id": "up", "check": "git_compare",
           "remote": "https://github.com/a/b", "ref": "main",
           "pinned": "abc", "order": None}
    fetch = fake_fetch({"/compare/": {
        "status_code": 200, "headers": {}, "json": ["oops", "not", "a", "dict"]}})
    r = git.check_compare(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "malformed" in r["note"].lower()


# --- check_tags: basic verdict + ranking -------------------------------------

def test_git_tags_ranks_with_semver():
    # The real ds4 tag list, verified live: v0.5.6 is newest and is the pin.
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/Entrpi/ds4",
           "pinned": "v0.5.6", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {},
        "json": [{"name": "v0.5.6"}, {"name": "v0.5.5"}, {"name": "v0.4.0"}]}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert r["latest"] == "v0.5.6"


def test_git_tags_carries_divergence_as_context_not_a_verdict():
    # ds4's tags live on batched-serving, 569/216 diverged from main. That is
    # context on the tag answer, never a status of its own.
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/Entrpi/ds4",
           "pinned": "v0.5.5", "order": "semver",
           "default_branch": "main", "ref": "batched-serving"}
    fetch = fake_fetch({
        "/tags": {"status_code": 200, "headers": {},
                  "json": [{"name": "v0.5.6"}, {"name": "v0.5.5"}]},
        "/compare/": {"status_code": 200, "headers": {},
                      "json": {"status": "diverged", "ahead_by": 569,
                               "behind_by": 216, "commits": []}},
    })
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.AVAILABLE
    assert r["latest"] == "v0.5.6"
    assert r["detail"]["diverges_from_default"] == {"ahead_by": 569, "behind_by": 216}


def test_git_tags_rate_limited_reuses_failure_classification():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 403, "headers": {"X-RateLimit-Remaining": "0"},
        "json": None}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "rate limit" in r["note"].lower()


# --- check_tags: CONTROLLER RULING 1 -- empty tag list is unavailable -------

def test_git_tags_empty_list_is_unavailable_not_current():
    # Zero tags returned is a failure to LEARN anything, not evidence of
    # being current. ordering.rank([], ...) hands back newer=[], and the
    # naive `UNDETERMINED if newer else CURRENT` mapping would read that as
    # "nothing newer" -- a false CURRENT built on having seen nothing.
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {}, "json": []}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "no tags" in r["note"].lower()


# --- check_tags: CONTROLLER RULING 2 -- validate response shape ------------

def test_git_tags_dict_body_is_unavailable_not_a_crash():
    # GitHub error bodies are dicts (e.g. {"message": "Not Found"}). Iterating
    # a dict yields its keys (strings), and calling .get on a string crashes.
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {},
        "json": {"message": "Not Found", "documentation_url": "https://x"}}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "malformed" in r["note"].lower()
    # distinct from the empty-list note -- "we got nothing" vs "we could
    # not read what we got" must stay tellable apart.
    assert "no tags" not in r["note"].lower()


def test_git_tags_non_dict_elements_are_unavailable_not_a_crash():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {}, "json": ["v1.0.0", "v1.1.0"]}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "malformed" in r["note"].lower()


def test_git_tags_all_names_missing_or_non_string_is_unavailable():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {},
        "json": [{"notname": "x"}, {"name": 123}, {"name": None}]}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.UNAVAILABLE
    assert "malformed" in r["note"].lower()


def test_git_tags_names_must_be_strings_survivors_still_rank():
    # A mix of well-formed and garbage entries: garbage is dropped, and
    # anything usable still drives a real answer rather than bailing out
    # entirely.
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver"}
    fetch = fake_fetch({"/tags": {
        "status_code": 200, "headers": {},
        "json": [{"name": "v1.0.0"}, {"name": 123}, {"notname": "x"},
                 "not-a-dict", {"name": None}]}})
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert r["latest"] == "v1.0.0"
    assert r["detail"]["tag_count"] == 1


# --- _divergence: context, never a verdict, never a crash -------------------

def test_divergence_fetch_failure_is_omitted_not_a_downgrade():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver",
           "default_branch": "main", "ref": "dev"}
    fetch = fake_fetch({
        "/tags": {"status_code": 200, "headers": {},
                  "json": [{"name": "v1.0.0"}]},
        "/compare/": {"status_code": 500, "headers": {}, "json": None},
    })
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert "diverges_from_default" not in r["detail"]


def test_divergence_malformed_body_is_omitted_not_a_crash():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver",
           "default_branch": "main", "ref": "dev"}
    fetch = fake_fetch({
        "/tags": {"status_code": 200, "headers": {},
                  "json": [{"name": "v1.0.0"}]},
        "/compare/": {"status_code": 200, "headers": {},
                      "json": ["not", "a", "dict"]},
    })
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert "diverges_from_default" not in r["detail"]


def test_divergence_non_int_fields_are_omitted_not_a_crash():
    src = {"id": "tags", "check": "git_tags",
           "remote": "https://github.com/a/b",
           "pinned": "v1.0.0", "order": "semver",
           "default_branch": "main", "ref": "dev"}
    fetch = fake_fetch({
        "/tags": {"status_code": 200, "headers": {},
                  "json": [{"name": "v1.0.0"}]},
        "/compare/": {"status_code": 200, "headers": {},
                      "json": {"ahead_by": "569", "behind_by": None}},
    })
    r = git.check_tags(src, fetch)
    assert r["status"] == updates.CURRENT
    assert "diverges_from_default" not in r["detail"]
