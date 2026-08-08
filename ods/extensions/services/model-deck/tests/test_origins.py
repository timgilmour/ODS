"""Origin kinds: identity, origin shapes, and grading. Pure — no I/O anywhere."""

import pytest

from app import origins
from app.origins import file as file_origin
from app.origins import git as git_origin
from app.origins import oci as oci_origin


# --- artifact ids ---------------------------------------------------------

def test_parse_splits_kind_node_ref():
    assert origins.parse_artifact_id("oci:sparky:ds4-spark") == ("oci", "sparky", "ds4-spark")


def test_parse_keeps_colons_and_slashes_in_the_ref():
    # A weights relpath can contain both; only the first two colons delimit.
    assert origins.parse_artifact_id("file:local:sub/dir:odd/model.gguf") == (
        "file", "local", "sub/dir:odd/model.gguf")


@pytest.mark.parametrize("bad", [
    "", "oci", "oci:sparky", "zip:sparky:thing", "oci::thing", "oci:sparky:",
    "oci:Sparky:thing",
])
def test_parse_refuses_malformed_ids(bad):
    with pytest.raises(origins.BadArtifactId):
        origins.parse_artifact_id(bad)


def test_build_round_trips_through_parse():
    built = origins.build_artifact_id("file", "local", "a/b.gguf")
    assert built == "file:local:a/b.gguf"
    assert origins.parse_artifact_id(built) == ("file", "local", "a/b.gguf")


# --- oci ------------------------------------------------------------------

def test_oci_origin_local_image_has_no_registry():
    assert oci_origin.origin_from_reference("ds4-spark:v0.5.6") == {
        "registry": None, "repository": "ds4-spark",
        "reference": "ds4-spark:v0.5.6", "build": None, "archive": None,
    }


def test_oci_origin_recognises_a_registry_host_by_dot_or_port():
    assert oci_origin.origin_from_reference(
        "ghcr.io/osmantic/ds4:v1")["registry"] == "ghcr.io"
    assert oci_origin.origin_from_reference(
        "localhost:5000/ds4:v1")["registry"] == "localhost:5000"


def test_oci_origin_treats_a_bare_org_prefix_as_repository_not_registry():
    # "osmantic" has no dot, no colon, and is not localhost -> Docker Hub org.
    o = oci_origin.origin_from_reference("osmantic/ds4:v1")
    assert o["registry"] is None
    assert o["repository"] == "osmantic/ds4"


def test_oci_origin_strips_a_digest_pin_from_the_repository():
    o = oci_origin.origin_from_reference("ds4-spark@sha256:abc")
    assert o["repository"] == "ds4-spark"
    assert o["reference"] == "ds4-spark@sha256:abc"


def test_oci_origin_refuses_an_empty_reference():
    with pytest.raises(ValueError):
        oci_origin.origin_from_reference("   ")


def test_oci_identity_from_inspect_splits_content_id_from_human_tag():
    body = {"Image": "sha256:deadbeef", "Config": {"Image": "ods-hipfire:latest"}}
    assert oci_origin.identity_from_inspect(body) == {
        "version": "sha256:deadbeef", "label": "ods-hipfire:latest", "detail": {}}


def test_oci_identity_from_inspect_tolerates_a_missing_config_block():
    assert oci_origin.identity_from_inspect({"Image": "sha256:x"})["label"] is None


def test_oci_identity_from_compose_keeps_the_tag_without_a_digest():
    assert oci_origin.identity_from_compose("ds4-spark:v0.5.6", None) == {
        "version": None, "label": "ds4-spark:v0.5.6", "detail": {}}


def test_oci_grade_is_exact_with_a_digest_and_unknown_without_one():
    assert oci_origin.grade({"version": "sha256:x", "label": None, "detail": {}}) == origins.EXACT
    assert oci_origin.grade({"version": None, "label": "t", "detail": {}}) == origins.UNKNOWN


# --- git ------------------------------------------------------------------

def test_git_origin_records_divergence_from_the_default_branch():
    o = git_origin.origin("https://github.com/x/ds4", "v0.5.6", "tag",
                          default_branch="main", diverges_from_default=True)
    assert o == {"remote": "https://github.com/x/ds4", "ref": "v0.5.6",
                 "ref_kind": "tag", "default_branch": "main",
                 "diverges_from_default": True}


def test_git_origin_refuses_an_unknown_ref_kind():
    with pytest.raises(ValueError):
        git_origin.origin("r", "v1", "sha")


def test_git_grade_is_unavailable_without_a_git_runner():
    # D10: the deck container cannot see host repos in v1.
    assert git_origin.grade(git_origin.identity(None, "v0.5.6")) == origins.UNAVAILABLE


def test_git_grade_is_exact_when_a_runner_produced_a_commit():
    identity = git_origin.identity("4ad370b4", "v0.5.6")
    assert git_origin.grade(identity, run_git=object()) == origins.EXACT


# --- file -----------------------------------------------------------------

def test_file_origin_defaults_to_unknown_source_kind():
    assert file_origin.origin() == {"source_kind": "unknown", "repo_id": None,
                                    "revision": None, "filename": None, "url": None}


def test_file_origin_refuses_an_unknown_source_kind():
    with pytest.raises(ValueError):
        file_origin.origin(source_kind="magnet")


def test_file_identity_from_unit_is_a_fingerprint_not_a_version():
    unit = {"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
            "size": 4096, "mtime": 1754689383.2}
    assert file_origin.identity_from_unit(unit) == {
        "version": None, "label": "m.gguf",
        "detail": {"size_bytes": 4096, "mtime": 1754689383.2}}


def test_file_grade_is_consistent_when_measured_and_unavailable_when_not():
    identity = file_origin.identity_from_unit(
        {"name": "m.gguf", "size": 1, "mtime": 2.0})
    assert file_origin.grade(identity, available=True) == origins.CONSISTENT
    assert file_origin.grade(identity, available=False) == origins.UNAVAILABLE


def test_file_grade_is_unknown_with_no_fingerprint_at_all():
    assert file_origin.grade({"version": None, "label": None, "detail": {}},
                             available=True) == origins.UNKNOWN


def test_file_deep_check_reports_agreement_not_a_verification_state():
    # A mismatch means the BYTES changed, not that the check is less certain
    # — the stored state is EXACT either way, because the file was hashed.
    assert file_origin.matches_recorded("abc", "abc") is True
    assert file_origin.matches_recorded("abc", "xyz") is False


def test_file_deep_check_of_a_first_hash_has_nothing_to_disagree_with():
    assert file_origin.matches_recorded("abc", None) is None
