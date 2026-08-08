"""Collection: observed inputs -> provenance entries. Pure — no clients."""

from app import origins
from app import provenance_collect as collect

_COMPOSE_DS4 = """
services:
  ds4:
    image: ds4-spark:v0.5.6
    container_name: ds4
    volumes:
      - /home/tim/models/Laguna:/model:ro
    command:
      - --served-model-name
      - laguna
"""


# --- local oci ------------------------------------------------------------

def test_local_oci_entry_carries_digest_and_tag():
    entries = collect.local_oci_entries(
        {"ods-hipfire": {"Image": "sha256:a",
                         "Config": {"Image": "ods-hipfire:latest"}}},
        node="local")
    assert entries == [{
        "artifact_id": "oci:local:ods-hipfire", "kind": "oci", "node": "local",
        "role": "engine",
        "current": {"version": "sha256:a", "label": "ods-hipfire:latest",
                    "detail": {}, "verification": origins.EXACT}}]


def test_local_oci_skips_a_container_that_could_not_be_inspected():
    # The collector passes None for a container it failed to read; that is
    # mark_unavailable's job upstream, not a fabricated entry here.
    assert collect.local_oci_entries({"ods-hipfire": None}, node="local") == []


def test_local_oci_is_in_name_order():
    bodies = {"z-engine": {"Image": "sha256:z", "Config": {}},
              "a-engine": {"Image": "sha256:a", "Config": {}}}
    assert [e["artifact_id"] for e in collect.local_oci_entries(bodies, node="local")] == [
        "oci:local:a-engine", "oci:local:z-engine"]


# --- local weights --------------------------------------------------------

def test_weights_entry_is_keyed_on_relpath_not_the_catalog_unit_id():
    # THE MOVE-STABILITY TRAP (design doc Section 1): catalog ids embed the
    # location and record_moved() rewrites them, so keying on the unit id
    # would orphan a model's provenance exactly when the mover runs.
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}]
    moved = [{**units[0], "id": "cold:m.gguf", "location": "cold"}]
    assert (collect.local_file_entries(units, node="local")[0]["artifact_id"]
            == collect.local_file_entries(moved, node="local")[0]["artifact_id"]
            == "file:local:m.gguf")


def test_weights_entry_grades_consistent_and_carries_the_fingerprint():
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 4096, "mtime": 2.0, "state": "resident"}]
    current = collect.local_file_entries(units, node="local")[0]["current"]
    assert current["version"] is None            # a fingerprint is not a version
    assert current["detail"] == {"size_bytes": 4096, "mtime": 2.0}
    assert current["verification"] == origins.CONSISTENT


def test_weights_on_an_unavailable_location_grade_unavailable_not_missing():
    units = [{"id": "cold:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "cold", "size": 4096, "mtime": 2.0,
              "state": "unavailable"}]
    entry = collect.local_file_entries(units, node="local")[0]
    assert entry["current"]["verification"] == origins.UNAVAILABLE


def test_weights_role_is_weights():
    units = [{"id": "hot:m.gguf", "name": "m.gguf", "relpath": "m.gguf",
              "location": "hot", "size": 1, "mtime": 2.0, "state": "resident"}]
    assert collect.local_file_entries(units, node="local")[0]["role"] == "weights"


def test_weights_skips_a_unit_with_no_relpath():
    assert collect.local_file_entries([{"id": "hot:x", "name": "x"}],
                                      node="local") == []


def test_weights_relpath_with_slashes_survives_id_construction():
    units = [{"id": "comfy:loras/x.safetensors", "name": "x.safetensors",
              "relpath": "loras/x.safetensors", "location": "comfy",
              "size": 1, "mtime": 2.0, "state": "resident"}]
    assert (collect.local_file_entries(units, node="local")[0]["artifact_id"]
            == "file:local:loras/x.safetensors")


# --- sparky oci -----------------------------------------------------------

def test_spark_entry_takes_the_repository_from_the_compose_image_line():
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, None, node="sparky")
    assert entries[0]["artifact_id"] == "oci:sparky:ds4-spark"
    assert entries[0]["current"]["label"] == "ds4-spark:v0.5.6"


def test_spark_entry_has_no_digest_without_a_matching_catalog():
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, None, node="sparky")
    assert entries[0]["current"]["version"] is None
    assert entries[0]["current"]["verification"] == origins.UNKNOWN


def test_spark_entry_takes_the_digest_when_the_catalog_names_this_profile():
    catalog = {"profile": "ds4", "image_id": "sha256:live",
               "harvested_ts": "2026-08-08T00:00:00Z", "engine": "vllm"}
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, catalog, node="sparky")
    assert entries[0]["current"]["version"] == "sha256:live"
    assert entries[0]["current"]["verification"] == origins.EXACT


def test_spark_entry_refuses_a_digest_from_a_DIFFERENT_profile():
    # D12's whole reason for existing: the newest catalog usually belongs to
    # some other profile, and attributing its digest here would record a
    # confidently wrong version.
    catalog = {"profile": "heretic", "image_id": "sha256:wrong",
               "harvested_ts": "2026-08-08T00:00:00Z", "engine": "vllm"}
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, catalog, node="sparky")
    assert entries[0]["current"]["version"] is None


def test_spark_entry_refuses_a_digest_from_an_UNSTAMPED_catalog():
    # A node-agent older than the profile stamp has no profile field.
    # Absent, not guessed.
    catalog = {"image_id": "sha256:unknown-owner",
               "harvested_ts": "2026-08-08T00:00:00Z", "engine": "vllm"}
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, catalog, node="sparky")
    assert entries[0]["current"]["version"] is None


def test_spark_skips_a_profile_whose_compose_has_no_image_line():
    text = "services:\n  x:\n    container_name: x\n"
    assert collect.spark_oci_entries({"x": text}, None, node="sparky") == []


def test_spark_skips_unparseable_compose_rather_than_raising():
    assert collect.spark_oci_entries({"x": "services: [unclosed"}, None,
                                     node="sparky") == []


def test_spark_skips_a_multi_service_compose():
    text = ("services:\n  a:\n    image: a:1\n  b:\n    image: b:1\n")
    assert collect.spark_oci_entries({"pair": text}, None, node="sparky") == []


def test_spark_role_is_engine():
    entries = collect.spark_oci_entries({"ds4": _COMPOSE_DS4}, None, node="sparky")
    assert entries[0]["role"] == "engine"
