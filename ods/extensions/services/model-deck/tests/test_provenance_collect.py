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

# THE ARTIFACT IS THE IMAGE, NOT THE DEPLOYMENT (keying decision, 2026-08-08).
#
# Local OCI used to key on the CONTAINER name while sparky keyed on the image
# repository — two nodes, two rules. Worse, the thing an operator declares
# ("aeon-vllm has no source repo, it is archived at /mnt/cold/images") is a
# fact about the IMAGE; keying per deployment means typing it once per
# container that runs it and maintaining N copies.
#
# So identity is the repository on BOTH nodes, and the containers/profiles
# referencing it are placement facts — exactly the rule the design already
# applies to weights, where identity is the relpath and `location` is "a
# current-placement fact, not part of identity".
#
# ⚠ These fixtures deliberately use containers whose NAME DIFFERS FROM THEIR
# REPOSITORY. `ods-hipfire` runs `ods-hipfire:latest`, so a test built on it
# passes under either rule and proves nothing — the same coincidence that let
# node_label ("local") masquerade as the node id for two days.

def test_local_oci_is_keyed_on_the_image_repository_not_the_container_name():
    entries = collect.local_oci_entries(
        {"ods-comfyui": {"Image": "sha256:a",
                         "Config": {"Image": "ignatberesnev/comfyui-gfx1151:v0.2"}}},
        node="local")
    assert entries == [{
        "artifact_id": "oci:local:ignatberesnev/comfyui-gfx1151",
        "kind": "oci", "node": "local", "role": "engine",
        "current": {"version": "sha256:a",
                    "label": "ignatberesnev/comfyui-gfx1151:v0.2",
                    "detail": {"containers": ["ods-comfyui"]},
                    "verification": origins.EXACT}}]


def test_local_oci_collapses_containers_sharing_one_image():
    """Two deployments of one image are one artifact, with both recorded as
    placement facts."""
    body = {"Image": "sha256:a", "Config": {"Image": "ods-lemonade-server:latest"}}
    entries = collect.local_oci_entries(
        {"ods-llama-server": dict(body), "ods-llama-2": dict(body)}, node="local")

    assert len(entries) == 1
    assert entries[0]["artifact_id"] == "oci:local:ods-lemonade-server"
    assert entries[0]["current"]["detail"]["containers"] == ["ods-llama-2",
                                                             "ods-llama-server"]


def test_local_oci_refuses_a_version_when_referrers_disagree():
    """Same repository, different resolved digests — the artifact has no one
    version, and a confidently wrong one is worse than none."""
    entries = collect.local_oci_entries(
        {"a": {"Image": "sha256:old", "Config": {"Image": "shared/img:v1"}},
         "b": {"Image": "sha256:new", "Config": {"Image": "shared/img:v2"}}},
        node="local")

    assert len(entries) == 1
    assert entries[0]["current"]["version"] is None
    assert entries[0]["current"]["verification"] == origins.UNKNOWN


def test_local_oci_skips_a_container_that_could_not_be_inspected():
    # The collector passes None for a container it failed to read; that is
    # mark_unavailable's job upstream, not a fabricated entry here.
    assert collect.local_oci_entries({"ods-hipfire": None}, node="local") == []


def test_local_oci_skips_a_container_with_no_image_reference():
    """Without `Config.Image` there is no repository, so there is no key.
    Refused rather than keyed on the container name as a guess (D8)."""
    assert collect.local_oci_entries(
        {"ods-hipfire": {"Image": "sha256:a", "Config": {}}}, node="local") == []


def test_local_oci_is_in_artifact_id_order():
    bodies = {"z": {"Image": "sha256:z", "Config": {"Image": "z-repo:latest"}},
              "a": {"Image": "sha256:a", "Config": {"Image": "a-repo:latest"}}}
    assert [e["artifact_id"] for e in collect.local_oci_entries(bodies, node="local")] == [
        "oci:local:a-repo", "oci:local:z-repo"]


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


# --- many profiles, one image (live-deploy defect, 2026-08-08) -------------
#
# Every test above passes ONE profile, which is why this went unnoticed until
# the live sweep: sparky runs seven profiles and five of them (heretic,
# laguna, mm27b, ornith, qwen35) share ONE image reference. The artifact id
# is keyed on the repository, so those five collapse to a single record --
# and the collector emitted one entry per PROFILE, so the store's last writer
# won. Only `laguna` matched the harvested catalog, and it sorts before
# mm27b/ornith/qwen35, so the digest it correctly resolved was overwritten
# with None three times. The live ledger read `version: null` for the very
# image that was running.

_COMPOSE_SHARED = """
services:
  vllm:
    image: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-27-v0.26.0
    container_name: aeon-vllm
"""


def test_spark_collapses_profiles_that_share_one_repository():
    texts = {p: _COMPOSE_SHARED for p in ("heretic", "laguna", "mm27b")}

    entries = collect.spark_oci_entries(texts, None, node="sparky")

    ids = [e["artifact_id"] for e in entries]
    assert ids == ["oci:sparky:aeon-7/aeon-vllm-ultimate"], (
        "one repository is one artifact — duplicate entries let whichever "
        "the caller stores last decide what the artifact's version is")


def test_spark_digest_survives_a_later_profile_sharing_the_repository():
    """The regression proper: `laguna` resolves the digest, and mm27b/ornith
    (which sort after it and share its image) must not erase it."""
    texts = {p: _COMPOSE_SHARED for p in ("heretic", "laguna", "mm27b", "ornith")}
    catalog = {"profile": "laguna", "image_id": "sha256:live",
               "harvested_ts": "2026-08-08T00:00:00Z", "engine": "vllm"}

    entries = collect.spark_oci_entries(texts, catalog, node="sparky")

    assert len(entries) == 1
    assert entries[0]["current"]["version"] == "sha256:live"
    assert entries[0]["current"]["verification"] == origins.EXACT


def test_spark_records_every_profile_that_references_the_image():
    """Same placement rule as local containers: which profiles run an image
    is a fact ABOUT the image, and the five that share aeon-vllm-ultimate
    must all be visible on its one record."""
    texts = {p: _COMPOSE_SHARED for p in ("laguna", "heretic", "mm27b")}

    entries = collect.spark_oci_entries(texts, None, node="sparky")

    assert len(entries) == 1
    assert entries[0]["current"]["detail"]["profiles"] == ["heretic", "laguna", "mm27b"]


def test_spark_keeps_distinct_repositories_apart():
    """The collapse must be by repository, not a blanket de-duplication."""
    texts = {"ds4": _COMPOSE_DS4, "laguna": _COMPOSE_SHARED}

    entries = collect.spark_oci_entries(texts, None, node="sparky")

    assert sorted(e["artifact_id"] for e in entries) == [
        "oci:sparky:aeon-7/aeon-vllm-ultimate", "oci:sparky:ds4-spark"]


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


# --- seed_watch (Task 9) ----------------------------------------------------
#
# ONLY a digest-pinned oci origin qualifies -- everything needed (registry,
# repository, pinned digest) is structured data already on the origin. A
# locally-built image with no registry, or any reference that is not
# digest-pinned, gets nothing: parsing `build` prose is the inference D8
# forbids (U4), and a floating tag with no digest has no exact point to
# compare a future digest against.

def test_seed_watch_derives_a_channel_source_from_a_digest_pinned_reference():
    entry = {"artifact_id": "oci:sparky:aeon-7/comfyui-aeon-spark", "kind": "oci",
             "origin": {"registry": "ghcr.io",
                        "repository": "aeon-7/comfyui-aeon-spark",
                        "reference": "ghcr.io/aeon-7/comfyui-aeon-spark:slim@sha256:7fda"}}
    sources = collect.seed_watch(entry)
    assert len(sources) == 1
    assert sources[0]["check"] == "oci_channel"
    assert sources[0]["reference"] == "slim"
    assert sources[0]["pinned"] == "sha256:7fda"
    assert sources[0]["derived"] is True


def test_seed_watch_returns_nothing_without_a_registry():
    # A locally built image has no upstream registry; its watch must be
    # DECLARED against its build inputs, never guessed from prose (U4).
    entry = {"artifact_id": "oci:local:ods-hipfire", "kind": "oci",
             "origin": {"repository": "ods-hipfire", "build": "built on autarch…"}}
    assert collect.seed_watch(entry) == []


def test_seed_watch_returns_nothing_for_an_artifact_with_no_origin():
    assert collect.seed_watch(
        {"artifact_id": "file:local:m.gguf", "kind": "file", "origin": None}) == []


def test_seed_watch_returns_nothing_for_a_non_oci_kind_even_with_a_matching_origin():
    """kind gates this before the reference is even parsed -- a file or git
    artifact has no registry/repository/reference vocabulary at all."""
    entry = {"artifact_id": "file:local:x", "kind": "file",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"}}
    assert collect.seed_watch(entry) == []


def test_seed_watch_returns_nothing_for_a_floating_reference_with_no_digest():
    """Risk 1: not digest-pinned. A moving tag alone has no exact point to
    compare a future reading against -- returning a half-built source here
    would silently watch nothing meaningful."""
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim"}}
    assert collect.seed_watch(entry) == []


def test_seed_watch_returns_nothing_for_a_malformed_reference():
    """Risk 1: malformed. Garbage in the reference field (hand-edited or a
    future origin shape) must degrade to 'nothing derivable', never raise
    and never half-populate a source."""
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "not-even-a-reference"}}
    assert collect.seed_watch(entry) == []


# --- merge_seeded_watch (Task 9) --------------------------------------------

def test_seeding_never_replaces_a_declared_source_on_the_same_id():
    """declared-over-derived, the precedence declared.json already uses --
    scoped to the ID the derived source would occupy, which is the only
    place the two can actually collide."""
    existing = [{"id": "channel", "check": "oci_channel", "registry": "ghcr.io",
                 "repository": "a/b", "reference": "stable",
                 "pinned": "sha256:hand", "order": None}]
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": existing}
    assert collect.merge_seeded_watch(entry) == existing


def test_a_declared_source_on_another_id_does_not_suppress_derivation():
    """A declaration is not a veto on everything else. The seed data
    declares `tags` on aeon-vllm; suppressing its derived `channel` digest
    check there silently disabled the very detector that declaration's own
    notes rely on."""
    declared = {"id": "upstream", "check": "git_compare",
                "remote": "https://github.com/a/b", "ref": "main",
                "pinned": "abc", "order": None}
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": [declared]}
    merged = collect.merge_seeded_watch(entry)
    assert [s["id"] for s in merged] == ["upstream", "channel"]
    assert merged[0] == declared


def test_a_derived_pin_is_refreshed_when_the_origin_is_re_declared():
    """THE FEATURE'S OWN CORE LOOP. Deck reports `available` -> operator
    pulls -> operator re-declares the origin (the documented remedy) ->
    the derived pin must follow, or check_channel keeps comparing against
    the OLD digest and reports a permanent false `available` that never
    self-heals."""
    stale = {"id": "channel", "check": "oci_channel", "derived": True,
             "label": "slim", "registry": "ghcr.io", "repository": "a/b",
             "reference": "slim", "pinned": "sha256:0ddd", "order": None}
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:beef"},
             "watch": [stale]}
    assert [s["pinned"] for s in collect.merge_seeded_watch(entry)] == ["sha256:beef"]


def test_a_derived_source_is_dropped_once_its_origin_stops_being_derivable():
    """The other side of re-deriving every pass: a derived source is only
    ever as good as the origin it came from. When the origin is withdrawn
    (PUT /origin with `null`), the source it produced goes with it rather
    than lingering as an unfalsifiable pin nobody declared."""
    derived = {"id": "channel", "check": "oci_channel", "derived": True,
               "label": "slim", "registry": "ghcr.io", "repository": "a/b",
               "reference": "slim", "pinned": "sha256:old", "order": None}
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci", "origin": None,
             "watch": [derived]}
    assert collect.merge_seeded_watch(entry) == []


def test_merging_tolerates_a_hand_edited_source_with_no_id():
    """`_stored_list` keeps a dict-shaped watch element even with no `id`
    (app/provenance.py:80-85, pinned by test_provenance.py's DECLARED-vs-
    DERIVED test), so the merge may never index `s["id"]` on stored data --
    a KeyError here escapes into Watcher._provenance_pass."""
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": [{"no": "id"}]}
    merged = collect.merge_seeded_watch(entry)
    assert merged[0] == {"no": "id"}
    assert [s.get("id") for s in merged] == [None, "channel"]


def test_merge_seeded_watch_seeds_a_genuinely_untouched_gap():
    """No watch, never checked (`update` absent/None) -- the ordinary gap
    seed_watch exists to fill on the very first pass that sees a derivable
    origin."""
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": [], "update": None}
    sources = collect.merge_seeded_watch(entry)
    assert len(sources) == 1
    assert sources[0]["id"] == "channel"


def test_merge_seeded_watch_does_not_duplicate_on_repeated_passes():
    """Risk 2, half A: feeding the previous pass's own output back in must
    be a fixed point, not a second 'channel' entry."""
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": []}
    first = collect.merge_seeded_watch(entry)
    entry["watch"] = first
    second = collect.merge_seeded_watch(entry)
    assert second == first
    assert len(second) == 1


def test_merge_seeded_watch_does_not_resurrect_a_deliberately_cleared_source():
    """Risk 2, half B -- the one the brief's own reference code gets wrong.

    `watch: []` is ALSO the blank-entry default, so on its own it cannot
    distinguish "nobody has looked at this yet" from "an operator watched
    it, then explicitly cleared it via PUT /watch {sources: []}". The
    second case must stick: re-deriving here would resurrect exactly what
    was deleted, forever, on every collector pass.

    `update` is not None precisely when this artifact's watch was checked
    at least once (app.provenance.record_update is the only writer, and it
    only runs for artifacts with a non-empty watch) -- the one bit of
    existing state that tells the two apart without a schema change.
    """
    entry = {"artifact_id": "oci:sparky:x", "kind": "oci",
             "origin": {"registry": "ghcr.io", "repository": "a/b",
                        "reference": "ghcr.io/a/b:slim@sha256:dd"},
             "watch": [],
             "update": {"status": "unavailable", "sources": [],
                        "checked_at": "2026-08-01T00:00:00+00:00"}}
    assert collect.merge_seeded_watch(entry) == []


def test_merge_seeded_watch_returns_nothing_when_neither_declared_nor_derivable():
    entry = {"artifact_id": "oci:local:ods-hipfire", "kind": "oci",
             "origin": {"repository": "ods-hipfire", "build": "built on autarch…"},
             "watch": []}
    assert collect.merge_seeded_watch(entry) == []
