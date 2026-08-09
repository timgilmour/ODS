"""Turn observations into provenance entries. Pure: values in, values out —
the caller (app.arbiter.Watcher._provenance_pass) owns every client call and
every failure.

Three sources, and the interesting one is sparky. The deck cannot inspect
that box — the node-agent deliberately holds no docker socket (see
app/engines/spark.py's module docstring) — so its engine provenance is
assembled from two reads that ALREADY EXIST:

* ``SparkClient.get_compose(profile)`` returns the raw compose YAML, whose
  ``image:`` line is the operator's declaration of what the profile runs.
* ``SparkClient.get_catalog()`` returns the swap helper's post-launch probe
  cache, including ``image_id`` — the resolved content id.

The catch, and the reason ``spark_oci_entries`` looks paranoid: the catalog
body carries ``engine`` and ``harvested_ts`` but has never carried the
profile, so attributing its digest to a profile was a guess. The node-agent
now stamps ``profile`` from the file's own name (catalog-<profile>.json).
This function uses the digest ONLY on an exact profile match — with an old
node-agent, a mismatched profile, or a missing stamp, the version is absent
rather than wrong. A confidently wrong version is worse than no version.
"""

import re

import yaml

from app import origins
from app.origins import file as file_origin
from app.origins import oci as oci_origin

_UNAVAILABLE_UNIT_STATE = "unavailable"


def local_oci_entries(inspect_bodies: dict, node: str) -> list[dict]:
    """Container name -> its inspect body (or None if the read failed).

    One GET per container gives both halves of the identity split: top-level
    ``Image`` is the content id, ``Config.Image`` the human reference.

    THE ARTIFACT IS THE IMAGE, NOT THE CONTAINER. Identity is the image
    REPOSITORY — the same key sparky uses — and the containers running it are
    recorded as placement facts, exactly as ``location`` is for weights
    (identity is the relpath; where it currently sits is not part of it).
    Keying per container would mean an operator declaring the same origin
    once per deployment of one image, and N copies of a hand-maintained fact
    drift.

    A ``None`` body yields NO entry — the caller marks that artifact
    unavailable, which preserves its last known version; inventing a blank
    entry here would erase it. A body with no ``Config.Image`` yields no
    entry either: without a reference there is no repository, and keying it
    on the container name instead would be a guess (D8).
    """
    grouped: dict[str, dict] = {}
    for name in sorted(inspect_bodies):
        body = inspect_bodies[name]
        if not body:
            continue
        identity = oci_origin.identity_from_inspect(body)
        reference = identity.get("label")
        if not reference:
            continue
        repository = oci_origin.origin_from_reference(reference)["repository"]
        group = grouped.setdefault(repository, {"containers": [],
                                                "versions": set(),
                                                "labels": set()})
        group["containers"].append(name)
        group["versions"].add(identity.get("version"))
        group["labels"].add(reference)

    entries = []
    for repository in sorted(grouped):
        group = grouped[repository]
        # Unanimous or absent. Two containers on one repository running
        # different digests means the artifact has no single version, and
        # this module's standing rule is that a confidently wrong version is
        # worse than no version.
        version = next(iter(group["versions"])) if len(group["versions"]) == 1 else None
        label = next(iter(group["labels"])) if len(group["labels"]) == 1 else None
        identity = {"version": version, "label": label,
                    "detail": {"containers": sorted(group["containers"])}}
        entries.append({
            "artifact_id": origins.build_artifact_id("oci", node, repository),
            "kind": "oci", "node": node, "role": "engine",
            "current": {**identity, "verification": oci_origin.grade(identity)},
        })
    return entries


def local_file_entries(units: list[dict], node: str) -> list[dict]:
    """Catalog units -> weights entries, keyed on ``relpath``.

    NOT on the unit id: app/catalog.py:45 builds ids as
    ``"<location>:<relpath>"`` and record_moved() (app/catalog.py:175-185)
    REWRITES the id on a move. With the any-drive mover deployed, keying on
    the unit id would orphan a model's origin and history exactly when
    somebody moves it. relpath survives; location is a placement fact.

    No new I/O: the catalog scan already stat()'d every one of these.
    """
    entries = []
    for unit in units:
        relpath = unit.get("relpath")
        if not relpath:
            continue
        identity = file_origin.identity_from_unit(unit)
        available = unit.get("state") != _UNAVAILABLE_UNIT_STATE
        entries.append({
            "artifact_id": origins.build_artifact_id("file", node, relpath),
            "kind": "file", "node": node, "role": "weights",
            "current": {**identity,
                        "verification": file_origin.grade(identity,
                                                          available=available)},
        })
    return entries


def spark_oci_entries(compose_texts: dict, catalog: dict | None,
                      node: str) -> list[dict]:
    """Profile -> its compose text, plus the node's newest harvested catalog.

    See the module docstring for why the digest is attributed only on an
    exact ``catalog["profile"]`` match.

    PROFILES ARE MANY, ARTIFACTS ARE FEW. The artifact id is keyed on the
    repository, and sparky runs several profiles off one image — five of
    seven share ``aeon-vllm-ultimate`` today. So this returns at most one
    entry PER REPOSITORY, not one per profile: emitting duplicates left the
    caller's last write deciding the artifact's version, and since only the
    catalog-matching profile carries a digest, any profile sharing its image
    and sorting after it erased that digest. Live, `laguna` resolved the
    running image and `mm27b`/`ornith`/`qwen35` each overwrote it with None.

    The merge rule is "a resolved digest wins", never "the last one wins" —
    a digest is a measurement, its absence is only the absence of one.
    """
    catalog_profile = (catalog or {}).get("profile")
    catalog_digest = (catalog or {}).get("image_id")

    merged: dict[str, dict] = {}
    profiles_by_artifact: dict[str, list[str]] = {}
    for profile in sorted(compose_texts):
        reference = _image_reference(compose_texts[profile])
        if not reference:
            continue
        origin = oci_origin.origin_from_reference(reference)
        digest = catalog_digest if (catalog_profile
                                    and catalog_profile == profile) else None
        identity = oci_origin.identity_from_compose(reference, digest)
        artifact_id = origins.build_artifact_id("oci", node,
                                                origin["repository"])
        # WHICH profiles run this image is a fact about the image — the same
        # placement rule local containers follow. Recorded for every profile,
        # including the ones whose entry loses the digest merge below.
        profiles_by_artifact.setdefault(artifact_id, []).append(profile)
        existing = merged.get(artifact_id)
        if existing is not None and existing["current"]["version"] is not None:
            # Already measured by the profile the catalog named. A profile
            # without a digest has nothing to add and must not subtract.
            continue
        merged[artifact_id] = {
            "artifact_id": artifact_id,
            "kind": "oci", "node": node, "role": "engine",
            "current": {**identity, "verification": oci_origin.grade(identity)},
        }
    for artifact_id, entry in merged.items():
        entry["current"]["detail"] = {
            **(entry["current"].get("detail") or {}),
            "profiles": sorted(profiles_by_artifact[artifact_id]),
        }
    return [merged[k] for k in sorted(merged)]


_REFERENCE_RE = re.compile(
    r"^(?:[^/]+/)?(?P<repo>.+?):(?P<tag>[^:@]+)@(?P<digest>sha256:[0-9a-f]+)$")


def seed_watch(entry: dict) -> list[dict]:
    """Watch sources derivable from the origin ALONE (Task 9).

    Only an oci origin that already names a registry, a repository and a
    digest-pinned reference qualifies -- everything needed is present as
    structured data. A locally built image gets nothing here: its upstream
    lives in build inputs recorded as English prose, and parsing that prose
    is the inference D8 forbids (U4). Those are declared by hand. A
    floating (non-digest-pinned) reference also gets nothing: there is no
    exact point to compare a future reading against, and returning a
    half-built source would silently watch nothing meaningful.
    """
    origin = entry.get("origin") or {}
    registry = origin.get("registry")
    repository = origin.get("repository")
    reference = origin.get("reference") or ""
    if entry.get("kind") != "oci" or not registry or not repository:
        return []

    match = _REFERENCE_RE.match(reference)
    if not match:
        return []
    return [{"id": "channel", "check": "oci_channel", "derived": True,
             "label": match["tag"], "registry": registry,
             "repository": repository, "reference": match["tag"],
             "pinned": match["digest"], "order": None}]


def merge_seeded_watch(entry: dict) -> list[dict]:
    """Declared-over-derived, PER ID: a hand-declared source wins the id it
    occupies, and every derivable source whose id nobody declared is added
    beside it.

    A DECLARATION IS NOT A VETO ON EVERYTHING ELSE. Returning early on the
    presence of any declared source made one declaration suppress every
    derived one -- live, the seed data declares `tags` on aeon-vllm, which
    silently disabled its derived `channel` digest check, the exact detector
    that declaration's own notes rely on.

    DERIVED SOURCES ARE RE-DERIVED, NEVER PRESERVED. A derived source is a
    function of the origin, so the stored copy is discarded and rebuilt every
    pass. That is what makes the feature's core loop close: deck reports
    `available` -> operator pulls -> operator re-declares the origin (the
    documented remedy) -> the pin follows. Keeping the stored copy instead
    left `check_channel` comparing against the OLD digest forever, a false
    `available` that could never self-heal. The same rule means a derived
    source disappears when its origin stops being derivable -- it was only
    ever as good as the origin behind it, and nobody declared it.

    RESURRECTION GUARD. `watch: []` is ALSO the blank-entry default, so on
    its own it cannot distinguish "nobody has looked at this yet" from "an
    operator watched it, then explicitly cleared it via
    PUT /watch {sources: []}". Re-deriving in the second case would put the
    same source back on every future collector pass -- the one thing a
    delete can never do here would be to stick.

    `entry["update"]` is not None exactly when this artifact's watch was
    checked at least once: app.provenance.record_update is its only writer,
    and app.update_check.run_pass only calls it for artifacts whose watch
    was non-empty at the time. So an empty watch alongside a non-None
    `update` can only mean "it was watched, then cleared" -- the one bit of
    existing state that tells the two empty-watch cases apart without a
    provenance.json schema change.
    """
    existing = entry.get("watch") or []
    if not existing and entry.get("update") is not None:
        return existing
    # `s.get("id")`, never `s["id"]`: `_stored_list` keeps a dict-shaped
    # watch element even with no id at all (app/provenance.py:80-85 -- the
    # DECLARED half of the gate's split), and a KeyError here escapes into
    # Watcher._provenance_pass.
    declared = [s for s in existing if not s.get("derived")]
    taken = {s.get("id") for s in declared}
    return declared + [s for s in seed_watch(entry) if s["id"] not in taken]


def _image_reference(text: str) -> str | None:
    """The single service's ``image:`` line, or None.

    Deliberately NOT app.compose_import.import_compose: that function's
    return shape is a fixed contract the settings adopt sweep and the reload
    route are built on (see its module docstring), and provenance needs one
    key it does not expose. A malformed file yields None rather than raising
    — one bad profile must not stop the sweep.
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    services = doc.get("services")
    if not isinstance(services, dict) or len(services) != 1:
        return None
    service = next(iter(services.values()))
    if not isinstance(service, dict):
        return None
    image = service.get("image")
    return image if isinstance(image, str) and image.strip() else None
