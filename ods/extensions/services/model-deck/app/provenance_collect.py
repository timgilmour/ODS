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

import yaml

from app import origins
from app.origins import file as file_origin
from app.origins import oci as oci_origin

_UNAVAILABLE_UNIT_STATE = "unavailable"


def local_oci_entries(inspect_bodies: dict, node: str) -> list[dict]:
    """Container name -> its inspect body (or None if the read failed).

    One GET per container gives both halves of the identity split: top-level
    ``Image`` is the content id, ``Config.Image`` the human reference. A
    ``None`` body yields NO entry — the caller marks that artifact
    unavailable, which preserves its last known version; inventing a blank
    entry here would erase it.
    """
    entries = []
    for name in sorted(inspect_bodies):
        body = inspect_bodies[name]
        if not body:
            continue
        identity = oci_origin.identity_from_inspect(body)
        entries.append({
            "artifact_id": origins.build_artifact_id("oci", node, name),
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
    """
    entries = []
    catalog_profile = (catalog or {}).get("profile")
    catalog_digest = (catalog or {}).get("image_id")

    for profile in sorted(compose_texts):
        reference = _image_reference(compose_texts[profile])
        if not reference:
            continue
        origin = oci_origin.origin_from_reference(reference)
        digest = catalog_digest if (catalog_profile
                                    and catalog_profile == profile) else None
        identity = oci_origin.identity_from_compose(reference, digest)
        entries.append({
            "artifact_id": origins.build_artifact_id("oci", node,
                                                     origin["repository"]),
            "kind": "oci", "node": node, "role": "engine",
            "current": {**identity, "verification": oci_origin.grade(identity)},
        })
    return entries


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
