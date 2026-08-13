"""Node registry against the live box (read-only; SAFE tier).

Tolerant of sparky being powered off: these assert registry SHAPE and the
seed's key-vocabulary preservation, never that the box is up. The test-
connection check accepts ok=False (that is a real answer when the node is
off) — it asserts the route works and never echoes a credential.
"""
import pytest

pytestmark = pytest.mark.safe

# The live install's registry id for the swap node — data (nodes.json), not
# coupling.
NODE = "sparky"

_STATUSES = {None, "online", "offline", "error", "unconfigured"}


def test_registry_lists_local_and_sparky(deck):
    body = deck.get("/api/nodes").json()
    ids = {n["id"] for n in body["nodes"]}
    assert {"local", "sparky"} <= ids, (
        "seed did not migrate the env config — check nodes.json and the "
        "MODEL_DECK_SPARK_* vars the container was started with")


def test_no_credential_field_anywhere_on_the_wire(deck):
    for path in ("/api/nodes", "/api/state"):
        body = deck.get(path).json()
        nodes = body["nodes"] if path == "/api/nodes" else body.get("nodes", [])
        for node in nodes:
            assert "credential" not in node
            assert isinstance(node.get("credential_set"), bool)


def test_sparky_seed_preserved_the_key_vocabulary(deck):
    # What this actually proves: the sparky key vocabulary (lifecycle's
    # sparky/ keys, provenance's oci:sparky:* artifacts) is still readable
    # and coherent post-deploy. It does NOT prove this boot's seed used the
    # right id — intent.json and provenance.json are additive stores with
    # months of prior sparky-keyed history, and the deck now keys everything
    # on registry ids (nodes.json), not a hardcoded constant, so a wrong id
    # written by seed_if_missing would not turn this test red.
    # test_registry_lists_local_and_sparky (a fresh GET /api/nodes read) is
    # the one that proves THIS seed used the right id; the two together are
    # the real coverage.
    state = deck.get("/api/state").json()
    assert any(k.startswith("sparky/") for k in state["lifecycle"]), (
        "lifecycle lost its sparky/ keys after the registry migration")
    prov = deck.get("/api/provenance").json()
    assert any(a["artifact_id"].startswith("oci:sparky:")
               for a in prov["artifacts"]), (
        "provenance lost its oci:sparky:* artifacts after the migration")


def test_state_nodes_statuses_stay_in_vocabulary(deck):
    for node in deck.get("/api/state").json()["nodes"]:
        assert node["status"] in _STATUSES


def test_local_node_reports_online_with_a_label(deck):
    nodes = {n["id"]: n for n in deck.get("/api/state").json()["nodes"]}
    assert nodes["local"]["status"] == "online"
    assert nodes["local"]["label"]


def test_test_connection_route_answers_without_echoing(deck):
    resp = deck.post("/api/nodes/test", json={"node_id": "sparky"})
    assert resp.status_code == 200
    body = resp.json()
    assert "ok" in body
    if not body["ok"]:
        assert body.get("error"), "a failed test must say why"
    assert "credential" not in resp.text


def test_spark_alias_removed(deck):
    """/api/spark/* (the one-deploy-cycle alias, design §6) is gone; a 404
    from the live deck proves no forwarder was resurrected by a deploy of
    an older image or a stale bundle."""
    assert deck.get("/api/spark/status").status_code == 404
