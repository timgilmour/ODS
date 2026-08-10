"""tests/test_node_store.py — nodes.json + node_credentials.json.

Labels are deliberately ≠ ids throughout ([[defaults-that-hide-bugs]]): a
fixture whose label equals its id cannot catch a label used as a key.
"""
import json
import os
import stat

import pytest

from app.engines import GuardError
from app.node_store import NodeStore


@pytest.fixture
def store(tmp_path):
    return NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")


def test_add_and_list_roundtrip(store):
    entry = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"}, credential="s3cret")
    assert entry["id"] == "hera"
    assert entry["added_ts"]
    assert [n["id"] for n in store.list()] == ["hera"]
    assert store.credential_set("hera") is True
    assert store.credential_for("hera") == "s3cret"


def test_credential_never_lands_in_nodes_json(store, tmp_path):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"}, credential="s3cret")
    assert "s3cret" not in (tmp_path / "nodes.json").read_text()
    assert "s3cret" in (tmp_path / "node_credentials.json").read_text()


def test_credentials_file_is_mode_0600(store, tmp_path):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"}, credential="s3cret")
    mode = stat.S_IMODE(os.stat(tmp_path / "node_credentials.json").st_mode)
    assert mode == 0o600


def test_add_duplicate_id_is_a_guard_refusal(store):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"})
    with pytest.raises(GuardError):
        store.add({"id": "hera", "label": "Other Label", "agent_kind": "node-agent",
                   "address": "http://other:7720"})


@pytest.mark.parametrize("bad_id", ["Hera", "hera box", "-hera", "", "hera/0"])
def test_id_must_be_a_lowercase_slug(store, bad_id):
    with pytest.raises(ValueError):
        store.add({"id": bad_id, "label": "Hera Box", "agent_kind": "node-agent",
                   "address": "http://hera:7720"})


def test_node_agent_kind_requires_an_address(store):
    with pytest.raises(ValueError):
        store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent"})


def test_local_kind_cannot_be_added_via_add(store):
    # local is seeded (Task 2), never operator-created.
    with pytest.raises(ValueError):
        store.add({"id": "box2", "label": "Second Box", "agent_kind": "local"})


def test_update_patches_only_patchable_fields(store):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"})
    out = store.update("hera", {"label": "Renamed Box",
                                "serving_address": "http://hera:8000"})
    assert out["label"] == "Renamed Box"
    assert out["serving_address"] == "http://hera:8000"
    with pytest.raises(ValueError):
        store.update("hera", {"id": "zeus"})          # id immutable
    with pytest.raises(ValueError):
        store.update("hera", {"agent_kind": "local"})  # kind immutable
    with pytest.raises(ValueError):
        store.update("zeus", {"label": "X"})           # unknown node


def test_update_can_replace_the_credential(store):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"}, credential="old")
    store.update("hera", {}, credential="new")
    assert store.credential_for("hera") == "new"


def test_remove_deletes_entry_and_credential_only(store, tmp_path):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"}, credential="s3cret")
    store.remove("hera")
    assert store.list() == []
    assert store.credential_set("hera") is False
    assert "s3cret" not in (tmp_path / "node_credentials.json").read_text()


def test_local_is_undeletable(store):
    # Seeded shape (Task 2 writes it); remove must refuse regardless of origin.
    store._save([{"id": "local", "label": "This Box", "agent_kind": "local",
                  "added_ts": "2026-08-10T00:00:00+00:00"}])
    with pytest.raises(GuardError):
        store.remove("local")


def test_corrupt_files_self_heal_to_empty(store, tmp_path):
    (tmp_path / "nodes.json").write_text("{not json")
    (tmp_path / "node_credentials.json").write_text("{not json")
    assert store.list() == []
    assert store.credential_for("hera") == ""


# ---- seed_if_missing: one-time env→registry migration ----

from app.node_store import seed_if_missing

_SEED_KW = dict(node_label="autarch", spark_id="sparky",
                spark_node_url="http://192.168.1.7:7720",
                spark_serving_url="http://192.168.1.7:8000",
                spark_node_name="sparky",
                spark_node_keys_json='{"sparky": "spark-key"}')


def test_seed_creates_local_and_sparky(store):
    assert seed_if_missing(store, **_SEED_KW) is True
    local = store.get("local")
    assert local["label"] == "autarch" and local["agent_kind"] == "local"
    sparky = store.get("sparky")
    assert sparky["agent_kind"] == "node-agent"
    assert sparky["address"] == "http://192.168.1.7:7720"
    assert sparky["serving_address"] == "http://192.168.1.7:8000"
    assert store.credential_for("sparky") == "spark-key"


def test_seed_runs_exactly_once(store):
    seed_if_missing(store, **_SEED_KW)
    store.remove("sparky")
    # Changed env after the seed must have NO effect — including resurrecting
    # a deliberately removed node. This is the mutation check the design
    # demands: env is consulted only while nodes.json does not exist.
    again = seed_if_missing(store, **{**_SEED_KW, "node_label": "changed",
                                      "spark_node_url": "http://elsewhere:7720"})
    assert again is False
    assert store.get("sparky") is None
    assert store.get("local")["label"] == "autarch"


def test_seed_without_spark_env_creates_local_only(store):
    seed_if_missing(store, **{**_SEED_KW, "spark_node_url": "",
                              "spark_serving_url": ""})
    assert store.get("local") is not None
    assert store.get("sparky") is None


def test_seed_with_missing_key_still_creates_the_entry(store):
    # Entry without credential: the node exists, credential_set is False,
    # and the observer will report it `unconfigured` rather than probing.
    seed_if_missing(store, **{**_SEED_KW, "spark_node_keys_json": "{}"})
    assert store.get("sparky") is not None
    assert store.credential_set("sparky") is False


def test_seed_tolerates_malformed_keys_json(store):
    seed_if_missing(store, **{**_SEED_KW, "spark_node_keys_json": "{broken"})
    assert store.get("sparky") is not None
    assert store.credential_set("sparky") is False
