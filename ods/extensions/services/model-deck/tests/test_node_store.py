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


# ---- element-level gate: a hand-edited bad element must never take the
# deck down (guard-at-the-boundary — see the module docstring). Each test
# writes nodes.json directly so the malformed element never has to survive
# _validate() to get on disk. ----

def _write_nodes(tmp_path, entries):
    (tmp_path / "nodes.json").write_text(json.dumps(entries))


def test_load_drops_non_dict_elements(store, tmp_path):
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"})
    _write_nodes(tmp_path, [good, "not a dict", 42, None, ["nested", "list"]])
    assert [n["id"] for n in store.list()] == ["hera"]
    assert store.get("hera") is not None


def test_load_drops_dict_missing_id(store, tmp_path):
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"})
    bad = {"label": "No Id", "agent_kind": "node-agent", "address": "http://x:1"}
    _write_nodes(tmp_path, [good, bad])
    assert [n["id"] for n in store.list()] == ["hera"]


def test_load_drops_dict_with_non_string_label(store, tmp_path):
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"})
    bad = {"id": "zeus", "label": 123, "agent_kind": "node-agent", "address": "http://z:1"}
    _write_nodes(tmp_path, [good, bad])
    assert [n["id"] for n in store.list()] == ["hera"]
    assert store.get("zeus") is None


def test_load_drops_dict_with_bogus_agent_kind(store, tmp_path):
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"})
    bad = {"id": "zeus", "label": "Zeus Box", "agent_kind": "vampire", "address": "http://z:1"}
    _write_nodes(tmp_path, [good, bad])
    assert [n["id"] for n in store.list()] == ["hera"]


def test_get_never_raises_on_malformed_siblings(store, tmp_path):
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://hera:7720"})
    _write_nodes(tmp_path, [
        good,
        "not a dict",
        {"label": "No Id", "agent_kind": "node-agent"},
        {"id": "z", "label": 1, "agent_kind": "node-agent"},
        {"id": "zombie", "label": "Zombie", "agent_kind": "vampire"},
    ])
    assert store.get("hera")["id"] == "hera"
    assert store.get("missing") is None
    assert store.get("z") is None
    assert store.get("zombie") is None


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
    # _SEED_KW carries a credential, so the seeded entry lands at
    # control:"swap" (N1 T6 ruling below) — remove() now refuses a "swap"
    # node directly (T1's declared-operable guard), so demote first; the
    # demote/remove pair is just setup to reach "sparky gone" for the
    # mutation check that follows.
    store.update("sparky", {"control": "none"})
    store.remove("sparky")
    # Changed env after the seed must have NO effect — including resurrecting
    # a deliberately removed node. This is the mutation check the design
    # demands: env is consulted only while nodes.json does not exist.
    again = seed_if_missing(store, **{**_SEED_KW, "node_label": "changed",
                                      "spark_node_url": "http://elsewhere:7720"})
    assert again is False
    assert store.get("sparky") is None
    assert store.get("local")["label"] == "autarch"


def test_seed_with_a_credential_stamps_control_swap(store):
    """N1 T6 ruling: the env-var config WAS the old scheme's declaration of
    operability (main.py used to bind a client from exactly this
    condition), so the seed migrates that declaration into `control` too —
    not the unconditional "none" a bare `add()` defaults to."""
    seed_if_missing(store, **_SEED_KW)  # _SEED_KW's keys-json has a credential
    assert store.get("sparky")["control"] == "swap"


def test_seed_without_a_credential_stamps_control_none(store):
    seed_if_missing(store, **{**_SEED_KW, "spark_node_keys_json": "{}"})
    assert store.get("sparky")["control"] == "none"


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


def test_node_id_with_a_trailing_newline_is_refused(store):
    """Python's `$` also matches just before a trailing newline, so
    "sparky\\n" passed a pattern meant to be anchored [max-review c33]. The
    UI mirrors this pattern in JS, where `$` does NOT admit the newline — so
    the two vocabularies disagreed, and an id the browser refuses could still
    be created through the API. `\\Z` is the anchor that means what the
    comment always claimed.

    An id is a KEY: intent, settings scopes and oci:<id>: provenance all
    attach through this string, so a trailing newline propagates into every
    one of them.
    """
    with pytest.raises(ValueError):
        store.add({"id": "sparky\n", "label": "Sparky", "agent_kind": "node-agent",
                   "address": "http://x:7720"})


def test_credential_fingerprint_contract(store):
    """Pins the ""/absent -> None contract its docstring states [T8 review].
    None must be distinct from any real digest: the nodes router's unbound
    branch treats it as "no credential", and a fingerprint of "" would read
    as one."""
    store.add({"id": "hera", "label": "Hera", "agent_kind": "node-agent",
               "address": "http://hera:7720"}, credential="s3cret")
    store.add({"id": "nokey", "label": "No Key", "agent_kind": "node-agent",
               "address": "http://nokey:7720"})

    fp = store.credential_fingerprint("hera")
    assert fp is not None
    assert fp != "s3cret"                       # never the value itself
    assert fp == store.credential_fingerprint("hera")   # stable
    assert store.credential_fingerprint("nokey") is None
    assert store.credential_fingerprint("ghost") is None  # unknown node

    store.update("hera", {}, credential="rotated")
    assert store.credential_fingerprint("hera") != fp   # rotation is visible


def test_load_drops_a_legacy_id_the_write_side_would_refuse(tmp_path):
    """An id is a KEY; pre-`\\Z`-fix data like ``"sparky\\n"`` passed the
    old anchor, and a row the write-side `_validate()` refuses is otherwise
    a ghost no PATCH can ever touch (every patch re-validates the id -> 422
    forever). The load gate drops it like any other malformed element; the
    clean sibling row must survive untouched."""
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps([
        {"id": "sparky\n", "label": "ghost", "agent_kind": "node-agent",
         "address": "http://sparky:7720"},
        {"id": "local", "label": "autarch", "agent_kind": "local"},
    ]))
    store = NodeStore(path, tmp_path / "creds.json")
    ids = [n["id"] for n in store.list()]
    assert ids == ["local"]


# --- N1: the control field -------------------------------------------------

def _swap_spec(node_id="boxa", label="Box Alpha"):
    return {"id": node_id, "label": label, "agent_kind": "node-agent",
            "address": f"http://{node_id}:7720",
            "serving_address": f"http://{node_id}:8000",
            "control": "swap"}


def test_control_heals_to_none_at_load(store, tmp_path):
    # Hand-written pre-N1 entry: no control key at all.
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
         "address": "http://boxa:7720"}]))
    assert store.get("boxa")["control"] == "none"


def test_control_invalid_value_heals_to_none(store, tmp_path):
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
         "address": "http://boxa:7720", "control": "maybe"}]))
    assert store.get("boxa")["control"] == "none"


def test_control_unhashable_value_heals_to_none(store, tmp_path):
    # A hand-edited nodes.json with control: [] must heal, not crash --
    # `["swap"] in _CONTROLS` (a set) raises TypeError if unguarded, which
    # is exactly the failure the element gate exists to prevent.
    (tmp_path / "nodes.json").write_text(json.dumps([
        {"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
         "address": "http://boxa:7720", "control": ["swap"]}]))
    assert store.get("boxa")["control"] == "none"


def test_add_unhashable_control_raises_value_error(store):
    spec = _swap_spec()
    spec["control"] = ["swap"]
    with pytest.raises(ValueError):
        store.add(spec)


def test_add_defaults_control_none(store):
    entry = store.add({"id": "boxa", "label": "Box Alpha",
                       "agent_kind": "node-agent", "address": "http://boxa:7720"})
    assert entry["control"] == "none"


def test_add_swap_with_prereqs_ok(store):
    entry = store.add(_swap_spec(), credential="key-boxa")
    assert entry["control"] == "swap"


def test_swap_refused_without_prereqs_names_missing(store):
    spec = _swap_spec()
    del spec["serving_address"]
    with pytest.raises(ValueError) as exc:
        store.add(spec)          # no credential either
    assert "serving_address" in str(exc.value)
    assert "credential" in str(exc.value)
    assert "address" not in str(exc.value).replace("serving_address", "")


def test_patch_to_swap_uses_stored_credential(store):
    spec = _swap_spec()
    spec["control"] = "none"
    store.add(spec, credential="key-boxa")
    entry = store.update("boxa", {"control": "swap"})
    assert entry["control"] == "swap"


def test_patch_to_swap_refused_without_credential(store):
    spec = _swap_spec()
    spec["control"] = "none"
    store.add(spec)              # no credential stored
    with pytest.raises(ValueError, match="credential"):
        store.update("boxa", {"control": "swap"})


def test_local_refuses_swap(store):
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    with pytest.raises(ValueError, match="local"):
        store.update("local", {"control": "swap"})


def test_remove_swap_node_refused(store):
    store.add(_swap_spec(), credential="key-boxa")
    with pytest.raises(GuardError, match="control"):
        store.remove("boxa")
    # Explicit two-step: demote first, then remove works.
    store.update("boxa", {"control": "none"})
    store.remove("boxa")
    assert store.get("boxa") is None


def test_demote_to_none_always_allowed(store):
    store.add(_swap_spec(), credential="key-boxa")
    entry = store.update("boxa", {"control": "none"})
    assert entry["control"] == "none"
    # Credential survives the demotion — demote-then-clear is two steps,
    # and demote alone must not destroy anything.
    assert store.credential_set("boxa")


# --- N1: migration stamp ----------------------------------------------------

def _legacy_file(tmp_path, entries):
    (tmp_path / "nodes.json").write_text(json.dumps(entries))


def test_stamp_sets_swap_on_matching_entry_with_prereqs(store, tmp_path):
    _legacy_file(tmp_path, [
        {"id": "local", "label": "This Box", "agent_kind": "local"},
        {"id": "sparky", "label": "sparky", "agent_kind": "node-agent",
         "address": "http://sparky:7720", "serving_address": "http://sparky:8000"},
    ])
    (tmp_path / "node_credentials.json").write_text(json.dumps({"sparky": "k"}))
    assert store.stamp_missing_control("sparky") is True
    assert store.get("sparky")["control"] == "swap"
    assert store.get("local")["control"] == "none"


def test_stamp_without_prereqs_stays_none(store, tmp_path):
    # No credential: today's main.py would not have bound a client either.
    _legacy_file(tmp_path, [
        {"id": "sparky", "label": "sparky", "agent_kind": "node-agent",
         "address": "http://sparky:7720", "serving_address": "http://sparky:8000"},
    ])
    store.stamp_missing_control("sparky")
    assert store.get("sparky")["control"] == "none"


def test_stamp_runs_exactly_once(store, tmp_path):
    """Mutation check (design §10): after the stamp, a deliberate demotion
    survives every later boot — the migration keys on the ABSENCE of the
    control key on disk, which exists only in pre-N1 files."""
    _legacy_file(tmp_path, [
        {"id": "sparky", "label": "sparky", "agent_kind": "node-agent",
         "address": "http://sparky:7720", "serving_address": "http://sparky:8000"},
    ])
    (tmp_path / "node_credentials.json").write_text(json.dumps({"sparky": "k"}))
    assert store.stamp_missing_control("sparky") is True
    store.update("sparky", {"control": "none"})           # operator demotes
    assert store.stamp_missing_control("sparky") is False  # never re-stamps
    assert store.get("sparky")["control"] == "none"


def test_stamp_ignores_non_matching_ids(store, tmp_path):
    _legacy_file(tmp_path, [
        {"id": "boxa", "label": "Box Alpha", "agent_kind": "node-agent",
         "address": "http://boxa:7720", "serving_address": "http://boxa:8000"},
    ])
    (tmp_path / "node_credentials.json").write_text(json.dumps({"boxa": "k"}))
    store.stamp_missing_control("sparky")
    assert store.get("boxa")["control"] == "none"
