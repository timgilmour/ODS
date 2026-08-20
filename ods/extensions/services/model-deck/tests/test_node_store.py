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
from app.settings import Settings


@pytest.fixture
def store(tmp_path):
    return NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")


def _settings(monkeypatch=None):
    """Build a Settings instance with non-default overrides for E1 engine seed tests.

    Ensures test uses values away from production defaults to catch accidental
    hardcoded defaults ([[defaults-that-hide-bugs]]). If monkeypatch is provided,
    sets env vars; otherwise creates Settings with direct kwargs."""
    if monkeypatch is None:
        # Called without pytest's monkeypatch fixture; use direct Settings kwargs
        return Settings(
            lemonade_gpu_index=5,
            hipfire_gpu_index=7,
            lemonade_url="http://test-lemonade:9999",
            lemonade_metrics_url="http://test-lemonade:9998/metrics",
            lemonade_container="test-ods-llama-server",
            comfyui_url="http://test-comfyui:9188",
            comfyui_container="test-ods-comfyui",
            hipfire_container="test-ods-hipfire",
        )
    # Monkeypatch provided; set env vars and instantiate
    monkeypatch.setenv("MODEL_DECK_LEMONADE_GPU_INDEX", "5")
    monkeypatch.setenv("MODEL_DECK_HIPFIRE_GPU_INDEX", "7")
    monkeypatch.setenv("MODEL_DECK_LEMONADE_URL", "http://test-lemonade:9999")
    monkeypatch.setenv("MODEL_DECK_LEMONADE_METRICS_URL", "http://test-lemonade:9998/metrics")
    monkeypatch.setenv("MODEL_DECK_LEMONADE_CONTAINER", "test-ods-llama-server")
    monkeypatch.setenv("MODEL_DECK_COMFYUI_URL", "http://test-comfyui:9188")
    monkeypatch.setenv("MODEL_DECK_COMFYUI_CONTAINER", "test-ods-comfyui")
    monkeypatch.setenv("MODEL_DECK_HIPFIRE_CONTAINER", "test-ods-hipfire")
    return Settings()


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


# --- N1 T8 review: a malformed address must never reach a client factory ---
# (httpx.Client(base_url=...) / urlparse raise on "http://h:port" or
# "http://[::1", and the old isinstance(str)-only gate let both through to
# app.node_clients.NodeClients.client_for, where the raise escaped into
# Watcher.tick()'s supervisor catch every tick. Refused at the write
# boundary instead -- [[guard-at-the-boundary]].)


def test_add_unparseable_port_address_is_refused(store):
    with pytest.raises(ValueError) as exc:
        store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                   "address": "http://h:port"})
    assert "address" in str(exc.value)


def test_update_malformed_ipv6_serving_address_is_refused(store):
    store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
               "address": "http://hera:7720"})
    with pytest.raises(ValueError) as exc:
        store.update("hera", {"serving_address": "http://[::1"})
    assert "serving_address" in str(exc.value)


def test_add_a_normal_address_still_passes(store):
    entry = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                       "address": "http://boxa:7720",
                       "serving_address": "http://boxa:8000"})
    assert entry["address"] == "http://boxa:7720"
    assert entry["serving_address"] == "http://boxa:8000"


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


# --- E1 engines declaration tests -----

def test_local_entry_accepts_validated_engines(store):
    store.add({"id": "local", "label": "Box L", "agent_kind": "local",
               "engines": [{"resource": "gguf-a", "kind": "lemonade",
                            "connection": {"url": "http://gguf-a:8080",
                                           "metrics_url": "http://gguf-a:8001/metrics",
                                           "container": "ods-gguf-a"},
                            "gpu_index": 3,
                            "container_consent": True,
                            "policy_defaults": {"priority": 10, "pinned": False,
                                                "idle_ttl": 60}}]})
    assert store.get("local")["engines"][0]["resource"] == "gguf-a"


def test_invalid_engines_refused_at_add(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.add({"id": "local", "label": "Box L", "agent_kind": "local",
                   "engines": [{"resource": "x", "kind": "nope",
                                "connection": {}, "gpu_index": 0,
                                "policy_defaults": {"priority": 1,
                                                    "pinned": False,
                                                    "idle_ttl": 0}}]})


def test_load_heals_local_entry_with_invalid_engines(store, tmp_path):
    """Local entry with schema-invalid engines heals to [], entry preserved."""
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                      "address": "http://hera:7720"})
    _write_nodes(tmp_path, [good,
        {"id": "local", "label": "Local Box", "agent_kind": "local",
         "engines": [{"resource": "bad", "kind": "invalid"}]}])
    loaded = store.get("local")
    assert loaded is not None
    assert loaded["id"] == "local"
    assert loaded["engines"] == []


def test_load_heals_local_entry_with_a_slot0_resource(store, tmp_path):
    """A hand-written entry declaring an otherwise-VALID engine whose
    `resource` is literally "slot0" heals to [] exactly like any other
    schema-invalid engines list (test_load_heals_local_entry_with_invalid_engines
    above) — `_heal_engines` needs no separate slot0 guard of its own, it
    already routes every list through `validate_engines`
    (app/engine_kinds.py), so that module's reservation of the name lands
    here automatically. This is the inheritance the single gate depends
    on: a second guard planted here instead would be exactly the
    duplicated-site pattern [[guard-at-the-boundary]] warns against."""
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                      "address": "http://hera:7720"})
    _write_nodes(tmp_path, [good,
        {"id": "local", "label": "Local Box", "agent_kind": "local",
         "engines": [{"resource": "slot0", "kind": "lemonade",
                      "connection": {"url": "http://gguf-a:8080",
                                    "metrics_url": "http://gguf-a:8001/metrics",
                                    "container": "ods-gguf-a"},
                      "gpu_index": 3,
                      "policy_defaults": {"priority": 10, "pinned": False,
                                          "idle_ttl": 60}}]}])
    loaded = store.get("local")
    assert loaded is not None
    assert loaded["id"] == "local"
    assert loaded["engines"] == []


def test_heal_engines_strips_from_unknown_agent_kind():
    """`_heal_engines` still strips engines from any OTHER/unknown
    agent_kind (E1 Task 5: only local/node-agent may carry the key now).
    `NodeStore._load()` can never actually reach this branch today —
    `_well_formed` (app/node_store.py) drops any entry whose agent_kind
    isn't already in {"local", "node-agent"} BEFORE `_heal_engines` ever
    runs on it (see test_load_drops_dict_with_bogus_agent_kind above) — so
    this pins the function's own documented contract directly, the only
    way to actually exercise the branch."""
    from app.node_store import _heal_engines
    entry = {"id": "ghost", "label": "Ghost", "agent_kind": "vampire",
             "engines": [{"resource": "x"}]}
    healed = _heal_engines(entry)
    assert "engines" not in healed


# --- E1 Task 5: engines[] on a node-agent entry (remote_capable gate) ------
# Fixture rule ([[defaults-that-hide-bugs]]): node id "nimbus" (not the
# live-seeded "sparky"), resource "gguf-r" (not "omni") — nothing here may
# pass by coinciding with a live seed.

def _remote_spec(node_id="nimbus", label="Nimbus Box"):
    return {"id": node_id, "label": label, "agent_kind": "node-agent",
            "address": f"http://{node_id}:7720"}


_REMOTE_ENGINE = {
    "resource": "gguf-r", "kind": "lemonade",
    "connection": {"url": "http://gguf-r:8080",
                   "metrics_url": "http://gguf-r:8001/metrics",
                   "container": "ods-gguf-r"},
    "gpu_index": 4,
    "container_consent": True,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 30},
}


def test_node_agent_engines_accepted_with_prereqs(store):
    """Happy path: engines[] on a node-agent entry is no longer an
    outright refusal. An EMPTY list is vacuously valid for ANY kind set —
    every kind named in THIS module is a non-remote-capable one, so a
    populated list here always fails the kind-not-remote_capable gate below
    — and with address + credential both present it is accepted
    (address+credential are NOT actually required for
    an empty list — see test_node_agent_engines_empty_list_accepted_without_credential
    below — but this pins the case where they happen to be present too)."""
    entry = store.add({**_remote_spec(), "engines": []}, credential="key-nimbus")
    assert entry["engines"] == []
    assert store.get("nimbus")["engines"] == []


def test_node_agent_engines_empty_list_accepted_without_credential(store):
    """Fix round 1 (Finding 1): `engines: []` declares nothing operable,
    so it must NOT require a credential the entry doesn't have —
    `_require_engine_prereqs` is gated on a NON-EMPTY list (truthiness),
    matching `validate_engines([], remote=True)`'s own vacuous pass."""
    entry = store.add({**_remote_spec(), "engines": []})  # no credential
    assert entry["engines"] == []


def test_node_agent_with_empty_engines_and_no_credential_is_not_bricked(store, tmp_path):
    """Regression (fix round 1, Finding 1 — reviewer-reproduced live bug):
    before the fix, `_require_engine_prereqs` fired on the mere PRESENCE
    of the `engines` key, so a node-agent entry with `engines: []` and no
    stored credential (e.g. a hand-written nodes.json with a lost
    credential sidecar) became unpatchable in EVERY field — a label-only
    PATCH 422'd with "requires credential". Combined with `_heal_engines`
    now PRESERVING (not stripping) the key on node-agent entries, this was
    reachable through ordinary use, not just a contrived hand-edit. Pins
    the property the old (now-superseded)
    test_load_strips_engines_from_non_local_entries used to pin under the
    old "strip" behavior: the entry stays patchable."""
    _write_nodes(tmp_path, [{**_remote_spec(), "engines": []}])
    loaded = store.get("nimbus")
    assert loaded is not None
    assert loaded["engines"] == []          # heal preserved it, didn't strip
    assert store.credential_set("nimbus") is False
    result = store.update("nimbus", {"label": "Renamed"})
    assert result["label"] == "Renamed"


def test_require_engine_prereqs_names_missing_fields(store):
    """Direct unit pin of `_require_engine_prereqs`'s own checklist
    contract. Exercised directly (not through add()/update()) because it
    is UNREACHABLE through the public API for the kinds THIS module
    declares: the callers gate it on a NON-EMPTY engines list (see the
    not-bricked regression above), and any non-empty LEMONADE-kind list on
    a node-agent spec fails `_validate`'s kind-not-remote_capable check
    FIRST (see the ordering test below). Task 7's remote-capable kind made
    it reachable in general; it is still exercised directly here because
    that remains the only way to reach it from this module's fixtures."""
    with pytest.raises(ValueError, match="credential") as exc:
        store._require_engine_prereqs({"address": "http://nimbus:7720"},
                                       credential_present=False)
    assert "engines on a node-agent entry" in str(exc.value)


def test_node_agent_engines_kind_not_remote_capable_refused_naming_kind(store):
    """Every declared kind must be `remote_capable`; the E1 triple are not,
    so a lemonade-kind list is refused BY NAME even with every other
    prerequisite (address + credential) satisfied. Still exactly the live
    behavior after Task 7 — that task made ONE kind remote-capable, it did
    not make the gate permissive."""
    with pytest.raises(ValueError, match="lemonade"):
        store.add({**_remote_spec(), "engines": [_REMOTE_ENGINE]},
                  credential="key-nimbus")


def test_node_agent_engines_kind_check_precedes_credential_check(store):
    """Ordering proof (fix round 1): a node-agent spec with BOTH a
    non-remote-capable kind AND no stored credential fails on the KIND
    first — `_validate` (called unconditionally at the top of add()) runs
    before `_require_engine_prereqs` ever would. The error names
    "lemonade", never "credential"."""
    with pytest.raises(ValueError, match="lemonade") as exc:
        store.add({**_remote_spec(), "engines": [_REMOTE_ENGINE]})  # no credential
    assert "credential" not in str(exc.value)


def test_node_agent_engines_gate_also_applies_on_update(store):
    """The same two checks run on update(), not just add() — mirrors the
    control:"swap" prereq check's own add/update parity."""
    store.add(_remote_spec(), credential="key-nimbus")  # no engines yet
    with pytest.raises(ValueError, match="lemonade"):
        store.update("nimbus", {"engines": [_REMOTE_ENGINE]})


def test_heal_preserves_empty_engines_on_node_agent_entry(store, tmp_path):
    """The core gate-opening behavior at the LOAD side: `_heal_engines`
    stops stripping engines from node-agent entries — a hand-written (or
    previously-written) node-agent row with a valid (empty) engines list
    survives `_load()` with the key intact, across a fresh `NodeStore`
    instance (proves it's the file, not in-memory state, that's healed)."""
    _write_nodes(tmp_path, [{**_remote_spec(), "engines": []}])
    reloaded = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")
    loaded = reloaded.get("nimbus")
    assert loaded is not None
    assert loaded["engines"] == []


def test_heal_node_agent_entry_with_non_remote_capable_kind_heals_to_empty(store, tmp_path):
    """A hand-edited node-agent entry with a POPULATED engines list of a
    NON-remote-capable kind — never legally reachable through
    add()/update() — heals to [] rather than being dropped or
    crashing the load, same posture as a local entry's schema-invalid
    engines (test_load_heals_local_entry_with_invalid_engines above)."""
    good = store.add({"id": "hera", "label": "Hera Box", "agent_kind": "node-agent",
                      "address": "http://hera:7720"})
    _write_nodes(tmp_path, [good, {**_remote_spec(), "engines": [_REMOTE_ENGINE]}])
    loaded = store.get("nimbus")
    assert loaded is not None
    assert loaded["id"] == "nimbus"
    assert loaded["engines"] == []
    # The sibling entry is untouched — one bad row heals in place, it
    # doesn't take the rest of the file down with it.
    assert store.get("hera")["id"] == "hera"


def test_local_engines_behavior_is_unaffected_by_the_node_agent_gate(store):
    """`local` byte-identical check: the pre-existing local-entry happy
    path (test_local_entry_accepts_validated_engines above) and this
    negative one both still hold exactly as before — local never goes
    through `_require_engine_prereqs` (that only fires for agent_kind ==
    "node-agent") and its own kind gate stays `remote=False`, so all three
    of today's (non-remote-capable) kinds remain valid there."""
    entry = store.add({"id": "local", "label": "Box L", "agent_kind": "local",
                       "engines": [_REMOTE_ENGINE]})
    assert entry["engines"][0]["kind"] == "lemonade"


# ---- E1 T2: seed_engines_if_missing ----

from app.node_store import seed_engines_if_missing


def _legacy_intent(tmp_path):
    (tmp_path / "intent.json").write_text(
        '{"local/lemonade": {"state": "loaded", "model": "m.gguf",'
        ' "engine": "lemonade"}}')


def test_seed_engines_stamps_triple_when_legacy_records_exist(store, tmp_path):
    store.add({"id": "local", "label": "Box L", "agent_kind": "local"})
    _legacy_intent(tmp_path)
    settings = _settings()
    changed = seed_engines_if_missing(store, settings, tmp_path)
    assert changed is True
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert set(engines) == {"lemonade", "comfyui", "hipfire"}
    # Comprehensive assertions: values flow from Settings, not hardcoded copies
    # (away from defaults to catch accidental hardcoding [[defaults-that-hide-bugs]]).
    assert engines["lemonade"]["connection"]["url"] == settings.lemonade_url
    assert engines["lemonade"]["connection"]["metrics_url"] == settings.lemonade_metrics_url
    assert engines["lemonade"]["connection"]["container"] == settings.lemonade_container
    assert engines["lemonade"]["gpu_index"] == settings.lemonade_gpu_index
    assert engines["comfyui"]["connection"]["url"] == settings.comfyui_url
    # Finding 1b (whole-branch review): the seed follows the
    # hipfire_container/lemonade_container SEED-ONLY settings pattern for
    # comfyui too, now that its schema accepts the field.
    assert engines["comfyui"]["connection"]["container"] == settings.comfyui_container
    # Most easily-inverted mapping: comfyui uses lemonade_gpu_index, not hipfire_gpu_index
    assert engines["comfyui"]["gpu_index"] == settings.lemonade_gpu_index
    assert engines["hipfire"]["connection"]["container"] == settings.hipfire_container
    assert engines["hipfire"]["gpu_index"] == settings.hipfire_gpu_index


def test_seed_engines_empty_without_presence_proof(store, tmp_path):
    """Coexistence (spec §6): env DEFAULTS naming llama-server are not
    proof llama-server exists. Fresh data dir -> empty declaration."""
    store.add({"id": "local", "label": "Box L", "agent_kind": "local"})
    changed = seed_engines_if_missing(store, _settings(), tmp_path)
    assert changed is True
    assert store.get("local")["engines"] == []


def test_seed_engines_runs_once(store, tmp_path):
    store.add({"id": "local", "label": "Box L", "agent_kind": "local"})
    _legacy_intent(tmp_path)
    seed_engines_if_missing(store, _settings(), tmp_path)
    store.update("local", {"engines": []})     # operator emptied it
    assert seed_engines_if_missing(store, _settings(), tmp_path) is False
    assert store.get("local")["engines"] == []  # seed did NOT resurrect


def test_seed_engines_stamps_container_consent_true_on_the_triple(store, tmp_path):
    """The seeded triple IS today's env-default allowlist — consent true
    preserves live behaviour across the allowlist->flag migration."""
    store.add({"id": "local", "label": "Box L", "agent_kind": "local"})
    _legacy_intent(tmp_path)
    settings = _settings()
    seed_engines_if_missing(store, settings, tmp_path)
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert all(engines[r]["container_consent"] is True
               for r in ("hipfire", "lemonade", "comfyui"))


def test_stamp_missing_container_consent_defaults_false_except_triple(store, tmp_path):
    """One-time upgrade for an already-E1 box: raw-file absence of the field
    is the trigger (N1 stamp pattern); triple -> True, others -> False;
    malformed entries untouched; second call is a no-op."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "hipfire", "kind": "hipfire",
              "connection": {"container": "ods-hipfire"}, "gpu_index": 0,
              "policy_defaults": {"priority": 100, "pinned": True, "idle_ttl": 0}},
             {"resource": "custom-thing", "kind": "lemonade",
              "connection": {"url": "http://custom:8080",
                             "metrics_url": "http://custom:8001/metrics",
                             "container": "ods-custom"},
              "gpu_index": 2,
              "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}},
         ]},  # entries WITHOUT container_consent, written raw
    ])
    assert store.stamp_missing_container_consent() is True
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert engines["hipfire"]["container_consent"] is True
    assert engines["custom-thing"]["container_consent"] is False
    assert store.stamp_missing_container_consent() is False  # idempotent


def test_stamp_missing_container_consent_tolerates_a_null_engines_entry(store, tmp_path):
    """Review fix round 1 (Finding 1, reproduced boot crash): a node entry
    with `"engines": null` on disk is a shape this codebase explicitly
    anticipates (`_heal_engines` heals it to [] on `_load`; the sibling
    ownership-scan at node_store.py's `for engine in other.get("engines")
    or []` already tolerates it) — but the migration ran BEFORE _load's
    healing, at boot, unconditionally, so a hand-edited/corrupted
    nodes.json that boots fine TODAY would hard-crash startup on
    `for eng in entry.get("engines", [])` (the `{}`/list default only
    applies when the key is ABSENT, not when it is present-and-None).
    Must not crash, and a null `engines` has nothing to stamp -> no
    write, `False`."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": None},
    ])
    assert store.stamp_missing_container_consent() is False


# --- Finding 1c (whole-branch review): stamp_missing_comfyui_container ----
# One-time upgrade for an already-E1 box's pre-existing comfyui entry, now
# that comfyui's schema accepts connection.container (Finding 1a) — the
# sibling to stamp_missing_container_consent, same raw-read/lock/idempotent
# pattern, keyed per-entry on absence of connection.container, kind=="comfyui"
# only.

def test_stamp_missing_comfyui_container_adds_the_settings_default(store, tmp_path):
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "comfyui", "kind": "comfyui",
              "connection": {"url": "http://comfyui:8188"}, "gpu_index": 1,
              "container_consent": True,
              "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
         ]},
    ])
    settings = _settings()
    assert store.stamp_missing_comfyui_container(settings) is True
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert engines["comfyui"]["connection"]["container"] == settings.comfyui_container
    # Idempotent: a second call finds nothing left to stamp.
    assert store.stamp_missing_comfyui_container(settings) is False


def test_stamp_missing_comfyui_container_ignores_non_comfyui_kinds(store, tmp_path):
    """Only kind=="comfyui" is touched — a lemonade-kind entry (whose
    schema REQUIRES container, so a raw entry missing it is already
    schema-invalid and heals to [] by a different path entirely) must never
    be mistaken for this migration's target."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "gguf-a", "kind": "lemonade",
              "connection": {"url": "http://gguf-a:8080",
                             "metrics_url": "http://gguf-a:8001/metrics",
                             "container": "ods-gguf-a"},
              "gpu_index": 2, "container_consent": True,
              "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}},
         ]},
    ])
    assert store.stamp_missing_comfyui_container(_settings()) is False
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert engines["gguf-a"]["connection"]["container"] == "ods-gguf-a"  # untouched


def test_stamp_missing_comfyui_container_leaves_an_already_stamped_entry_alone(store, tmp_path):
    """An operator-set (or already-migrated) container name is never
    overwritten — the raw-file ABSENCE check is the whole gate, same as
    stamp_missing_control never re-promoting a demoted node."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "comfyui", "kind": "comfyui",
              "connection": {"url": "http://comfyui:8188",
                             "container": "operator-named-comfy"},
              "gpu_index": 1, "container_consent": True,
              "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
         ]},
    ])
    assert store.stamp_missing_comfyui_container(_settings()) is False
    engines = {e["resource"]: e for e in store.get("local")["engines"]}
    assert engines["comfyui"]["connection"]["container"] == "operator-named-comfy"


def test_stamp_missing_comfyui_container_tolerates_a_null_engines_entry(store, tmp_path):
    """Mirrors stamp_missing_container_consent's own null-engines
    regression test: a hand-edited `"engines": null` must not crash the
    migration at boot."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": None},
    ])
    assert store.stamp_missing_comfyui_container(_settings()) is False


def test_stamp_missing_comfyui_container_tolerates_a_non_dict_connection(store, tmp_path):
    """`isinstance(eng, dict)` guards the outer element (Finding 7's idiom,
    applied here too); a malformed `connection` on an otherwise well-formed
    comfyui entry must not raise either."""
    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "comfyui", "kind": "comfyui", "connection": "not-a-dict",
              "gpu_index": 1, "container_consent": True,
              "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
             "not a dict either",
         ]},
    ])
    assert store.stamp_missing_comfyui_container(_settings()) is False


def test_declared_containers_includes_a_stamped_comfyui_entry_end_to_end(store, tmp_path):
    """Finding 1d: the whole chain, real store to real declared_containers
    — an upgrading box's pre-existing comfyui entry regains OCI provenance
    coverage exactly like the fresh-seed path (Finding 1b) does."""
    from app.node_store import declared_containers

    _legacy_file(tmp_path, [
        {"id": "local", "label": "Box L", "agent_kind": "local",
         "engines": [
             {"resource": "comfyui", "kind": "comfyui",
              "connection": {"url": "http://comfyui:8188"}, "gpu_index": 1,
              "container_consent": True,
              "policy_defaults": {"priority": 40, "pinned": False, "idle_ttl": 300}},
         ]},
    ])
    settings = _settings()
    assert declared_containers(store) == set()  # before the stamp: not enumerable
    store.stamp_missing_comfyui_container(settings)
    assert declared_containers(store) == {settings.comfyui_container}


# --- Task 7 fix round 1: the LOCAL direction of the run-location gate ------

_LOCAL_ONLY_REFUSED_ENGINE = {
    "resource": "song-r", "kind": "sglang-omni",
    "connection": {"url": "http://127.0.0.1:8008"},
    "gpu_index": 4,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}


def test_heal_local_entry_with_a_remote_only_kind_heals_to_empty(store, tmp_path):
    """Fix round 1 (review finding 2), the LOAD side of the same gate: a
    hand-edited nodes.json declaring a remote-only kind on the LOCAL entry
    heals to [] instead of surviving to brick every world snapshot. Mirrors
    test_heal_node_agent_entry_with_non_remote_capable_kind_heals_to_empty,
    the other direction of the same rule."""
    _write_nodes(tmp_path, [{"id": "local", "label": "This Box",
                             "agent_kind": "local",
                             "engines": [_LOCAL_ONLY_REFUSED_ENGINE]}])
    reloaded = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")

    assert reloaded.get("local")["engines"] == []


def test_local_entry_refuses_a_remote_only_kind_naming_it(store):
    """...and the WRITE side: refused by name, before anything lands."""
    with pytest.raises(ValueError, match="sglang-omni") as exc:
        store.add({"id": "local", "label": "Box L", "agent_kind": "local",
                   "engines": [_LOCAL_ONLY_REFUSED_ENGINE]})
    assert "remote-only" in str(exc.value)



# --- sglang-omni Task 9 / ruling R10: resource names are unique DECK-WIDE --
#
# PolicyStore keys its rows by BARE resource and has no node dimension
# anywhere (app/policy.py). R10 keeps that keying and pays for it here, at
# the declaration boundary [[guard-at-the-boundary]]: if a resource name can
# only ever be declared on ONE node, a bare-resource policy row unambiguously
# belongs to that node's engine, and `forget` can drop it without needing to
# know which box it was on.

_OMNI_ENGINE = {
    "resource": "song-r", "kind": "sglang-omni",
    "connection": {"url": "http://nimbus:8008"},
    "gpu_index": 4,
    "container_consent": True,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}


def _local_gguf(resource="gguf-a"):
    return {"resource": resource, "kind": "lemonade",
            "connection": {"url": f"http://{resource}:8080",
                           "metrics_url": f"http://{resource}:8001/metrics",
                           "container": f"ods-{resource}"},
            "gpu_index": 2,
            "container_consent": True,
            "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}}


def test_a_resource_another_node_already_declares_is_refused_naming_the_owner(store):
    """The refusal has to name BOTH halves: which resource collided and
    which node already owns it — an operator holding one node's editor
    cannot see the other's declaration."""
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    store.add({**_remote_spec(), "engines": [_OMNI_ENGINE]}, credential="key-nimbus")

    with pytest.raises(ValueError, match="song-r") as exc:
        store.update("local", {"engines": [_local_gguf("song-r")]})
    assert "nimbus" in str(exc.value)
    # Refused before any write: the local entry is untouched.
    assert store.get("local").get("engines", []) == []


def test_redeclaring_a_resource_on_its_OWN_node_is_not_a_collision(store):
    """The gate compares against OTHER nodes only — editing a node's own
    declaration (the update route's whole job) must not see itself."""
    store.add({**_remote_spec(), "engines": [_OMNI_ENGINE]}, credential="key-nimbus")

    entry = store.update("nimbus", {"engines": [{**_OMNI_ENGINE, "gpu_index": 6}]})

    assert entry["engines"][0]["gpu_index"] == 6


def test_load_heals_a_hand_edited_cross_node_duplicate(store, tmp_path):
    """Heal-side consistency: a nodes.json hand-edited past the write gate
    must not leave two nodes claiming one policy row. The FIRST entry in file
    order keeps the name (deterministic, no guessing which is "real"); the
    later one loses that engine and keeps the rest — the surgical heal, not
    the whole-list wipe a schema-invalid declaration gets, because the other
    entries are individually fine."""
    local = {"id": "local", "label": "This Box", "agent_kind": "local",
             "control": "none", "engines": [_local_gguf("song-r")]}
    remote = {**_remote_spec(), "control": "none",
              "engines": [_OMNI_ENGINE, {**_OMNI_ENGINE, "resource": "song-b"}]}
    _write_nodes(tmp_path, [local, remote])

    reloaded = NodeStore(tmp_path / "nodes.json", tmp_path / "node_credentials.json")

    assert [e["resource"] for e in reloaded.get("local")["engines"]] == ["song-r"]
    assert [e["resource"] for e in reloaded.get("nimbus")["engines"]] == ["song-b"]


# --- Finding 2 (whole-branch review): consented_containers / declared_
# containers had ZERO direct test coverage despite being the guard docker_
# ctl.py's `_guard` and the provenance sweep both depend on. Direct unit
# tests on a real NodeStore (not the FakeDeclaredNodeStore precedent
# tests/test_arbiter.py uses for its own higher-level provenance-pass
# coverage) below.

from app.node_store import consented_containers, declared_containers


def test_consented_containers_no_store_is_empty():
    assert consented_containers(None) == set()


def test_consented_containers_no_local_entry_is_empty(store):
    # A fresh store has no entries at all -- get("local") answers None.
    assert consented_containers(store) == set()


def test_consented_containers_filters_entries_without_connection_container(store):
    """comfyui's container is OPTIONAL (Finding 1a) -- an entry that
    consents but names no container has nothing for the guard to allowlist."""
    store.add({"id": "local", "label": "Box L", "agent_kind": "local",
               "engines": [{"resource": "img", "kind": "comfyui",
                            "connection": {"url": "http://comfyui:8188"},
                            "gpu_index": 1, "container_consent": True,
                            "policy_defaults": {"priority": 1, "pinned": False,
                                                "idle_ttl": 0}}]})
    assert consented_containers(store) == set()


def test_consented_containers_requires_consent_is_true_not_merely_truthy(store, monkeypatch):
    """The guard's own comparison is `is True`, strict — not
    `if consent:`. A real NodeStore load can never surface an int consent
    value (validate_engines' `isinstance(consent, bool)` check makes
    `_heal_engines` wipe the WHOLE engines list to [] before this function
    ever sees it), so this test monkeypatches THIS store's own `.get`
    directly rather than writing the shape to disk — isolating the guard's
    comparison from the load-time healing that would otherwise make ANY
    non-bool consent untestable this way."""
    raw_local = {"id": "local", "engines": [
        {"resource": "hipfire", "connection": {"container": "ods-hipfire"},
         "container_consent": 1},
    ]}
    monkeypatch.setattr(store, "get",
                        lambda node_id: raw_local if node_id == "local" else None)
    assert consented_containers(store) == set()


def test_consented_containers_never_includes_a_node_agents_engines(store):
    """consented_containers reads `node_store.get("local")` only -- a
    node-agent entry's own consented, container-bearing... except no
    remote_capable kind's schema HAS a container field today (sglang-omni
    is url-only), so this pins the structural fact directly: even with a
    populated engines[] on a node-agent entry, nothing from it ever
    reaches either function, because neither ever asks for that node."""
    store.add({"id": "local", "label": "Box L", "agent_kind": "local"})
    store.add({**_remote_spec(), "engines": [_OMNI_ENGINE]}, credential="key-nimbus")
    assert consented_containers(store) == set()
    assert declared_containers(store) == set()


def test_declared_containers_no_store_is_empty():
    assert declared_containers(None) == set()


def test_declared_containers_no_local_entry_is_empty(store):
    assert declared_containers(store) == set()


def test_consented_containers_never_raises_on_a_non_dict_engine_element(store, monkeypatch):
    """Finding 7 (code half): the docstring already promises "malformed
    engines -> empty set, never an exception" -- pin it true for a raw
    non-dict sibling, matching stamp_missing_container_consent's own
    `isinstance(eng, dict)` idiom. Reachable in principle from a caller that
    hands in a node_store whose `.get()` does not itself validate/heal (this
    module's docstring claims independence from that healing)."""
    raw_local = {"id": "local", "engines": [
        "not a dict", 42, None,
        {"resource": "hipfire", "connection": {"container": "ods-hipfire"},
         "container_consent": True},
    ]}
    monkeypatch.setattr(store, "get",
                        lambda node_id: raw_local if node_id == "local" else None)
    assert consented_containers(store) == {"ods-hipfire"}


def test_declared_containers_never_raises_on_a_non_dict_engine_element(store, monkeypatch):
    raw_local = {"id": "local", "engines": [
        "not a dict", 42, None,
        {"resource": "hipfire", "connection": {"container": "ods-hipfire"}},
    ]}
    monkeypatch.setattr(store, "get",
                        lambda node_id: raw_local if node_id == "local" else None)
    assert declared_containers(store) == {"ods-hipfire"}


def test_declared_containers_is_consent_blind(store):
    """CONTROLLER RULING (declared_containers' own docstring): enumeration
    is not consent -- a container_consent:False entry still enumerates for
    the provenance sweep, even though the SAME entry is excluded from
    consented_containers."""
    store.add({"id": "local", "label": "Box L", "agent_kind": "local",
               "engines": [{"resource": "hipfire", "kind": "hipfire",
                            "connection": {"container": "ods-hipfire"},
                            "gpu_index": 0, "container_consent": False,
                            "policy_defaults": {"priority": 100, "pinned": True,
                                                "idle_ttl": 0}}]})
    assert declared_containers(store) == {"ods-hipfire"}
    assert consented_containers(store) == set()


# ===========================================================================
# INST I1 Task 1 — control: "instances", instance_port_range, the gpu_indices
# / gateway_host boot stamps
# ===========================================================================


def test_controls_include_instances_and_local_may_declare_it(store):
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    store.update("local", {"address": "http://172.18.0.1:7720",
                           "instance_port_range": {"start": 11500, "end": 11509}},
                 credential="k")
    entry = store.update("local", {"control": "instances"})
    assert entry["control"] == "instances"


def test_instances_prereqs_are_named(store):
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    with pytest.raises(ValueError) as exc:
        store.update("local", {"control": "instances"})
    assert str(exc.value) == ('control: "instances" requires address, credential, '
                              'instance_port_range to be set first')


def test_instance_port_range_shape(store):
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    for bad, why in (({"start": 80, "end": 90}, "1024-65535"),
                     ({"start": 2000, "end": 1999}, "start <= end"),
                     ({"start": 2000}, "exactly"), ([2000, 2001], "object")):
        with pytest.raises(ValueError, match=why):
            store.update("local", {"instance_port_range": bad})


def test_stamp_missing_gpu_indices_normalises_the_raw_file_once(store, tmp_path):
    import json
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    raw = json.loads(store._path.read_text())
    local = next(e for e in raw if e["id"] == "local")
    local["engines"] = [{"resource": "gguf-a", "kind": "lemonade", "gpu_index": 3,
                         "connection": {"url": "http://g:8080", "metrics_url": "http://g:8001/metrics",
                                        "container": "deck-gguf-a"},
                         "container_consent": True,
                         "policy_defaults": {"priority": 0, "pinned": False, "idle_ttl": 0}}]
    store._path.write_text(json.dumps(raw))
    assert store.stamp_missing_gpu_indices() is True
    eng = store.get("local")["engines"][0]
    assert "gpu_index" not in eng and eng["gpu_indices"] == [3]
    assert store.stamp_missing_gpu_indices() is False   # at most once


def test_stamp_missing_gateway_host_only_touches_the_seeded_hipfire_container(store, tmp_path):
    import json
    store.add({"id": "local", "label": "This Box", "agent_kind": "local"})
    raw = json.loads(store._path.read_text())
    local = next(e for e in raw if e["id"] == "local")
    local["engines"] = [
        {"resource": "hipfire", "kind": "hipfire", "gpu_indices": [0],
         "connection": {"container": "ods-hipfire"}, "container_consent": True,
         "policy_defaults": {"priority": 100, "pinned": False, "idle_ttl": 0}},
        {"resource": "agent", "kind": "hipfire", "gpu_indices": [2],
         "connection": {"container": "deck-agent"}, "container_consent": True,
         "policy_defaults": {"priority": 100, "pinned": False, "idle_ttl": 0}},
    ]
    store._path.write_text(json.dumps(raw))
    assert store.stamp_missing_gateway_host("ods-hipfire") is True
    a, b = store.get("local")["engines"]
    assert a["connection"]["gateway_host"] == "hipfire"      # = container minus the "ods-" prefix: the compose SERVICE name
    assert "gateway_host" not in b["connection"]
    assert store.stamp_missing_gateway_host("ods-hipfire") is False
