"""Tests for app.policy — the resource policy store (E1 Task 4).

E1 Task 4: policy.json is a flat mapping of {resource: {priority, pinned,
idle_ttl}} with defaults from declared engines[]. get() returns one row per
DECLARED resource (declared defaults overlaid by stored overrides); stored rows
for undeclared resources are invisible on read but kept on disk (orphan defense:
hand-edits, older writes). Defaults materialize and persist on first get()
(missing or corrupt file self-heals to declared defaults). Writes are atomic
(temp file + os.replace), matching app.registry's idiom.
"""

import json

import pytest

from app.policy import PolicyStore

# Known defaults for test purposes — replaces the deleted DEFAULT_POLICIES.
# Tests use this to declare the 3 legacy engines when testing default seeding.
_TEST_KNOWN_DEFAULTS = {
    "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
    "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
    "comfyui": {"priority": 40, "pinned": False, "idle_ttl": 300},
}


def _policy_store(tmp_path, declared=None):
    """Test helper: create a PolicyStore with optional declared defaults provider.

    Args:
        tmp_path: pytest's tmp_path fixture
        declared: dict of {resource: {priority, pinned, idle_ttl}} or None

    Returns:
        PolicyStore configured with declared defaults
    """
    policy_path = tmp_path / "policy.json"

    if declared is None:
        # Legacy behavior: no declared defaults provider
        return PolicyStore(policy_path)

    # New behavior: provide declared defaults
    def provider():
        return declared

    return PolicyStore(policy_path, declared_defaults=provider)


def test_known_defaults_values():
    """Verify the known defaults match expectations (E1 Task 4)."""
    assert _TEST_KNOWN_DEFAULTS == {
        "hipfire": {"priority": 100, "pinned": True, "idle_ttl": 0},
        "lemonade": {"priority": 50, "pinned": False, "idle_ttl": 900},
        "comfyui": {"priority": 40, "pinned": False, "idle_ttl": 300},
    }


def test_get_materializes_defaults_when_file_missing(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    result = store.get()

    assert result == _TEST_KNOWN_DEFAULTS


def test_get_persists_defaults_to_disk_on_first_read(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    store.get()

    assert json.loads((tmp_path / "policy.json").read_text()) == _TEST_KNOWN_DEFAULTS


def test_get_parent_dir_created_on_first_read(tmp_path):
    policy_path = tmp_path / "nested" / "does" / "not" / "exist" / "policy.json"
    store = PolicyStore(policy_path)

    store.get()

    assert policy_path.is_file()


def test_get_does_not_rewrite_file_once_it_exists(tmp_path):
    # All three known tenants present and valid (unlike a partial hand-edit)
    # so the boundary gate is a no-op and this proves the narrower claim:
    # an already-well-formed file's custom values survive untouched.
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.get()
    custom = {
        "hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5},
        "lemonade": dict(_TEST_KNOWN_DEFAULTS["lemonade"]),
        "comfyui": dict(_TEST_KNOWN_DEFAULTS["comfyui"]),
    }
    (tmp_path / "policy.json").write_text(json.dumps(custom))

    result = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS).get()

    assert result == custom


def test_put_then_get_roundtrips_across_instances(tmp_path):
    first = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    first.get()

    first.put({"hipfire": {"priority": 200, "pinned": False, "idle_ttl": 60}})

    second = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    result = second.get()

    assert result["hipfire"] == {"priority": 200, "pinned": False, "idle_ttl": 60}


def test_put_is_partial_update_other_tenants_keep_current_values(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.get()

    store.put({"lemonade": {"priority": 77, "pinned": True, "idle_ttl": 42}})

    result = store.get()
    assert result["lemonade"] == {"priority": 77, "pinned": True, "idle_ttl": 42}
    assert result["hipfire"] == _TEST_KNOWN_DEFAULTS["hipfire"]
    assert result["comfyui"] == _TEST_KNOWN_DEFAULTS["comfyui"]


def test_put_write_is_atomic_no_temp_files_left_behind(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.get()

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    leftovers = list((tmp_path / "policy.json").parent.glob("*.tmp"))
    assert leftovers == []


def test_put_parent_dir_created_when_file_never_read(tmp_path):
    policy_path = tmp_path / "nested" / "policy.json"
    store = PolicyStore(policy_path, declared_defaults=lambda: _TEST_KNOWN_DEFAULTS)

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    assert policy_path.is_file()


def test_put_rejects_the_reserved_auto_key_naming_it(tmp_path):
    """The reserved config key is not a resource and put() must say so."""
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError, match="_auto"):
        store.put({"_auto": {"priority": 1, "pinned": False, "idle_ttl": 5}})


def test_put_rejects_missing_field(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False}})


def test_put_rejects_extra_field(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put(
            {"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 5, "extra": 1}}
        )


def test_put_rejects_priority_bool_instead_of_int(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": True, "pinned": False, "idle_ttl": 5}})


def test_put_rejects_priority_wrong_type(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": "high", "pinned": False, "idle_ttl": 5}})


def test_put_rejects_pinned_wrong_type(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": "yes", "idle_ttl": 5}})


def test_put_rejects_idle_ttl_wrong_type(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": "5"}})


def test_put_rejects_idle_ttl_bool(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": True}})


def test_put_rejects_negative_idle_ttl(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": -1}})


def test_put_accepts_idle_ttl_zero(tmp_path):
    store = _policy_store(tmp_path, declared={"hipfire": _TEST_KNOWN_DEFAULTS["hipfire"]})

    store.put({"hipfire": {"priority": 1, "pinned": False, "idle_ttl": 0}})

    assert store.get()["hipfire"]["idle_ttl"] == 0


def test_put_rejected_payload_leaves_file_unchanged(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    before = store.get()

    with pytest.raises(ValueError):
        store.put(
            {
                "hipfire": {"priority": 9, "pinned": False, "idle_ttl": 5},
                "sdxl": {"priority": "high", "pinned": False, "idle_ttl": 5},
            }
        )

    assert store.get() == before


def test_corrupt_policy_json_treated_as_defaults(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not valid json")
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    assert store.get() == _TEST_KNOWN_DEFAULTS


def test_corrupt_policy_json_self_heals_on_get(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not valid json")
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    store.get()

    assert json.loads(policy_path.read_text()) == _TEST_KNOWN_DEFAULTS


def test_non_dict_json_treated_as_defaults(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(["not", "a", "dict"]))
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    assert store.get() == _TEST_KNOWN_DEFAULTS


def test_put_accepts_a_resource_in_declaration(tmp_path):
    """Declared resources are accepted in put()."""
    store = _policy_store(tmp_path, declared={
        "sparky-vllm": {"priority": 60, "pinned": False, "idle_ttl": 0}})

    store.put({"sparky-vllm": {"priority": 70, "pinned": True, "idle_ttl": 100}})

    assert store.get()["sparky-vllm"]["priority"] == 70


def test_put_still_validates_field_types_for_declared_resources(tmp_path):
    store = _policy_store(tmp_path, declared={
        "sparky-vllm": {"priority": 60, "pinned": False, "idle_ttl": 0}})

    with pytest.raises(ValueError):
        store.put({"sparky-vllm": {"priority": "high", "pinned": False, "idle_ttl": 0}})


def test_put_still_rejects_unknown_fields(tmp_path):
    store = _policy_store(tmp_path, declared={})

    with pytest.raises(ValueError):
        store.put({"sparky-vllm": {"priority": 1, "pinned": False, "idle_ttl": 0, "wat": 1}})


def test_declared_resources_seeded_on_first_get(tmp_path):
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    assert set(store.get()) == set(_TEST_KNOWN_DEFAULTS)


def test_auto_enabled_defaults_to_true(tmp_path):
    """Lifecycle auto-restore is ON by default (a deliberate difference from
    storage tiering, whose automation defaults off)."""
    store = _policy_store(tmp_path, declared={})

    assert store.auto_enabled() is True


def test_set_auto_persists(tmp_path):
    first = _policy_store(tmp_path, declared={})

    first.set_auto(False)

    second = _policy_store(tmp_path, declared={})
    assert second.auto_enabled() is False


def test_auto_key_is_not_returned_as_a_resource(tmp_path):
    """'_auto' is config, not a resource — it must never show up in the
    policy table the UI renders or the arbiter iterates."""
    store = _policy_store(tmp_path, declared={})
    store.set_auto(False)

    assert "_auto" not in store.get()


def test_set_auto_on_a_fresh_file_still_seeds_the_declared_resources(tmp_path):
    """set_auto must not be able to create a policy.json that permanently
    suppresses declared-resource seeding: `_load()` only heals an existing file,
    so a file that has never been written still needs set_auto's own fallback,
    or a file containing just _auto would look valid and leave resources unpolicied."""
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)

    store.set_auto(False)

    second = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    assert set(second.get()) == set(_TEST_KNOWN_DEFAULTS)
    assert second.auto_enabled() is False


# ===========================================================================
# Boundary gate — element-level healing (NodeStore._load's pattern, 017fd207)
# ===========================================================================


def test_get_heals_a_missing_declared_resource(tmp_path):
    """Hand-edit threat model: a parseable file missing a declared resource
    must come back with the resource's default materialized."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    del data["comfyui"]
    path.write_text(json.dumps(data))
    got = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS).get()
    assert got["comfyui"] == _TEST_KNOWN_DEFAULTS["comfyui"]
    # and the heal persists (next reader of the raw file sees it too)
    assert "comfyui" in json.loads(path.read_text())


def test_get_replaces_a_malformed_declared_resource_record(tmp_path):
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    data["lemonade"] = "not-a-dict"
    path.write_text(json.dumps(data))
    got = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS).get()
    assert got["lemonade"] == _TEST_KNOWN_DEFAULTS["lemonade"]


def test_get_keeps_declared_resources_drops_undeclared_ones(tmp_path):
    """Undeclared stored rows stay on disk but don't appear in get()."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    data["sparky-thing"] = {"priority": 10, "pinned": False, "idle_ttl": 60}   # undeclared, invisible
    path.write_text(json.dumps(data))
    got = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS).get()
    # Only declared resources appear
    assert set(got) == set(_TEST_KNOWN_DEFAULTS)
    # But undeclared row stays on disk
    assert "sparky-thing" in json.loads(path.read_text())


def test_get_preserves_auto_key_through_the_gate(tmp_path):
    """The gate must not cost the file its _auto record."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    data["_auto"] = {"enabled": False}
    path.write_text(json.dumps(data))
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.get()
    assert store.auto_enabled() is False


def test_put_onto_a_partial_file_also_heals(tmp_path):
    """put()'s merge reads through `_load()` (the sole boundary gate), so a
    hand-edit missing a declared resource heals on the put that touches an
    unrelated resource, not just on the next get()."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    del data["comfyui"]
    path.write_text(json.dumps(data))

    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.put({"lemonade": {"priority": 5, "pinned": True, "idle_ttl": 0}})

    on_disk = json.loads(path.read_text())
    assert on_disk["comfyui"] == _TEST_KNOWN_DEFAULTS["comfyui"]
    assert on_disk["lemonade"] == {"priority": 5, "pinned": True, "idle_ttl": 0}


def test_set_auto_onto_a_partial_file_also_heals(tmp_path):
    """set_auto() also reads through `_load()`, so it inherits the same
    heal as get()/put() — a hand-edit missing a declared resource does not survive."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    del data["comfyui"]
    path.write_text(json.dumps(data))

    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.set_auto(False)

    on_disk = json.loads(path.read_text())
    assert on_disk["comfyui"] == _TEST_KNOWN_DEFAULTS["comfyui"]
    assert on_disk["_auto"] == {"enabled": False}


def test_get_drops_a_non_dict_auto_record(tmp_path):
    """Malformed _auto: not a dict is dropped, healing to default-True."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    data["_auto"] = True
    path.write_text(json.dumps(data))

    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    assert store.auto_enabled() is True
    assert "_auto" not in json.loads(path.read_text())


def test_get_drops_an_auto_record_with_a_non_bool_enabled(tmp_path):
    """Malformed _auto: non-bool enabled is dropped, healing to default-True."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    data["_auto"] = {"enabled": "no"}
    path.write_text(json.dumps(data))

    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    assert store.auto_enabled() is True
    assert "_auto" not in json.loads(path.read_text())


# ===========================================================================
# Write suppression — get() must not touch an already well-formed file
# ===========================================================================


def test_get_does_not_rewrite_an_already_well_formed_file(tmp_path):
    """Suppression property: get() doesn't rewrite a well-formed file.
    Checked at the filesystem level (mtime) since resaved bytes are identical."""
    path = tmp_path / "policy.json"
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    store.get()  # materializes + persists the defaults once

    before = path.stat().st_mtime_ns
    store.get()
    after = path.stat().st_mtime_ns

    assert after == before


def test_get_rewrites_a_healed_file_exactly_once(tmp_path):
    """Suppression property: a partial file gets rewritten by the healing
    get(), but never again after that."""
    path = tmp_path / "policy.json"
    data = {t: dict(p) for t, p in _TEST_KNOWN_DEFAULTS.items()}
    del data["comfyui"]
    path.write_text(json.dumps(data))
    store = _policy_store(tmp_path, declared=_TEST_KNOWN_DEFAULTS)
    before_heal = path.stat().st_mtime_ns

    store.get()
    after_heal = path.stat().st_mtime_ns
    assert after_heal != before_heal        # first get() heals + persists

    store.get()
    after_second_get = path.stat().st_mtime_ns
    assert after_second_get == after_heal   # second get() is a no-op


# ===========================================================================
# E1 Task 4: Policy defaults from the declaration
# ===========================================================================


def test_policy_rows_come_from_declaration(tmp_path):
    """Policy defaults come from the declared engines, not hardcoded."""
    store = _policy_store(tmp_path, declared={
        "gguf-a": {"priority": 10, "pinned": False, "idle_ttl": 60}})
    got = store.get()
    assert got["gguf-a"] == {"priority": 10, "pinned": False, "idle_ttl": 60}
    store.put({"gguf-a": {"priority": 99, "pinned": True, "idle_ttl": 0}})
    assert store.get()["gguf-a"]["priority"] == 99


def test_forget_removes_a_stored_row(tmp_path):
    """E1 Task 10: forget() pops the row from disk (IntentStore.forget's
    sibling — tests/test_intent.py's test_forget_removes_the_key)."""
    store = _policy_store(tmp_path, declared={})
    store.put({"gguf-a": {"priority": 1, "pinned": False, "idle_ttl": 5}})

    store.forget("gguf-a")

    assert "gguf-a" not in json.loads((tmp_path / "policy.json").read_text())


def test_forget_of_a_key_with_no_row_is_a_noop(tmp_path):
    store = _policy_store(tmp_path, declared={})

    store.forget("ghost")  # must not raise


def test_forget_then_redeclare_gets_a_fresh_default_not_the_old_row(tmp_path):
    """The point of forget deleting rather than merely hiding: a stored
    override must not resurrect under a re-declaration of the same name."""
    store = _policy_store(tmp_path, declared={
        "gguf-a": {"priority": 10, "pinned": False, "idle_ttl": 60}})
    store.put({"gguf-a": {"priority": 999, "pinned": True, "idle_ttl": 0}})
    store.forget("gguf-a")

    assert store.get()["gguf-a"] == {"priority": 10, "pinned": False, "idle_ttl": 60}


def test_undeclared_stored_row_is_invisible_but_kept(tmp_path):
    """Orphaned row (e.g. hand-edited in): kept on disk but invisible on read.

    This defends against rows left by older writes or hand-edits. When the
    resource is re-declared, the row becomes visible again.
    """
    store = _policy_store(tmp_path, declared={})
    # a row survives on disk from hand-edits or older writes
    store_with = _policy_store(tmp_path, declared={
        "gguf-a": {"priority": 10, "pinned": False, "idle_ttl": 60}})
    store_with.put({"gguf-a": {"priority": 7, "pinned": False, "idle_ttl": 5}})
    assert "gguf-a" not in store.get()           # undeclared -> invisible
    assert store_with.get()["gguf-a"]["priority"] == 7   # redeclare -> back


# --- sglang-omni Task 9: declared defaults come from EVERY node ------------


class _FakeRegistry:
    """NodeStore's read half — `list()` is all `declared_defaults` needs."""

    def __init__(self, entries):
        self._entries = entries

    def list(self):
        return [dict(e) for e in self._entries]


_REMOTE_OMNI = {
    "resource": "song-r", "kind": "sglang-omni",
    "connection": {"url": "http://nimbus:8008"},
    "gpu_index": 4,
    "policy_defaults": {"priority": 5, "pinned": False, "idle_ttl": 120},
}

_LOCAL_GGUF = {
    "resource": "gguf-a", "kind": "lemonade",
    "connection": {"url": "http://gguf-a:8080",
                   "metrics_url": "http://gguf-a:8001/metrics",
                   "container": "ods-gguf-a"},
    "gpu_index": 2,
    "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60},
}


def test_declared_defaults_include_engines_on_a_node_agent_entry(tmp_path):
    """A remote engine gets its policy row seeded from its OWN declaration,
    exactly as a local one does — the same walk, extended past the local
    entry rather than forked. Without this a declared remote engine has no
    policy row at all, and app.arbiter's idle rule skips any resource whose
    row is missing (`policy.get(resource) is None` -> nothing safe to
    decide), so it could never be idle-released."""
    from app.policy import declared_defaults

    registry = _FakeRegistry([
        {"id": "local", "agent_kind": "local", "engines": [_LOCAL_GGUF]},
        {"id": "nimbus", "agent_kind": "node-agent", "engines": [_REMOTE_OMNI]},
    ])

    assert declared_defaults(registry) == {
        "gguf-a": _LOCAL_GGUF["policy_defaults"],
        "song-r": _REMOTE_OMNI["policy_defaults"],
    }


def test_declared_defaults_materialize_a_remote_row_on_first_read(tmp_path):
    """End to end through the store: the seeded row is the DECLARED one
    (priority 5 / idle_ttl 120, both away from every legacy default), which
    is what the arbiter's remote idle rule reads."""
    from app.policy import declared_defaults

    registry = _FakeRegistry([
        {"id": "nimbus", "agent_kind": "node-agent", "engines": [_REMOTE_OMNI]}])
    store = PolicyStore(tmp_path / "policy.json",
                        declared_defaults=lambda: declared_defaults(registry))

    assert store.get() == {"song-r": _REMOTE_OMNI["policy_defaults"]}
    assert json.loads((tmp_path / "policy.json").read_text())["song-r"][
        "idle_ttl"] == 120


def test_declared_defaults_tolerates_an_entry_with_no_engines(tmp_path):
    """A registry row that declares nothing (every node before its first
    declaration) contributes nothing — absence is representable, not a
    KeyError on the deck's first read."""
    from app.policy import declared_defaults

    registry = _FakeRegistry([{"id": "local", "agent_kind": "local"},
                              {"id": "nimbus", "agent_kind": "node-agent",
                               "engines": []}])

    assert declared_defaults(registry) == {}
