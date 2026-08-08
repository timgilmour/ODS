"""Tests for app.settings_store — the human-owned configuration store.

Three scopes (engine, model, engine x model) x three namespaces (args, env,
container). Namespaces exist because a real profile sets far more than
flags: VLLM_USE_FLASHINFER_SAMPLER=1, a pinned image digest, shm_size 16g.
"""

import json

import pytest

from app.settings_store import CONTAINER_ALLOWLIST, SettingsStore


def test_empty_when_missing_or_corrupt(tmp_path):
    assert SettingsStore(tmp_path / "s.json").get() == {
        "engines": {}, "models": {}, "engine_models": {}}
    path = tmp_path / "s2.json"
    path.write_text("garbage")
    assert SettingsStore(path).get()["engines"] == {}


def test_put_and_read_engine_args(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args", {"generation-config": "auto"})

    assert store.scope("engines", "sparky/vllm")["args"] == {"generation-config": "auto"}


def test_engine_scope_is_node_keyed(tmp_path):
    """autarch/lemonade and sparky/vllm are independent scopes — the same
    engine on two nodes is two things."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "autarch/lemonade", "args", {"a": "1"})
    store.put("engines", "sparky/vllm", "args", {"a": "2"})

    assert store.scope("engines", "autarch/lemonade")["args"]["a"] == "1"
    assert store.scope("engines", "sparky/vllm")["args"]["a"] == "2"


def test_model_scope_is_node_neutral(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.put("models", "Qwen3.6-35B-A3B-heretic-NVFP4", "args", {"max-model-len": "131072"})

    assert store.scope("models", "Qwen3.6-35B-A3B-heretic-NVFP4")["args"]["max-model-len"] == "131072"


def test_engine_model_scope_uses_the_composite_key(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.put("engine_models", "sparky/vllm|Qwen3.6-35B-A3B-heretic-NVFP4",
              "args", {"quantization": None})

    assert store.scope("engine_models", "sparky/vllm|Qwen3.6-35B-A3B-heretic-NVFP4")["args"]


def test_put_merges_per_key_not_per_blob(tmp_path):
    """Setting one flag must not discard the others in the same namespace."""
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"a": "1", "b": "2"})

    store.put("engines", "sparky/vllm", "args", {"b": "9"})

    assert store.scope("engines", "sparky/vllm")["args"] == {"a": "1", "b": "9"}


def test_namespaces_are_independent(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"a": "1"})

    store.put("engines", "sparky/vllm", "env", {"VLLM_USE_FLASHINFER_SAMPLER": "1"})

    scope = store.scope("engines", "sparky/vllm")
    assert scope["args"] == {"a": "1"}
    assert scope["env"] == {"VLLM_USE_FLASHINFER_SAMPLER": "1"}


def test_container_namespace_allowlist(tmp_path):
    assert set(CONTAINER_ALLOWLIST) == {"image", "shm_size", "ulimits"}


def test_container_rejects_volumes(tmp_path):
    """The model mount IS the placement. Editing it here would desync
    placement from settings — that is the Deck's to manage."""
    store = SettingsStore(tmp_path / "s.json")

    with pytest.raises(ValueError, match="volumes"):
        store.put("engines", "sparky/vllm", "container", {"volumes": ["/a:/b"]})


def test_unknown_namespace_rejected(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    with pytest.raises(ValueError):
        store.put("engines", "sparky/vllm", "nonsense", {"a": "1"})


def test_unknown_kind_rejected(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    with pytest.raises(ValueError):
        store.put("nonsense", "k", "args", {"a": "1"})


def test_note_is_stored_alongside_the_namespace(tmp_path):
    """Compose comments carry load-bearing knowledge ('no --quantization:
    forcing modelopt breaks the load'). Adoption imports them here."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args", {"a": "1"},
              note="forcing modelopt breaks the load — fixed 2026-07-31")

    assert "modelopt" in store.scope("engines", "sparky/vllm")["notes"]["args"]


def test_rejected_put_leaves_the_file_untouched(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"keep": "me"})

    with pytest.raises(ValueError):
        store.put("engines", "sparky/vllm", "container", {"volumes": []})

    assert store.scope("engines", "sparky/vllm")["args"] == {"keep": "me"}


def test_forget_removes_the_scope_entry(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"a": "1"})

    store.forget("engines", "sparky/vllm")

    assert store.scope("engines", "sparky/vllm") == {}


# --- app.argline normalization axes, applied on write (Task 1 RULING 2026-08-07) ---
#
# argline's map<->text round trip is exact modulo two normalizations: a
# singleton list collapses to its scalar, and a numeric scalar becomes a
# string. A caller that stores or diffs resolved settings must normalize on
# write or compare via rendered text, never raw == on maps that might mix
# int/str or scalar/singleton-list — this store normalizes on write so a
# later == against parse_argline's output is always safe.


def test_singleton_list_arg_value_normalizes_to_a_scalar_on_write(tmp_path):
    """A one-element --served-model-name list and a scalar render
    identically through argline; storing the list form would make a later
    == against a freshly parsed argline disagree on Python type alone."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args", {"served-model-name": ["solo"]})

    assert store.scope("engines", "sparky/vllm")["args"]["served-model-name"] == "solo"


def test_numeric_arg_value_normalizes_to_a_string_on_write(tmp_path):
    """parse_argline always yields str; storing the int form here would
    make the same disagreement possible in the other direction."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args", {"max-model-len": 262144})

    assert store.scope("engines", "sparky/vllm")["args"]["max-model-len"] == "262144"


def test_multi_value_arg_list_normalizes_each_element_to_a_string(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args", {"ports": [1, 2]})

    assert store.scope("engines", "sparky/vllm")["args"]["ports"] == ["1", "2"]


def test_bare_flag_and_string_arg_values_pass_through_unnormalized(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "args",
              {"enable-prefix-caching": True, "reasoning-parser": "qwen3"})

    scope = store.scope("engines", "sparky/vllm")["args"]
    assert scope["enable-prefix-caching"] is True
    assert scope["reasoning-parser"] == "qwen3"


def test_empty_list_arg_value_warns_and_drops_the_key(tmp_path):
    """RULING 2026-08-07 (review): overturns an earlier True-normalization.
    render_argline({"k": []}) and render_argline({"k": True}) emit
    byte-identical argv, so round-trip congruence with app.argline never
    distinguished the two; a bare flag is not what an operator meant by
    an empty list; and app.intent's idiom ("deleting a key is the only
    way to say 'no opinion'") reads a zero-value list the same way.
    Warned, not silently dropped — the container allowlist remains the
    one hard validation failure in this module."""
    store = SettingsStore(tmp_path / "s.json")

    with pytest.warns(UserWarning, match="empty list"):
        store.put("engines", "sparky/vllm", "args", {"served-model-name": []})

    assert "served-model-name" not in store.scope("engines", "sparky/vllm").get("args", {})


def test_put_with_only_empty_list_values_is_a_clean_no_op_merge(tmp_path):
    """A put() whose only values are empty lists must not crash or corrupt
    the namespace — it merges an empty dict, which is a no-op on the
    existing keys (the write to disk still happens; see put()'s docstring
    for why that's an acceptable no-op rather than a special case)."""
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"a": "1"})

    with pytest.warns(UserWarning, match="empty list"):
        store.put("engines", "sparky/vllm", "args", {"tags": []})

    assert store.scope("engines", "sparky/vllm")["args"] == {"a": "1"}


def test_corrupt_scope_entry_heals_to_empty_dict(tmp_path):
    """One level deeper than the existing per-kind self-heal: a scope
    entry that is not a dict (hand-edit gone wrong, partial write) resets
    to {} instead of leaking a bare value out of get()/scope()."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"engines": {"sparky/vllm": "oops"}}))
    store = SettingsStore(path)

    assert store.scope("engines", "sparky/vllm") == {}
    assert store.get()["engines"] == {"sparky/vllm": {}}


def test_put_succeeds_after_a_corrupt_scope_entry_heals(tmp_path):
    """Same corruption as above must not crash put() with an uncaught
    AttributeError (the pre-fix behavior: .setdefault() on a str)."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"engines": {"sparky/vllm": "oops"}}))
    store = SettingsStore(path)

    store.put("engines", "sparky/vllm", "args", {"a": "1"})

    assert store.scope("engines", "sparky/vllm")["args"] == {"a": "1"}


def test_corrupt_namespace_within_an_entry_heals_to_empty_dict(tmp_path):
    """Same posture, one level further: args/env/container are each
    individually guarded against being anything other than a dict, so one
    corrupt namespace doesn't take a healthy sibling namespace down with it."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "engines": {"sparky/vllm": {"args": "oops", "env": {"a": "1"}}}
    }))
    store = SettingsStore(path)

    scope = store.scope("engines", "sparky/vllm")
    assert scope["args"] == {}
    assert scope["env"] == {"a": "1"}


def test_put_succeeds_after_a_corrupt_namespace_heals(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "engines": {"sparky/vllm": {"args": "oops"}}
    }))
    store = SettingsStore(path)

    store.put("engines", "sparky/vllm", "args", {"a": "1"})

    assert store.scope("engines", "sparky/vllm")["args"] == {"a": "1"}


CORRUPT_NOTES_SHAPES = ["oops-a-string", ["x", "y"], 42]


@pytest.mark.parametrize("corrupt_notes", CORRUPT_NOTES_SHAPES)
def test_corrupt_notes_field_heals_to_empty_dict(tmp_path, corrupt_notes):
    """Same posture as the namespace heal above, applied to `notes`: a
    scope entry's notes field that is not a dict resets to {} instead of
    leaking a bare value out of get()/scope()."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "engines": {"sparky/vllm": {"args": {"a": "1"}, "notes": corrupt_notes}}
    }))
    store = SettingsStore(path)

    scope = store.scope("engines", "sparky/vllm")
    assert scope["notes"] == {}
    assert scope["args"] == {"a": "1"}


@pytest.mark.parametrize("corrupt_notes", CORRUPT_NOTES_SHAPES)
def test_put_with_note_succeeds_after_corrupt_notes_heals(tmp_path, corrupt_notes):
    """note= is a first-party exercised parameter of put() — a corrupt
    notes field must not crash it with an uncaught TypeError (setdefault()
    returning the corrupt present value, subscript-assigned into: a str,
    list, or int none support item assignment)."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"engines": {"sparky/vllm": {"notes": corrupt_notes}}}))
    store = SettingsStore(path)

    store.put("engines", "sparky/vllm", "args", {"a": "1"}, note="keep me")

    assert store.scope("engines", "sparky/vllm")["notes"]["args"] == "keep me"


def test_normalization_is_scoped_to_the_args_namespace(tmp_path):
    """env and container are not argv — an int env value has nothing to do
    with app.argline's rules and is stored as given, unnormalized."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "env", {"COUNT": 5})

    assert store.scope("engines", "sparky/vllm")["env"]["COUNT"] == 5


# ===========================================================================
# restore() — bulk replace (Task 9: sets snapshot the whole settings store)
# ===========================================================================


def test_restore_replaces_the_whole_store(tmp_path):
    """A bulk restore is a real REPLACE, not a merge — whatever was there
    before (sparky/vllm here) is gone if the restored data doesn't carry it."""
    store = SettingsStore(tmp_path / "s.json")
    store.put("engines", "sparky/vllm", "args", {"old": "1"})

    store.restore({
        "engines": {"autarch/lemonade": {"args": {"new": "2"}}},
        "models": {}, "engine_models": {},
    })

    assert "sparky/vllm" not in store.get()["engines"]
    assert store.scope("engines", "autarch/lemonade")["args"] == {"new": "2"}


def test_restore_stamps_fresh_per_namespace_updated_ts(tmp_path):
    """A restore is a write: the namespace's updated_ts is stamped NOW, not
    carried over from whatever the snapshot recorded — settings_drift may
    honestly flag it, even though nothing here reloads the running engine
    (reload stays human)."""
    store = SettingsStore(tmp_path / "s.json")

    store.restore({
        "engines": {"sparky/vllm": {
            "args": {"x": "1"},
            "updated_ts": {"args": "2020-01-01T00:00:00+00:00"},
        }},
        "models": {}, "engine_models": {},
    })

    ts = store.scope("engines", "sparky/vllm")["updated_ts"]["args"]
    assert ts != "2020-01-01T00:00:00+00:00"


def test_restore_only_stamps_namespaces_actually_present(tmp_path):
    """An entry that never carried an env/container namespace doesn't grow
    a fabricated timestamp for one it never had."""
    store = SettingsStore(tmp_path / "s.json")

    store.restore({
        "engines": {"sparky/vllm": {"args": {"x": "1"}}},
        "models": {}, "engine_models": {},
    })

    ts = store.scope("engines", "sparky/vllm")["updated_ts"]
    assert set(ts) == {"args"}


def test_restore_renormalizes_args(tmp_path):
    """A snapshot captured before a normalization rule changed (or one that
    simply rode in with a raw shape) gets the same normalize_args_map pass
    put() applies on write — a singleton list collapses to its scalar and
    becomes a string."""
    store = SettingsStore(tmp_path / "s.json")

    store.restore({
        "engines": {"sparky/vllm": {"args": {"flag": [1]}}},
        "models": {}, "engine_models": {},
    })

    assert store.scope("engines", "sparky/vllm")["args"]["flag"] == "1"


def test_restore_heals_corrupt_entries_like_a_file_load(tmp_path):
    """restore() validates through the same _load-style healing a corrupt
    file read gets — a non-dict scope entry resets to {} instead of
    raising or leaking a non-dict value out through get()/scope()."""
    store = SettingsStore(tmp_path / "s.json")

    store.restore({
        "engines": {"sparky/vllm": "not-a-dict"},
        "models": {}, "engine_models": {},
    })

    assert store.scope("engines", "sparky/vllm") == {}


def test_restore_preserves_notes(tmp_path):
    """Human rationale in notes is data the snapshot carries, not a
    write-timestamp — restore must not discard or reset it."""
    store = SettingsStore(tmp_path / "s.json")

    store.restore({
        "engines": {"sparky/vllm": {
            "args": {"x": "1"}, "notes": {"args": "no --quantization: breaks the load"},
        }},
        "models": {}, "engine_models": {},
    })

    assert store.scope("engines", "sparky/vllm")["notes"]["args"] == (
        "no --quantization: breaks the load"
    )


def test_restore_is_atomic_no_temp_files_left(tmp_path):
    store = SettingsStore(tmp_path / "s.json")

    store.restore({"engines": {}, "models": {}, "engine_models": {}})

    assert list(tmp_path.glob("*.tmp")) == []


# ===========================================================================
# Journal and remove parameter (Task 1, Phase 3)
# ===========================================================================


def test_put_journals_old_and_new(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.put("engines", "sparky/vllm", "args", {"max-model-len": "131072"})
    store.put("engines", "sparky/vllm", "args", {"max-model-len": "262144"})
    journal = store.get()["engines"]["sparky/vllm"]["journal"]["args"]
    assert journal[0]["old"] is None and journal[0]["new"] == "131072"
    assert journal[1]["old"] == "131072" and journal[1]["new"] == "262144"
    assert all(e["ts"] for e in journal)


def test_put_remove_deletes_key_and_journals(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.put("engines", "sparky/vllm", "args", {"quantization": "q4_k_m"})
    store.put("engines", "sparky/vllm", "args", {}, remove=["quantization"])
    entry = store.get()["engines"]["sparky/vllm"]
    assert "quantization" not in entry["args"]
    last = entry["journal"]["args"][-1]
    assert last["old"] == "q4_k_m" and last["new"] is None


def test_put_remove_and_set_same_key_refused(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    with pytest.raises(ValueError):
        store.put("engines", "sparky/vllm", "args",
                  {"seed": "1"}, remove=["seed"])


def test_unchanged_value_appends_no_journal_entry(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.put("engines", "sparky/vllm", "args", {"seed": "1"})
    store.put("engines", "sparky/vllm", "args", {"seed": "1"})
    assert len(store.get()["engines"]["sparky/vllm"]["journal"]["args"]) == 1


def test_journal_capped(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    for i in range(60):
        store.put("engines", "sparky/vllm", "args", {"seed": str(i)})
    assert len(store.get()["engines"]["sparky/vllm"]["journal"]["args"]) == 50


def test_corrupt_journal_heals(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"engines": {"sparky/vllm": {
        "args": {"seed": "1"}, "journal": "corrupt"}},
        "models": {}, "engine_models": {}}))
    store = SettingsStore(path)
    assert store.get()["engines"]["sparky/vllm"]["journal"] == {}


def test_corrupt_per_namespace_journal_heals(tmp_path):
    """A journal dict that contains a corrupt per-namespace value (not a list)
    must be healed to an empty list to prevent AttributeError on log.extend()."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"engines": {"sparky/vllm": {
        "args": {"seed": "1"}, "journal": {"args": "corrupt"}}},
        "models": {}, "engine_models": {}}))
    store = SettingsStore(path)

    # This put() should not crash when trying to extend a corrupt journal namespace
    store.put("engines", "sparky/vllm", "args", {"seed": "2"})

    entry = store.get()["engines"]["sparky/vllm"]
    # The journal namespace should be a proper list with the new entry
    assert isinstance(entry["journal"]["args"], list)
    assert len(entry["journal"]["args"]) == 1
    assert entry["journal"]["args"][0]["old"] == "1"
    assert entry["journal"]["args"][0]["new"] == "2"


def test_restore_merges_journals_not_replaces(tmp_path):
    """Restore must merge journals with previous entries, not replace them.
    A put that changed k=A→B followed by restore back to k=A must preserve
    both the pre-restore entries AND the restore's B→A entry in order."""
    store = SettingsStore(tmp_path / "settings.json")

    # Initial put: k=A (journals None→A)
    store.put("engines", "sparky/vllm", "args", {"k": "A"})
    # Second put: k=B (journals A→B)
    store.put("engines", "sparky/vllm", "args", {"k": "B"})

    # Take a snapshot and modify it back to A
    snapshot = store.get()
    snapshot["engines"]["sparky/vllm"]["args"]["k"] = "A"

    # Restore should merge journals, not replace
    store.restore(snapshot)

    journal = store.get()["engines"]["sparky/vllm"]["journal"]["args"]
    # Should have 3 entries: None→A, A→B, B→A
    assert len(journal) == 3
    assert journal[0]["old"] is None and journal[0]["new"] == "A"
    assert journal[1]["old"] == "A" and journal[1]["new"] == "B"
    assert journal[2]["old"] == "B" and journal[2]["new"] == "A"


def test_restore_journals_diff_against_previous(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.put("engines", "sparky/vllm", "args", {"max-model-len": "262144"})
    snapshot = store.get()
    snapshot["engines"]["sparky/vllm"]["args"]["max-model-len"] = "131072"
    store.restore(snapshot)
    last = store.get()["engines"]["sparky/vllm"]["journal"]["args"][-1]
    assert last["old"] == "262144" and last["new"] == "131072"
