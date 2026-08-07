"""Tests for app.settings_store — the human-owned configuration store.

Three scopes (engine, model, engine x model) x three namespaces (args, env,
container). Namespaces exist because a real profile sets far more than
flags: VLLM_USE_FLASHINFER_SAMPLER=1, a pinned image digest, shm_size 16g.
"""

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


def test_empty_list_arg_value_warns_and_normalizes_to_true(tmp_path):
    """Edge case ruled to be handled at this layer: an empty list has no
    argline representation — it renders as a bare flag and reparses as
    True (see app.argline module docstring). This store's posture is
    warn-and-normalize, not reject: the container allowlist is the one
    hard validation failure here, everything else about a value's shape
    is fixed up rather than blocking the save."""
    store = SettingsStore(tmp_path / "s.json")

    with pytest.warns(UserWarning, match="empty list"):
        store.put("engines", "sparky/vllm", "args", {"served-model-name": []})

    assert store.scope("engines", "sparky/vllm")["args"]["served-model-name"] is True


def test_normalization_is_scoped_to_the_args_namespace(tmp_path):
    """env and container are not argv — an int env value has nothing to do
    with app.argline's rules and is stored as given, unnormalized."""
    store = SettingsStore(tmp_path / "s.json")

    store.put("engines", "sparky/vllm", "env", {"COUNT": 5})

    assert store.scope("engines", "sparky/vllm")["env"]["COUNT"] == 5
