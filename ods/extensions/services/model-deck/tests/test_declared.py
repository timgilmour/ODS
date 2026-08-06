"""Tests for app.declared — the small human-owned fact allowlist.

The allowlist IS the enforcement of derive-don't-duplicate. If a human can
declare a fact the machine can read, the two will disagree eventually and
the whole characteristics layer becomes decoration.
"""

import pytest

from app.declared import ALLOWED_FIELDS, DeclaredStore


def test_allowlist_is_exactly_these_fields():
    """Deliberately tiny. Growing it is a design decision, not a shortcut —
    hence a test that fails when someone adds a field quietly."""
    assert set(ALLOWED_FIELDS) == {"tools_verified", "label", "notes", "tags", "engine_preference"}


def test_put_accepts_allowed_fields(tmp_path):
    store = DeclaredStore(tmp_path / "d.json")

    store.put("model/Qwen3.6-35B-A3B-heretic-NVFP4", {"tools_verified": True})

    assert store.entry("model/Qwen3.6-35B-A3B-heretic-NVFP4")["tools_verified"] is True


def test_put_rejects_a_derivable_field(tmp_path):
    """max_model_len is readable from the checkpoint. Declaring it is
    exactly the duplication this layer exists to prevent."""
    store = DeclaredStore(tmp_path / "d.json")

    with pytest.raises(ValueError, match="max_model_len"):
        store.put("model/m", {"max_model_len": 262144})


def test_put_merges_with_existing_fields(tmp_path):
    store = DeclaredStore(tmp_path / "d.json")
    store.put("model/m", {"tools_verified": True})

    store.put("model/m", {"label": "the good one"})

    assert store.entry("model/m") == {"tools_verified": True, "label": "the good one"}


def test_tags_accepts_a_list_of_strings(tmp_path):
    """Role names like 'fast'/'deep' become tags — this is where the retired
    aliases land. The taxonomy itself is a later increment."""
    store = DeclaredStore(tmp_path / "d.json")

    store.put("model/m", {"tags": ["fast", "tool-capable"]})

    assert store.entry("model/m")["tags"] == ["fast", "tool-capable"]


def test_tags_rejects_a_bare_string(tmp_path):
    store = DeclaredStore(tmp_path / "d.json")

    with pytest.raises(ValueError):
        store.put("model/m", {"tags": "fast"})


def test_tools_verified_must_be_bool(tmp_path):
    store = DeclaredStore(tmp_path / "d.json")

    with pytest.raises(ValueError):
        store.put("model/m", {"tools_verified": "yes"})


def test_get_empty_when_missing_or_corrupt(tmp_path):
    assert DeclaredStore(tmp_path / "d.json").get() == {}
    path = tmp_path / "d2.json"
    path.write_text("nope")
    assert DeclaredStore(path).get() == {}


def test_rejected_put_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "d.json"
    store = DeclaredStore(path)
    store.put("model/m", {"label": "keep me"})

    with pytest.raises(ValueError):
        store.put("model/m", {"label": "lost", "max_model_len": 1})

    assert store.entry("model/m") == {"label": "keep me"}


def test_forget_removes_entry(tmp_path):
    store = DeclaredStore(tmp_path / "d.json")
    store.put("model/m", {"label": "x"})

    store.forget("model/m")

    assert store.get() == {}
