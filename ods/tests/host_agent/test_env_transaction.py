"""Unit + integration tests for the host-agent key-scoped .env rollback.

The agent lives in ``ods/bin/ods-host-agent.py`` — a hyphenated filename that
cannot be imported by name, and a module whose server start and ``main()`` are
gated behind ``if __name__ == "__main__":``. Module level only defines
constants/helpers/classes (``INSTALL_DIR``/``DATA_DIR`` default to empty
``Path()``), so loading it via importlib has no side effects.

Run:  python3 -m pytest ods/tests/host_agent/ -v
"""

import importlib.util
from pathlib import Path

import pytest

_AGENT_PATH = Path(__file__).resolve().parents[2] / "bin" / "ods-host-agent.py"


def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "ods_host_agent_under_test", _AGENT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load_agent()


def _seed(tmp_path: Path, text: str) -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(text, encoding="utf-8")
    return env_path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 1 — _EnvTransaction primitive
# ---------------------------------------------------------------------------


def test_update_then_rollback_restores_priors_and_concurrent_edit_survives(tmp_path):
    """The incident in miniature: an unrelated edit made BETWEEN update and
    rollback must survive; only the written keys revert."""
    env_path = _seed(tmp_path, "A=1\nB=2\nUNRELATED=old\n")
    txn = agent._EnvTransaction(env_path)

    txn.update({"A": "10", "B": "20"})
    assert txn.dirty is True
    assert txn.written_keys() == {"A", "B"}
    # activation wrote its keys
    assert "A=10" in _read(env_path)
    assert "B=20" in _read(env_path)

    # Concurrent operator fix to a line OUTSIDE the write-set.
    agent._update_env_keys(env_path, {"UNRELATED": "fixed"})

    restored = txn.rollback()

    text = _read(env_path)
    assert "A=1" in text
    assert "B=2" in text
    # The concurrent fix survives — rollback did NOT clobber it.
    assert "UNRELATED=fixed" in text
    assert "UNRELATED=old" not in text
    assert restored == {"A": ("1", "10"), "B": ("2", "20")}


def test_absent_key_is_deleted_on_rollback(tmp_path):
    env_path = _seed(tmp_path, "A=1\n")
    txn = agent._EnvTransaction(env_path)

    txn.update({"NEW": "x"})
    assert "NEW=x" in _read(env_path)

    restored = txn.rollback()

    text = _read(env_path)
    assert "NEW" not in text
    assert "A=1" in text
    assert restored == {"NEW": (None, "x")}


def test_key_deleted_by_concurrent_edit_but_in_write_set_is_restored(tmp_path):
    env_path = _seed(tmp_path, "A=1\nB=2\n")
    txn = agent._EnvTransaction(env_path)

    txn.update({"A": "10"})
    # Concurrent edit removes the A line entirely.
    env_path.write_text("B=2\n", encoding="utf-8")

    restored = txn.rollback()

    text = _read(env_path)
    assert "A=1" in text
    assert "B=2" in text
    assert restored == {"A": ("1", None)}


def test_two_updates_same_key_rolls_back_to_original(tmp_path):
    env_path = _seed(tmp_path, "A=1\n")
    txn = agent._EnvTransaction(env_path)

    txn.update({"A": "10"})
    txn.update({"A": "20"})  # first-seen prior (1) must win
    assert "A=20" in _read(env_path)

    restored = txn.rollback()

    assert "A=1" in _read(env_path)
    assert restored == {"A": ("1", "20")}


def test_rollback_with_no_updates_is_byte_identical_noop(tmp_path):
    original = "# header\nA=1\nB=2\n\nC=3\n"
    env_path = _seed(tmp_path, original)
    before = env_path.read_bytes()

    txn = agent._EnvTransaction(env_path)
    assert txn.dirty is False
    assert txn.written_keys() == set()

    restored = txn.rollback()

    assert restored == {}
    assert env_path.read_bytes() == before  # byte-identical, untouched


def test_comments_and_unknown_lines_and_order_preserved(tmp_path):
    original = "# comment\nA=1\nRANDOM line without equals\nB=2\n\nC=3\n"
    env_path = _seed(tmp_path, original)

    txn = agent._EnvTransaction(env_path)
    txn.update({"B": "20"})

    restored = txn.rollback()

    # Only B was touched; every other line (comment, non-kv line, blank,
    # ordering) survives byte-for-byte.
    assert _read(env_path) == original
    assert restored == {"B": ("2", "20")}


def test_record_captures_priors_for_externally_written_keys(tmp_path):
    """The GGUF activate block performs its own write (upsert + delete) that
    _update_env_keys cannot express; it uses record() so rollback restores it."""
    env_path = _seed(tmp_path, "A=1\nDROP=keep\n")
    txn = agent._EnvTransaction(env_path)

    # Simulate the closure: record the mutation set, then write externally.
    txn.record({"A", "NEW", "DROP"})
    env_path.write_text("A=10\nNEW=x\n", encoding="utf-8")  # A changed, DROP deleted, NEW added

    assert txn.written_keys() == {"A", "NEW", "DROP"}

    restored = txn.rollback()

    text = _read(env_path)
    assert "A=1" in text
    assert "DROP=keep" in text  # deleted-by-activation key restored
    assert "NEW" not in text  # added-by-activation key removed
    assert restored == {"A": ("1", "10"), "NEW": (None, "x"), "DROP": ("keep", None)}


# ---------------------------------------------------------------------------
# Task 3 — incident replay: a concurrent .env fix must survive rollback
#
# 2026-07-21 live incident: a stale LLAMA_SERVER_GPU_INDICES=0,1 line — OUTSIDE
# the activation's write-set — caused activation to fail. The operator fixed
# that line mid-activation; the old wholesale snapshot restore then rolled the
# whole file back and resurrected the stale 0,1. These tests drive the two
# activate closures the way they now drive the transaction and prove the fix.
# ---------------------------------------------------------------------------


def _apply_gguf_inline_write(env_path: Path, updates: dict, remove_keys: set) -> None:
    """Byte-for-byte replica of _do_model_activate's inline .env write:
    upsert every key in ``updates`` and delete every key in ``remove_keys``."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    new_lines = []
    seen = set()
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else None
        if key and key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        elif key and key in remove_keys:
            continue
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _parse_env(text: str) -> dict:
    return dict(
        line.split("=", 1)
        for line in text.splitlines()
        if "=" in line and not line.startswith("#")
    )


def test_incident_replay_gguf_activation_concurrent_fix_survives(tmp_path):
    # Seed the .env as the failing GGUF activation saw it: route keys, a stale
    # spec-decode arg the activation drops, the stale failure-causing GPU line
    # OUTSIDE the write-set, and an unrelated line.
    env_path = _seed(
        tmp_path,
        "# ODS environment\n"
        "GGUF_FILE=old-model.gguf\n"
        "LLM_MODEL=old-model\n"
        "CTX_SIZE=8192\n"
        "LLAMA_ARG_SPEC_TYPE=draft\n"       # stale key the activation deletes
        "LLAMA_SERVER_GPU_INDICES=0,1\n"    # stale, failure-causing line
        "UNTOUCHED=keep-me\n",
    )
    txn = agent._EnvTransaction(env_path)

    updates = {"GGUF_FILE": "new-model.gguf", "LLM_MODEL": "new-model", "CTX_SIZE": "16384"}
    remove_keys = {"LLAMA_ARG_SPEC_TYPE"}

    # Closure step: record the mutation set, then perform the inline write.
    txn.record(set(updates) | remove_keys)
    _apply_gguf_inline_write(env_path, updates, remove_keys)

    # Mid-activation warm-up, the operator fixes the stale line via the
    # /v1/env/set-keys endpoint (which calls _update_env_keys directly).
    agent._update_env_keys(env_path, {"LLAMA_SERVER_GPU_INDICES": "1"})

    # Health gate fails → rollback.
    restored = txn.rollback()

    route = _parse_env(_read(env_path))
    # Route keys reverted to priors; the deleted stale key came back.
    assert route["GGUF_FILE"] == "old-model.gguf"
    assert route["LLM_MODEL"] == "old-model"
    assert route["CTX_SIZE"] == "8192"
    assert route["LLAMA_ARG_SPEC_TYPE"] == "draft"
    # THE FIX: the operator's concurrent correction SURVIVES — the stale 0,1
    # the old wholesale restore would have resurrected is gone for good.
    assert route["LLAMA_SERVER_GPU_INDICES"] == "1"
    assert route["UNTOUCHED"] == "keep-me"
    # Restored-pairs dict names ONLY the write-set, never the concurrent fix.
    assert set(restored) == set(updates) | remove_keys
    assert "LLAMA_SERVER_GPU_INDICES" not in restored


def test_incident_replay_hipfire_activation_concurrent_fix_survives(tmp_path):
    env_path = _seed(
        tmp_path,
        "HIPFIRE_MODEL=old.mq4\n"
        "HIPFIRE_ACTIVE=false\n"
        "LLAMA_SERVER_GPU_INDICES=0,1\n"    # stale line outside the write-set
        "UNTOUCHED=keep-me\n",
    )
    txn = agent._EnvTransaction(env_path)

    # Closure step: _do_hipfire_activate routes its write through update().
    txn.update({"HIPFIRE_MODEL": "new.mq4", "HIPFIRE_ACTIVE": "true"})
    assert _parse_env(_read(env_path))["HIPFIRE_MODEL"] == "new.mq4"

    # Concurrent operator fix to the unrelated stale line.
    agent._update_env_keys(env_path, {"LLAMA_SERVER_GPU_INDICES": "1"})

    restored = txn.rollback()

    route = _parse_env(_read(env_path))
    assert route["HIPFIRE_MODEL"] == "old.mq4"
    assert route["HIPFIRE_ACTIVE"] == "false"
    assert route["LLAMA_SERVER_GPU_INDICES"] == "1"   # the fix survives
    assert route["UNTOUCHED"] == "keep-me"
    assert set(restored) == {"HIPFIRE_MODEL", "HIPFIRE_ACTIVE"}
    assert "LLAMA_SERVER_GPU_INDICES" not in restored
