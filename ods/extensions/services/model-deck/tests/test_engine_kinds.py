"""Declaration validation (E1 spec §1). Kinds and their connection
schemas live in app/engine_kinds.py — the single module allowed to
know engine names (spec §8)."""

import pytest

from app.engine_kinds import KNOWN_KINDS, validate_engines


def _entry(**over):
    # Fixture rule: resource deliberately NOT an engine name, GPU not 0/1.
    e = {"resource": "gguf-a", "kind": "lemonade",
         "connection": {"url": "http://gguf-a:8080",
                        "metrics_url": "http://gguf-a:8001/metrics",
                        "container": "ods-gguf-a"},
         "gpu_index": 3,
         "policy_defaults": {"priority": 10, "pinned": False, "idle_ttl": 60}}
    e.update(over)
    return e


def test_known_kinds_are_exactly_the_three_adapters():
    assert set(KNOWN_KINDS) == {"lemonade", "comfyui", "hipfire"}


def test_valid_list_passes():
    validate_engines([_entry()])


def test_unknown_kind_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        validate_engines([_entry(kind="vllm")])


def test_missing_required_connection_field_refused():
    with pytest.raises(ValueError, match="url"):
        validate_engines([_entry(connection={"container": "c"})])


def test_unknown_connection_field_refused():
    with pytest.raises(ValueError, match="extra"):
        validate_engines([_entry(connection={"url": "http://x",
                                             "metrics_url": "http://x/m",
                                             "container": "c",
                                             "port": 9})])


def test_duplicate_resource_refused():
    with pytest.raises(ValueError, match="duplicate"):
        validate_engines([_entry(), _entry()])


def test_gpu_index_must_be_nonnegative_int():
    with pytest.raises(ValueError, match="gpu_index"):
        validate_engines([_entry(gpu_index=True)])   # bool is not an int here
    with pytest.raises(ValueError, match="gpu_index"):
        validate_engines([_entry(gpu_index=-1)])


def test_policy_defaults_exact_fields():
    with pytest.raises(ValueError, match="policy_defaults"):
        validate_engines([_entry(policy_defaults={"priority": 1})])


def test_resource_shape_refused_when_slashy():
    # resource keys build "local/<resource>" lifecycle keys — a slash
    # would forge a foreign key ([[literal-declared-inputs]]).
    with pytest.raises(ValueError, match="resource"):
        validate_engines([_entry(resource="local/evil")])
