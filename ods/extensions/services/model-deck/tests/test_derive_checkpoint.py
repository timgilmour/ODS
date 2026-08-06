"""Tests for app.derive_checkpoint — checkpoint dir -> derived facts.

Plain JSON reads only: no torch, no safetensors parsing, no weight loading.
A characteristics refresh must be cheap enough to run on a timer.
"""

import json

from app.derive_checkpoint import derive_checkpoint


def _checkpoint(tmp_path, name="Qwen3.6-35B-A3B-heretic-NVFP4", config=None, gen=None):
    d = tmp_path / name
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    if gen is not None:
        (d / "generation_config.json").write_text(json.dumps(gen))
    return d


def test_identity_is_the_directory_name_verbatim(tmp_path):
    """The naming rule: identity is what is on disk, not an alias, not a
    normalization. 'aeon' is not a model."""
    d = _checkpoint(tmp_path, config={})

    facts = derive_checkpoint(d, now="t")

    assert facts["identity"]["value"] == "Qwen3.6-35B-A3B-heretic-NVFP4"
    assert facts["identity"]["source"] == "directory name"


def test_quant_method_read_from_config(tmp_path):
    d = _checkpoint(tmp_path, config={"quantization_config": {"quant_method": "compressed-tensors"}})

    facts = derive_checkpoint(d, now="t")

    assert facts["quant_method"]["value"] == "compressed-tensors"
    assert facts["quant_method"]["source"] == "config.json"


def test_unquantized_checkpoint_has_no_quant_method_field(tmp_path):
    """Absent, not None: a field that is missing means 'cannot check', and
    detect_drift relies on that distinction."""
    d = _checkpoint(tmp_path, config={"max_position_embeddings": 4096})

    facts = derive_checkpoint(d, now="t")

    assert "quant_method" not in facts


def test_context_capability_read_from_config(tmp_path):
    d = _checkpoint(tmp_path, config={"max_position_embeddings": 262144})

    facts = derive_checkpoint(d, now="t")

    assert facts["max_position_embeddings"]["value"] == 262144


def test_architecture_read_from_config(tmp_path):
    d = _checkpoint(tmp_path, config={"architectures": ["Qwen3MoeForCausalLM"]})

    facts = derive_checkpoint(d, now="t")

    assert facts["architecture"]["value"] == "Qwen3MoeForCausalLM"


def test_recommended_sampling_read_from_generation_config(tmp_path):
    """The values --generation-config vllm currently discards."""
    d = _checkpoint(tmp_path, config={}, gen={"temperature": 0.7, "top_p": 0.95, "top_k": 20})

    facts = derive_checkpoint(d, now="t")

    assert facts["recommended_sampling"]["value"] == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 20,
    }
    assert facts["recommended_sampling"]["source"] == "generation_config.json"


def test_generation_config_absent_yields_no_sampling_field(tmp_path):
    d = _checkpoint(tmp_path, config={})

    assert "recommended_sampling" not in derive_checkpoint(d, now="t")


def test_chat_template_detected_from_jinja_file(tmp_path):
    d = _checkpoint(tmp_path, config={})
    (d / "chat_template.jinja").write_text("{{ x }}")

    assert derive_checkpoint(d, now="t")["chat_template_present"]["value"] is True


def test_chat_template_detected_from_tokenizer_config(tmp_path):
    d = _checkpoint(tmp_path, config={})
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))

    assert derive_checkpoint(d, now="t")["chat_template_present"]["value"] is True


def test_corrupt_config_does_not_raise(tmp_path):
    """A half-downloaded checkpoint must degrade to fewer facts, never take
    down the derive pass for every other model."""
    d = tmp_path / "broken"
    d.mkdir()
    (d / "config.json").write_text("{{{")

    facts = derive_checkpoint(d, now="t")

    assert facts["identity"]["value"] == "broken"
    assert "quant_method" not in facts


def test_missing_directory_yields_only_nothing(tmp_path):
    assert derive_checkpoint(tmp_path / "ghost", now="t") == {}


def test_every_field_is_stamped_with_the_given_now(tmp_path):
    d = _checkpoint(tmp_path, config={"max_position_embeddings": 1})

    facts = derive_checkpoint(d, now="2026-08-04T00:00:00+00:00")

    assert all(f["derived_ts"] == "2026-08-04T00:00:00+00:00" for f in facts.values())
