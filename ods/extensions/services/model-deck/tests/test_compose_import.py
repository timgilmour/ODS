"""Tests for app.compose_import — real compose files -> settings.

Fixtures are the SEVEN ACTUAL sparky profiles captured live 2026-08-07
(~/notes/evidence/2026-08-07-spark-profiles/). Invented fixtures would not
contain a six-valued flag, a JSON blob argument, a comment explaining why a
flag is absent, or a non-vLLM profile — precisely the cases that break
naive importers.
"""

from pathlib import Path

import pytest

from app.argline import POSITIONAL_KEY
from app.compose_import import import_compose

FIXTURES = Path(__file__).parent / "fixtures" / "spark-profiles"


def _profile(name):
    return (FIXTURES / f"compose-{name}.yaml").read_text()


def test_heretic_args_imported():
    imported = import_compose(_profile("heretic"))

    assert imported["args"]["max-model-len"] == "262144"
    assert imported["args"]["kv-cache-dtype"] == "fp8_e4m3"


def test_bare_flags_import_as_true():
    imported = import_compose(_profile("heretic"))

    assert imported["args"]["enable-chunked-prefill"] is True
    assert imported["args"]["enable-prefix-caching"] is True


def test_positional_serve_model_is_preserved():
    imported = import_compose(_profile("heretic"))

    assert imported["args"][POSITIONAL_KEY] == ["serve", "/model"]


def test_multi_valued_served_model_name_imports_as_a_list():
    """mm27b passes SIX names (verified live 2026-08-07). Importing only the
    first would silently change what the engine answers to."""
    imported = import_compose(_profile("mm27b"))

    assert imported["args"]["served-model-name"] == [
        "aeon", "aeon-fast", "aeon-deep", "aeon-ultimate",
        "qwen36-ultimate", "aeon-ultimate-xs",
    ]


def test_identity_derived_from_the_model_mount():
    """The checkpoint directory name lives ONLY in the mount — this field is
    what lets anything build the engine_models/<node>/vllm|<identity> key."""
    imported = import_compose(_profile("heretic"))

    assert imported["identity"] == "Qwen3.6-35B-A3B-heretic-NVFP4"


def test_service_and_container_name_extracted():
    imported = import_compose(_profile("heretic"))

    assert imported["service"] == "aeon-vllm"
    assert imported["container_name"] == "aeon-vllm"


def test_environment_imported():
    imported = import_compose(_profile("heretic"))

    assert imported["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_container_allowlist_fields_only():
    imported = import_compose(_profile("heretic"))

    assert imported["container"]["image"]
    assert "volumes" not in imported["container"]


def test_comment_explaining_an_absent_flag_is_kept_as_a_note():
    """compose-heretic.yaml explains why --quantization is ABSENT (the
    2026-07-31 modelopt crash-loop fix). Discarding that comment would lose
    the only written record."""
    imported = import_compose(_profile("heretic"))

    assert "modelopt" in imported["notes"]["args"]


def test_ds4_profile_imports_without_vllm_assumptions():
    """ds4 is NOT vLLM (DwarfStar 4; verified live 2026-08-07). It has no
    /model mount and no `serve` positional — it must import cleanly."""
    imported = import_compose(_profile("ds4"))

    assert imported["identity"] is None
    assert isinstance(imported["args"], dict)
    assert imported["container_name"] == "spark-ds4"


def test_comfyui_profile_imports_without_vllm_assumptions():
    imported = import_compose(_profile("comfyui"))

    assert imported["identity"] is None
    assert imported["container"]["image"]


def test_import_is_idempotent():
    text = _profile("heretic")

    assert import_compose(text) == import_compose(text)


def test_malformed_yaml_raises_value_error():
    with pytest.raises(ValueError):
        import_compose("services: [unclosed")
