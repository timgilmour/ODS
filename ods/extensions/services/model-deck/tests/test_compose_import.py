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


def _with_environment(profile: str, block: str) -> str:
    """A real fixture with ONLY its `environment:` block swapped — the rest
    of the file (command block, mounts, comment) stays live truth, so a
    variant test still exercises the whole import, not a hand-shaped stub.

    compose-heretic.yaml's block is two lines (10-11): the header and its
    single mapping entry.
    """
    text = _profile(profile)
    old = "    environment:\n      VLLM_USE_FLASHINFER_SAMPLER: \"1\"\n"
    assert old in text, "fixture's environment block moved; update this helper"
    return text.replace(old, block)


def test_list_form_environment_imports():
    """`environment:` as a LIST (`["FOO=bar"]`) is the other half of
    compose's own schema and is at least as common as the mapping form.
    Before the fix, `.items()` raised AttributeError — NOT a ValueError, so
    it escaped adopt's `(ValueError, EngineError)` isolation and 500'd the
    sweep after earlier profiles' writes had committed."""
    imported = import_compose(_with_environment("heretic", (
        "    environment:\n"
        "      - VLLM_USE_FLASHINFER_SAMPLER=1\n"
        "      - VLLM_LOGGING_LEVEL=DEBUG\n"
    )))

    assert imported["env"] == {"VLLM_USE_FLASHINFER_SAMPLER": "1",
                               "VLLM_LOGGING_LEVEL": "DEBUG"}
    # The rest of the import is unaffected by the environment encoding.
    assert imported["args"]["kv-cache-dtype"] == "fp8_e4m3"


def test_list_form_environment_splits_on_the_first_equals_only():
    """A value containing `=` (a JSON blob, a base64 token) must survive
    whole: splitting on every `=` would truncate it silently."""
    imported = import_compose(_with_environment("heretic", (
        "    environment:\n"
        '      - VLLM_ATTENTION_CONFIG={"a":1,"b":"x=y"}\n'
    )))

    assert imported["env"]["VLLM_ATTENTION_CONFIG"] == '{"a":1,"b":"x=y"}'


def test_list_form_entry_without_equals_imports_as_empty():
    """`- FOO` is host-passthrough in compose; the Deck has no host
    environment to resolve it from at import time, so it imports as the
    empty value rather than being dropped (dropping it would lose the
    operator's only record that the variable is meant to be set)."""
    imported = import_compose(_with_environment("heretic",
                                                "    environment:\n      - FOO\n"))

    assert imported["env"] == {"FOO": ""}


def test_environment_of_an_unsupported_shape_raises_value_error():
    """Neither mapping nor list -> ValueError naming the type, NOT an
    AttributeError: adopt isolates ValueError per profile, and any other
    exception class escapes that isolation."""
    with pytest.raises(ValueError, match="str"):
        import_compose(_with_environment("heretic",
                                         "    environment: VLLM_LOGGING_LEVEL=DEBUG\n"))


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
