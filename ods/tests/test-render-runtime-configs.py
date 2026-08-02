#!/usr/bin/env python3
"""Tests for scripts/render-runtime-configs.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render-runtime-configs.py"


def load_renderer_module() -> ModuleType:
    name = "ods_render_runtime_configs_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_renderer(*args: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def file_by_surface(payload: dict[str, object], surface: str) -> dict[str, str]:
    for item in payload["files"]:
        if item["surface"] == surface:
            return item
    raise AssertionError(f"missing surface {surface}")


def model_provider_by_id(settings: dict[str, object], provider_id: str) -> dict[str, object]:
    for provider in settings["modelProviders"]:
        if provider["id"] == provider_id:
            return provider
    raise AssertionError(f"missing model provider {provider_id}")


def test_all_surfaces_render() -> None:
    payload = run_renderer("--surface", "all")
    surfaces = {item["surface"] for item in payload["files"]}
    assert surfaces == {
        "env", "opencode", "litellm-local", "perplexica", "hermes",
        "model-router-endpoints",
    }
    assert payload["mode"] == "dry-run"


def test_switchboard_surface_gated_on_enabled_mode() -> None:
    observed = run_renderer("--surface", "all", "--switchboard-mode", "observe")
    assert "litellm-switchboard" not in {i["surface"] for i in observed["files"]}
    enabled = run_renderer("--surface", "all", "--switchboard-mode", "enabled")
    surfaces = {i["surface"] for i in enabled["files"]}
    assert "litellm-switchboard" in surfaces
    switchboard = next(i for i in enabled["files"] if i["surface"] == "litellm-switchboard")
    assert "model_name: ods/current" in switchboard["content"]
    assert "model_name: local" in switchboard["content"]
    assert "model_name: default" in switchboard["content"]
    assert 'model_name: "*"' in switchboard["content"]
    assert "api_base: http://model-router:9099/v1" in switchboard["content"]
    assert switchboard["content"].count("http://model-router:9099/v1") == 4


def test_all_selects_one_mode_config() -> None:
    expected = {
        "local": "litellm-local",
        "cloud": "litellm-cloud",
        "hybrid": "litellm-hybrid",
        "lemonade": "litellm-lemonade",
    }
    all_mode_surfaces = set(expected.values())
    for mode, expected_surface in expected.items():
        payload = run_renderer("--surface", "all", "--ods-mode", mode)
        surfaces = {item["surface"] for item in payload["files"]}
        assert surfaces & all_mode_surfaces == {expected_surface}


def test_cloud_enabled_never_renders_local_switchboard() -> None:
    payload = run_renderer(
        "--surface",
        "all",
        "--ods-mode",
        "cloud",
        "--switchboard-mode",
        "enabled",
    )
    surfaces = {item["surface"] for item in payload["files"]}
    assert "litellm-cloud" in surfaces
    assert "litellm-switchboard" not in surfaces
    cloud = file_by_surface(payload, "litellm-cloud")["content"]
    assert "model_name: ods/current" in cloud
    assert "model-router" not in cloud


def test_remote_cloud_projection_uses_internal_egress_and_state_receipt() -> None:
    payload = run_renderer(
        "--surface",
        "all",
        "--ods-mode",
        "cloud",
        "--remote-llm-enabled",
        "true",
        "--remote-llm-transport",
        "direct",
        "--remote-llm-base-url",
        "https://gpu.example.test",
        "--remote-llm-model",
        "qwen/remote:latest",
    )
    surfaces = {item["surface"] for item in payload["files"]}
    assert "litellm-cloud" in surfaces
    assert "remote-routing-state" in surfaces
    assert "litellm-switchboard" not in surfaces

    cloud = file_by_surface(payload, "litellm-cloud")["content"]
    assert "model_name: ods/current" in cloud
    assert 'model: "openai/qwen/remote:latest"' in cloud
    assert 'model_name: "qwen/remote:latest"' in cloud
    assert 'api_base: "http://remote-provider-egress:8091/v1"' in cloud
    assert "api_key: not-needed" in cloud
    assert "https://gpu.example.test" not in cloud
    assert "REMOTE_LLM_API_KEY" not in cloud

    env_content = file_by_surface(payload, "env")["content"]
    assert "REMOTE_LLM_ENABLED=true" in env_content
    assert "REMOTE_LLM_TRANSPORT=direct" in env_content
    assert "REMOTE_LLM_BASE_URL=https://gpu.example.test/v1" in env_content
    assert "REMOTE_LLM_MODEL=qwen/remote:latest" in env_content

    state = json.loads(file_by_surface(payload, "remote-routing-state")["content"])
    assert state["schema"] == "ods.remote-routing-state.v1"
    assert state["enabled"] is True
    assert state["mode"] == "cloud"
    assert state["provider"] == {
        "baseUrl": "https://gpu.example.test/v1",
        "capability": "openai-compatible",
        "model": "qwen/remote:latest",
        "transport": "direct",
    }
    assert state["projection"] == {
        "consumerRoute": "gateway",
        "egressBaseUrl": "http://remote-provider-egress:8091/v1",
        "gateway": "litellm-cloud",
        "publicModel": "ods/current",
    }
    assert state["status"] == {
        "proven": False,
        "reason": "pending-provider-handshake",
    }
    assert "key" not in json.dumps(state).lower()


def test_remote_routing_state_disabled_receipt_has_no_provider() -> None:
    payload = run_renderer("--surface", "remote-routing-state")
    state = json.loads(file_by_surface(payload, "remote-routing-state")["content"])
    assert state["enabled"] is False
    assert state["provider"] is None
    assert state["status"] == {"proven": False, "reason": "disabled"}
    assert "REMOTE_LLM_API_KEY" not in json.dumps(state)


def test_remote_projection_requires_cloud_mode() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--surface",
            "all",
            "--ods-mode",
            "local",
            "--remote-llm-enabled",
            "true",
            "--remote-llm-transport",
            "direct",
            "--remote-llm-base-url",
            "https://gpu.example.test/v1",
            "--remote-llm-model",
            "qwen-remote",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2
    assert "ODS_MODE=cloud" in proc.stderr


def test_remote_projection_rejects_unsafe_model_id() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--surface",
            "all",
            "--ods-mode",
            "cloud",
            "--remote-llm-enabled",
            "true",
            "--remote-llm-transport",
            "direct",
            "--remote-llm-base-url",
            "https://gpu.example.test/v1",
            "--remote-llm-model",
            "bad model; touch nope",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2
    assert "model id without spaces" in proc.stderr


def test_explicit_cloud_switchboard_render_fails_closed() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--surface",
            "litellm-switchboard",
            "--ods-mode",
            "cloud",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2
    assert "local-runtime-only" in proc.stderr


def test_native_local_projection_uses_host_route_and_concrete_model() -> None:
    payload = run_renderer(
        "--surface",
        "litellm-local-native",
        "--ods-mode",
        "local",
        "--gguf-file",
        "Native-Model.gguf",
        "--llm-base-url",
        "http://host.docker.internal:13306/v1",
    )
    content = file_by_surface(payload, "litellm-local-native")["content"]
    assert "model: openai/Native-Model.gguf" in content
    assert "api_base: http://host.docker.internal:13306/v1" in content
    assert "enable_thinking: false" in content
    assert "request_timeout: 900" in content
    assert "stream_timeout: 900" in content


def test_checked_in_mode_configs_match_renderer() -> None:
    for mode in ("local", "cloud", "hybrid"):
        payload = run_renderer("--surface", f"litellm-{mode}", "--ods-mode", mode)
        rendered = file_by_surface(payload, f"litellm-{mode}")["content"]
        checked_in = (ROOT / "config" / "litellm" / f"{mode}.yaml").read_text(
            encoding="utf-8"
        )
        assert rendered == checked_in


def test_enabled_env_exports_switchboard_webui_gateway() -> None:
    payload = run_renderer(
        "--surface",
        "env",
        "--switchboard-mode",
        "enabled",
        "--litellm-key",
        "sk-test-litellm",
    )
    content = file_by_surface(payload, "env")["content"]
    assert "ODS_MODEL_SWITCHBOARD=enabled" in content
    assert "OPEN_WEBUI_LLM_BASE_URL=http://litellm:4000" in content
    assert "OPEN_WEBUI_LLM_API_KEY=sk-test-litellm" in content


def test_enabled_perplexica_uses_stable_alias() -> None:
    payload = run_renderer(
        "--surface",
        "perplexica",
        "--switchboard-mode",
        "enabled",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Concrete.gguf",
        "--litellm-key",
        "sk-test-litellm",
    )
    content = json.loads(file_by_surface(payload, "perplexica")["content"])
    openai_provider = model_provider_by_id(content, "openai")
    assert content["preferences"]["defaultChatModel"] == "ods/current"
    assert openai_provider["config"]["baseURL"] == "http://litellm:4000/v1"
    assert openai_provider["config"]["apiKey"] == "sk-test-litellm"
    assert openai_provider["chatModels"][0]["key"] == "ods/current"


def test_enabled_hermes_uses_stable_switchboard_alias() -> None:
    payload = run_renderer(
        "--surface",
        "hermes",
        "--switchboard-mode",
        "enabled",
        "--gguf-file",
        "Raw-Runtime.gguf",
        "--llm-base-url",
        "http://llama-server:8080/v1",
    )
    content = file_by_surface(payload, "hermes")["content"]
    assert 'default: "ods/current"' in content
    assert "Raw-Runtime.gguf" not in content
    assert 'base_url: "http://litellm:4000/v1"' in content


def test_enabled_opencode_uses_stable_switchboard_alias() -> None:
    payload = run_renderer(
        "--surface",
        "opencode",
        "--switchboard-mode",
        "enabled",
        "--litellm-key",
        "switch-secret",
    )
    content = json.loads(file_by_surface(payload, "opencode")["content"])
    assert content["model"] == "ods/current"
    assert content["baseURL"] == "http://litellm:4000/v1"
    assert content["apiKey"] == "switch-secret"


def test_router_endpoints_strip_trailing_v1() -> None:
    payload = run_renderer(
        "--surface", "model-router-endpoints",
        "--llm-base-url", "http://llama-server:8080/v1",
        "--gpu-backend", "amd",
        "--lemonade-api-base", "http://lemonade:8000/api/v1",
    )
    import json as _json
    content = _json.loads(payload["files"][0]["content"])
    by_id = {e["id"]: e["baseUrl"] for e in content["endpoints"]}
    assert by_id["llama-server-default"] == "http://llama-server:8080"
    assert by_id["lemonade-default"] == "http://lemonade:8000/api"


def test_lemonade_disables_thinking_and_uses_extra_alias() -> None:
    payload = run_renderer(
        "--surface",
        "litellm-lemonade",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Model.gguf",
        "--litellm-key",
        "sk-test",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "model: openai/extra.Model.gguf" in content
    assert "api_key: sk-test" in content
    assert "enable_thinking: false" in content


def test_external_lemonade_uses_supplied_model_and_api_base() -> None:
    payload = run_renderer(
        "--surface",
        "litellm-lemonade",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--lemonade-model-id",
        "Qwen3-0.6B-GGUF",
        "--lemonade-api-base",
        "http://host.docker.internal:13305/api/v1",
        "--litellm-key",
        "lemonade-secret",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "model: openai/Qwen3-0.6B-GGUF" in content
    assert "api_base: http://host.docker.internal:13305/api/v1" in content
    assert "api_key: lemonade-secret" in content


def test_exact_lemonade_id_propagates_to_every_runtime_surface() -> None:
    payload = run_renderer(
        "--surface",
        "all",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Modern-Model.gguf",
        "--lemonade-model-id",
        "Modern-Model",
    )

    env_content = file_by_surface(payload, "env")["content"]
    litellm_content = file_by_surface(payload, "litellm-lemonade")["content"]
    hermes_content = file_by_surface(payload, "hermes")["content"]
    opencode = json.loads(file_by_surface(payload, "opencode")["content"])
    perplexica = json.loads(file_by_surface(payload, "perplexica")["content"])

    assert "LEMONADE_MODEL=Modern-Model" in env_content
    assert "model: openai/Modern-Model" in litellm_content
    assert 'default: "Modern-Model"' in hermes_content
    assert opencode["model"] == "Modern-Model"
    assert perplexica["preferences"]["defaultChatModel"] == "Modern-Model"


def test_amd_local_env_does_not_invent_a_lemonade_model() -> None:
    payload = run_renderer(
        "--surface",
        "env",
        "--ods-mode",
        "local",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Fallback-Model.gguf",
    )

    env_content = file_by_surface(payload, "env")["content"]
    assert "LEMONADE_MODEL=\n" in env_content
    assert "LEMONADE_MODEL=extra.Fallback-Model.gguf" not in env_content


def test_hermes_uses_lemonade_model_id_for_amd() -> None:
    payload = run_renderer(
        "--surface",
        "hermes",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Amd.gguf",
        "--llm-base-url",
        "http://litellm:4000/v1",
        "--context-length",
        "65536",
    )
    content = file_by_surface(payload, "hermes")["content"]
    assert 'default: "extra.Amd.gguf"' in content
    assert 'base_url: "http://litellm:4000/v1"' in content
    assert "context_length: 65536" in content
    assert "max_tokens: 1024" in content


def test_perplexica_default_model_matches_route() -> None:
    payload = run_renderer(
        "--surface",
        "perplexica",
        "--ods-mode",
        "lemonade",
        "--gpu-backend",
        "amd",
        "--gguf-file",
        "Research.gguf",
    )
    content = json.loads(file_by_surface(payload, "perplexica")["content"])
    openai_provider = model_provider_by_id(content, "openai")
    assert content["preferences"]["defaultChatModel"] == "extra.Research.gguf"
    assert openai_provider["chatModels"][0]["name"] == "extra.Research.gguf"


def test_write_mode_writes_under_output_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--surface",
                "litellm-lemonade",
                "--ods-mode",
                "lemonade",
                "--gpu-backend",
                "amd",
                "--gguf-file",
                "Written.gguf",
                "--output-root",
                tmp,
                "--write",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        payload = json.loads(proc.stdout)
        target = Path(tmp) / "config" / "litellm" / "lemonade.yaml"
        assert payload["mode"] == "write"
        assert target.exists()
        assert "openai/extra.Written.gguf" in target.read_text(encoding="utf-8")
        if os.name != "nt":
            assert target.stat().st_mode & 0o777 == 0o644
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_atomic_write_failure_preserves_known_good_config() -> None:
    renderer = load_renderer_module()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "lemonade.yaml"
        target.write_text("known-good\n", encoding="utf-8")
        original_replace = renderer.os.replace
        original_sleep = renderer.time.sleep

        def fail_replace(*_args, **_kwargs) -> None:
            raise PermissionError("injected replace failure")

        renderer.os.replace = fail_replace
        renderer.time.sleep = lambda _seconds: None
        try:
            try:
                renderer.atomic_write_text(target, "new-route\n")
            except PermissionError as exc:
                assert "injected replace failure" in str(exc)
            else:
                raise AssertionError("fault injection did not fail the replace")
        finally:
            renderer.os.replace = original_replace
            renderer.time.sleep = original_sleep

        assert target.read_text(encoding="utf-8") == "known-good\n"
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def _hipfire_args(*extra: str) -> list[str]:
    return [
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--litellm-key", "sk-test",
        *extra,
    ]


def test_hipfire_disabled_renders_stock_routes() -> None:
    payload = run_renderer(*_hipfire_args())
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "hipfire" not in content
    assert content.count("model_name") == 2


def test_hipfire_active_routes_default_to_hipfire() -> None:
    payload = run_renderer(*_hipfire_args(
        "--hipfire-enabled", "--hipfire-active",
        "--hipfire-model", "qwen36-35b-a3b.mq4",
    ))
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    hipfire_route = (
        "    litellm_params:\n"
        "      model: openai/qwen36-35b-a3b.mq4\n"
        "      api_base: http://hipfire:11435/v1\n"
        "      api_key: not-needed\n"
    )
    assert f"- model_name: default\n{hipfire_route}" in content
    assert 'model_name: "*"' not in content
    assert f"- model_name: hipfire\n{hipfire_route}" in content
    # Explicit escape hatch back to Lemonade/llama-server survives.
    assert "- model_name: lemonade\n    litellm_params:\n      model: openai/extra.Model.gguf" in content
    assert "api_key: sk-test" in content


def test_hipfire_enabled_inactive_keeps_default_on_lemonade() -> None:
    payload = run_renderer(*_hipfire_args(
        "--hipfire-enabled",
        "--hipfire-model", "qwen36-35b-a3b.mq4",
    ))
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "- model_name: default\n    litellm_params:\n      model: openai/extra.Model.gguf" in content
    # hipfire stays reachable by name even when it is not the default route.
    assert "- model_name: hipfire\n    litellm_params:\n      model: openai/qwen36-35b-a3b.mq4" in content


def test_hipfire_flag_without_model_renders_stock() -> None:
    payload = run_renderer(*_hipfire_args("--hipfire-enabled"))
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "hipfire" not in content


def test_hipfire_env_fallback_when_flags_absent() -> None:
    # `ods model swap` invokes the renderer with no hipfire flags; routing
    # state must come from the install tree's .env, not default to off.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text(
            "ENABLE_HIPFIRE=true\n"
            "HIPFIRE_MODEL=qwen36-35b-a3b.mq4\n"
            "HIPFIRE_ACTIVE=true\n",
            encoding="utf-8",
        )
        payload = run_renderer(*_hipfire_args("--output-root", str(root)))
        content = file_by_surface(payload, "litellm-lemonade")["content"]
        hipfire_route = (
            "    litellm_params:\n"
            "      model: openai/qwen36-35b-a3b.mq4\n"
            "      api_base: http://hipfire:11435/v1\n"
            "      api_key: not-needed\n"
        )
        assert f"- model_name: default\n{hipfire_route}" in content
        assert f"- model_name: hipfire\n{hipfire_route}" in content
        assert "- model_name: lemonade\n    litellm_params:\n      model: openai/extra.Model.gguf" in content


def test_hipfire_env_fallback_inactive_keeps_default_on_lemonade() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text(
            "ENABLE_HIPFIRE=true\n"
            "HIPFIRE_MODEL=qwen36-35b-a3b.mq4\n"
            "HIPFIRE_ACTIVE=false\n",
            encoding="utf-8",
        )
        payload = run_renderer(*_hipfire_args("--output-root", str(root)))
        content = file_by_surface(payload, "litellm-lemonade")["content"]
        assert "- model_name: default\n    litellm_params:\n      model: openai/extra.Model.gguf" in content
        assert "- model_name: hipfire\n    litellm_params:\n      model: openai/qwen36-35b-a3b.mq4" in content


def test_hipfire_env_absent_renders_stock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_renderer(*_hipfire_args("--output-root", tmp))
        content = file_by_surface(payload, "litellm-lemonade")["content"]
        assert "hipfire" not in content


def test_hipfire_explicit_flags_ignore_env() -> None:
    # An explicit invocation is authoritative: env disagreement must not
    # leak into the render (flags say inactive + a different model).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text(
            "ENABLE_HIPFIRE=true\n"
            "HIPFIRE_MODEL=env-model.mq4\n"
            "HIPFIRE_ACTIVE=true\n",
            encoding="utf-8",
        )
        payload = run_renderer(*_hipfire_args(
            "--hipfire-enabled",
            "--hipfire-model", "flag-model.mq4",
            "--output-root", str(root),
        ))
        content = file_by_surface(payload, "litellm-lemonade")["content"]
        assert "openai/flag-model.mq4" in content
        assert "env-model.mq4" not in content
        assert "- model_name: default\n    litellm_params:\n      model: openai/extra.Model.gguf" in content


def main() -> int:
    tests = [
        test_all_surfaces_render,
        test_switchboard_surface_gated_on_enabled_mode,
        test_all_selects_one_mode_config,
        test_cloud_enabled_never_renders_local_switchboard,
        test_remote_cloud_projection_uses_internal_egress_and_state_receipt,
        test_remote_routing_state_disabled_receipt_has_no_provider,
        test_remote_projection_requires_cloud_mode,
        test_remote_projection_rejects_unsafe_model_id,
        test_explicit_cloud_switchboard_render_fails_closed,
        test_native_local_projection_uses_host_route_and_concrete_model,
        test_checked_in_mode_configs_match_renderer,
        test_enabled_env_exports_switchboard_webui_gateway,
        test_enabled_perplexica_uses_stable_alias,
        test_enabled_hermes_uses_stable_switchboard_alias,
        test_enabled_opencode_uses_stable_switchboard_alias,
        test_lemonade_disables_thinking_and_uses_extra_alias,
        test_external_lemonade_uses_supplied_model_and_api_base,
        test_exact_lemonade_id_propagates_to_every_runtime_surface,
        test_amd_local_env_does_not_invent_a_lemonade_model,
        test_hermes_uses_lemonade_model_id_for_amd,
        test_perplexica_default_model_matches_route,
        test_write_mode_writes_under_output_root,
        test_hipfire_disabled_renders_stock_routes,
        test_hipfire_active_routes_default_to_hipfire,
        test_hipfire_enabled_inactive_keeps_default_on_lemonade,
        test_hipfire_flag_without_model_renders_stock,
        test_hipfire_env_fallback_when_flags_absent,
        test_hipfire_env_fallback_inactive_keeps_default_on_lemonade,
        test_hipfire_env_absent_renders_stock,
        test_hipfire_explicit_flags_ignore_env,
        test_atomic_write_failure_preserves_known_good_config,
        test_extra_routes_injected_before_lemonade_wildcard,
        test_extra_routes_injected_in_lemonade_hipfire_branch,
        test_extra_routes_injected_in_switchboard_surface,
        test_extra_routes_missing_file_renders_without_extras,
        test_extra_routes_reserved_name_fails_closed,
        test_extra_routes_malformed_json_fails_closed,
        test_extra_routes_missing_field_fails_closed,
        test_lemonade_routes_carry_context_windows,
        test_hipfire_context_window_env_fallback,
        test_extra_routes_carry_context_windows,
        test_core_routes_advertise_function_calling,
        test_extra_routes_carry_supports_function_calling,
        test_extra_routes_non_bool_supports_function_calling_fails_closed,
        test_lemonade_has_no_wildcard_and_aliases_llm_model,
        test_lemonade_alias_skips_reserved_names,
        test_lemonade_alias_skips_extra_route_names,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    return 0


# --- extra litellm routes sidecar (config/litellm/extra-routes.json) ---


def _extra_routes_file(tmpdir: str, payload: object) -> str:
    path = Path(tmpdir) / "extra-routes.json"
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return str(path)


def _spark_routes() -> list[dict[str, str]]:
    return [
        {"model_name": "spark-aeon", "model": "openai/aeon",
         "api_base": "http://192.168.1.15:8000/v1"},
        {"model_name": "spark-laguna", "model": "openai/laguna",
         "api_base": "http://192.168.1.15:8000/v1", "api_key": "not-needed"},
    ]


def test_extra_routes_injected_before_lemonade_wildcard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, _spark_routes())
        payload = run_renderer("--surface", "all", "--ods-mode", "lemonade",
                               "--extra-routes-file", path)
        content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "model_name: spark-aeon" in content
    assert "model_name: spark-laguna" in content
    assert "api_base: http://192.168.1.15:8000/v1" in content
    # The lemonade surface has no wildcard anymore — unknown names fail loudly.
    assert 'model_name: "*"' not in content
    # Defaulted api_key still renders (litellm requires the field).
    assert content.count("api_key: not-needed") >= 2


def test_extra_routes_injected_in_lemonade_hipfire_branch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, _spark_routes())
        payload = run_renderer("--surface", "all", "--ods-mode", "lemonade",
                               "--hipfire-enabled", "--hipfire-model", "m",
                               "--extra-routes-file", path)
        content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "model_name: spark-aeon" in content
    assert "model_name: hipfire" in content
    assert 'model_name: "*"' not in content


def test_extra_routes_injected_in_switchboard_surface() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, _spark_routes())
        payload = run_renderer("--surface", "all", "--switchboard-mode", "enabled",
                               "--extra-routes-file", path)
        content = file_by_surface(payload, "litellm-switchboard")["content"]
    assert "model_name: spark-aeon" in content
    assert content.index("spark-aeon") < content.index('model_name: "*"')
    # The model-router aliases stay untouched by the sidecar.
    assert content.count("http://model-router:9099/v1") == 4


def test_extra_routes_missing_file_renders_without_extras() -> None:
    payload = run_renderer("--surface", "all", "--ods-mode", "lemonade",
                           "--extra-routes-file", "/nonexistent/extra.json")
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert "spark" not in content


def test_extra_routes_reserved_name_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, [
            {"model_name": "default", "model": "openai/x",
             "api_base": "http://evil:1/v1"}])
        try:
            run_renderer("--surface", "all", "--ods-mode", "lemonade",
                         "--extra-routes-file", path)
        except subprocess.CalledProcessError as exc:
            assert "shadow" in exc.stderr
        else:
            raise AssertionError("reserved model_name must fail the render")


def test_extra_routes_malformed_json_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, "{not json")
        try:
            run_renderer("--surface", "all", "--ods-mode", "lemonade",
                         "--extra-routes-file", path)
        except subprocess.CalledProcessError as exc:
            assert "extra-routes" in exc.stderr
        else:
            raise AssertionError("malformed sidecar must fail the render")


def test_extra_routes_missing_field_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, [{"model_name": "spark-aeon"}])
        try:
            run_renderer("--surface", "all", "--ods-mode", "lemonade",
                         "--extra-routes-file", path)
        except subprocess.CalledProcessError as exc:
            assert "model" in exc.stderr
        else:
            raise AssertionError("incomplete entry must fail the render")


# --- model_info context windows (catalog-generator branch) ---


def test_lemonade_routes_carry_context_windows() -> None:
    payload = run_renderer(
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--context-length", "32768",
        "--hipfire-enabled", "--hipfire-active",
        "--hipfire-model", "qwen36-35b-a3b.mq4",
        "--hipfire-context-length", "262144",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    hipfire_block = content.split("- model_name: hipfire\n", 1)[1]
    assert "max_input_tokens: 262144" in hipfire_block.split("- model_name:", 1)[0]
    lemonade_block = content.split("- model_name: lemonade\n", 1)[1]
    assert "max_input_tokens: 32768" in lemonade_block.split("- model_name:", 1)[0]
    default_block = content.split("- model_name: default\n", 1)[1]
    assert "max_input_tokens: 262144" in default_block.split("- model_name:", 1)[0]


def test_hipfire_context_window_env_fallback() -> None:
    # Flag-less callers (the host-agent invocation) get hipfire's context from
    # the install tree's .env — HIPFIRE_MAX_SEQ, NOT CTX_SIZE (llama-server's).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text(
            "ENABLE_HIPFIRE=true\n"
            "HIPFIRE_MODEL=qwen36-35b-a3b.mq4\n"
            "HIPFIRE_ACTIVE=true\n"
            "HIPFIRE_MAX_SEQ=200000\n"
            "CTX_SIZE=32768\n",
            encoding="utf-8",
        )
        payload = run_renderer(*_hipfire_args("--output-root", str(root)))
        content = file_by_surface(payload, "litellm-lemonade")["content"]
        hipfire_block = content.split("- model_name: hipfire\n", 1)[1]
        assert "max_input_tokens: 200000" in hipfire_block.split("- model_name:", 1)[0]


def test_lemonade_has_no_wildcard_and_aliases_llm_model() -> None:
    payload = run_renderer(
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--model", "qwen3.5-27b",
        "--hipfire-enabled",
        "--hipfire-model", "qwen36-35b-a3b.mq4",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert 'model_name: "*"' not in content
    # hipfire enabled but NOT active -> default (and therefore the alias) is lemonade
    assert '- model_name: "qwen3.5-27b"\n    litellm_params:\n      model: openai/extra.Model.gguf' in content


def test_lemonade_alias_skips_reserved_names() -> None:
    payload = run_renderer(
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--model", "default",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert content.count("- model_name: default\n") == 1
    assert 'model_name: "default"' not in content


def test_lemonade_alias_skips_extra_route_names() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, _spark_routes())
        payload = run_renderer(
            "--surface", "litellm-lemonade",
            "--ods-mode", "lemonade",
            "--gguf-file", "Model.gguf",
            "--model", "spark-aeon",
            "--extra-routes-file", path,
        )
        content = file_by_surface(payload, "litellm-lemonade")["content"]
    assert content.count("model_name: spark-aeon") == 1
    assert 'model_name: "spark-aeon"' not in content


def test_extra_routes_carry_context_windows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, [
            {"model_name": "spark-aeon", "model": "openai/aeon",
             "api_base": "http://192.168.1.15:8000/v1",
             "max_input_tokens": 229376},
            {"model_name": "spark-heretic", "model": "openai/heretic",
             "api_base": "http://192.168.1.15:8000/v1"},
        ])
        payload = run_renderer("--surface", "all", "--ods-mode", "lemonade",
                               "--extra-routes-file", path)
        content = file_by_surface(payload, "litellm-lemonade")["content"]
    aeon_block = content.split("- model_name: spark-aeon\n", 1)[1].split("- model_name:", 1)[0]
    assert "max_input_tokens: 229376" in aeon_block
    heretic_block = content.split("- model_name: spark-heretic\n", 1)[1].split("- model_name:", 1)[0]
    assert "model_info" not in heretic_block


def test_core_routes_advertise_function_calling() -> None:
    # hipfire (llama.cpp fork) and lemonade (llama-server, jinja templates)
    # both do native OpenAI tool calling — live-verified 2026-08-02 through
    # litellm, stream + non-stream. default/alias reuse those same backends.
    payload = run_renderer(
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--model", "qwen3.5-27b",
        "--context-length", "32768",
        "--hipfire-enabled",
        "--hipfire-model", "qwen36-35b-a3b.mq4",
        "--hipfire-context-length", "262144",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    for name in ("default", "hipfire", "lemonade", '"qwen3.5-27b"'):
        block = content.split(f"- model_name: {name}\n", 1)[1].split("- model_name:", 1)[0]
        assert "supports_function_calling: true" in block, name
    # The hipfire-disabled branch advertises it on default/alias too.
    payload = run_renderer(
        "--surface", "litellm-lemonade",
        "--ods-mode", "lemonade",
        "--gpu-backend", "amd",
        "--gguf-file", "Model.gguf",
        "--context-length", "32768",
    )
    content = file_by_surface(payload, "litellm-lemonade")["content"]
    default_block = content.split("- model_name: default\n", 1)[1].split("- model_name:", 1)[0]
    assert "supports_function_calling: true" in default_block


def test_extra_routes_carry_supports_function_calling() -> None:
    # Discovery consumers (omp litellm-rich) gate NATIVE tool calling on
    # model_info.supports_function_calling; without it they fall back to
    # prompt-embedded tools, which the spark models mangle.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, [
            {"model_name": "spark-heretic", "model": "openai/heretic",
             "api_base": "http://192.168.1.15:8000/v1",
             "max_input_tokens": 262144,
             "supports_function_calling": True},
            {"model_name": "spark-tools-only", "model": "openai/x",
             "api_base": "http://192.168.1.15:8000/v1",
             "supports_function_calling": True},
            {"model_name": "spark-aeon", "model": "openai/aeon",
             "api_base": "http://192.168.1.15:8000/v1",
             "max_input_tokens": 229376},
        ])
        payload = run_renderer("--surface", "all", "--ods-mode", "lemonade",
                               "--extra-routes-file", path)
        content = file_by_surface(payload, "litellm-lemonade")["content"]
    heretic_block = content.split("- model_name: spark-heretic\n", 1)[1].split("- model_name:", 1)[0]
    assert "max_input_tokens: 262144" in heretic_block
    assert "supports_function_calling: true" in heretic_block
    # The flag alone is enough to emit a model_info block.
    tools_block = content.split("- model_name: spark-tools-only\n", 1)[1].split("- model_name:", 1)[0]
    assert "supports_function_calling: true" in tools_block
    assert "max_input_tokens" not in tools_block
    # Entries without the flag stay exactly as before.
    aeon_block = content.split("- model_name: spark-aeon\n", 1)[1].split("- model_name:", 1)[0]
    assert "supports_function_calling" not in aeon_block


def test_extra_routes_non_bool_supports_function_calling_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _extra_routes_file(tmpdir, [
            {"model_name": "spark-heretic", "model": "openai/heretic",
             "api_base": "http://192.168.1.15:8000/v1",
             "supports_function_calling": "yes"}])
        try:
            run_renderer("--surface", "all", "--ods-mode", "lemonade",
                         "--extra-routes-file", path)
        except subprocess.CalledProcessError as exc:
            assert "supports_function_calling" in exc.stderr
        else:
            raise AssertionError("non-bool supports_function_calling must fail the render")


if __name__ == "__main__":
    raise SystemExit(main())
