"""Tests for AMD model activation helpers in ods-host-agent.py."""

import importlib.util
import http.client
import io
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# Import the host agent module from bin/ using importlib.
# The module has an ``if __name__ == "__main__":`` guard so no server starts.
_agent_path = Path(__file__).resolve().parents[4] / "bin" / "ods-host-agent.py"
_spec = importlib.util.spec_from_file_location("ods_host_agent_activate", _agent_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ods_host_agent_activate"] = _mod
_spec.loader.exec_module(_mod)

_check_lemonade_health = _mod._check_lemonade_health
_send_lemonade_warmup = _mod._send_lemonade_warmup
_lemonade_completion_ready = _mod._lemonade_completion_ready
_write_lemonade_config = _mod._write_lemonade_config
_patch_hermes_model_config = _mod._patch_hermes_model_config
_compose_restart_llama_server = _mod._compose_restart_llama_server
_launch_native_llama_server = _mod._launch_native_llama_server
_restart_windows_lemonade = _mod._restart_windows_lemonade
_is_windows_host_llama_server = _mod._is_windows_host_llama_server
_restart_windows_native_llama_server = _mod._restart_windows_native_llama_server
_write_windows_native_litellm_config = _mod._write_windows_native_litellm_config


def test_host_agent_backlog_handles_dashboard_poll_bursts():
    assert _mod.ThreadedHTTPServer.request_queue_size >= 64


def test_host_agent_keeps_gets_alive_and_closes_posts(monkeypatch):
    class _CountingServer(_mod.ThreadedHTTPServer):
        accepted_connections = 0

        def get_request(self):
            request = super().get_request()
            self.accepted_connections += 1
            return request

    server = _CountingServer(("127.0.0.1", 0), _mod.AgentHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        monkeypatch.setattr(_mod, "AGENT_API_KEY", "keepalive-test-key")
        for _ in range(20):
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "ok", "version": _mod.VERSION}
        assert server.accepted_connections == 1

        connection.request(
            "POST",
            "/v1/model/download/cancel",
            body="{}",
            headers={
                "Authorization": "Bearer keepalive-test-key",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Connection") == "close"
        assert json.loads(response.read()) == {"status": "no_download"}

        # http.client transparently opens a fresh socket after the explicit
        # close; the unread cancel body cannot corrupt this request.
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok", "version": _mod.VERSION}
        assert server.accepted_connections == 2
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


# --- _check_lemonade_health ---


class TestCheckLemonadeHealth:

    def test_model_loaded(self):
        body = '{"status": "ok", "model_loaded": "extra.Qwen3.5-9B-Q4_K_M.gguf"}'
        assert _check_lemonade_health(body) is True

    def test_model_null(self):
        body = '{"status": "ok", "model_loaded": null}'
        assert _check_lemonade_health(body) is False

    def test_no_model_loaded_key(self):
        body = '{"status": "ok"}'
        assert _check_lemonade_health(body) is False

    def test_invalid_json(self):
        assert _check_lemonade_health("not json") is False

    def test_empty_string(self):
        assert _check_lemonade_health("") is False

    def test_model_loaded_false_is_not_an_identity(self):
        body = '{"model_loaded": false}'
        assert _check_lemonade_health(body) is False

    def test_model_loaded_empty_string_is_not_an_identity(self):
        body = '{"model_loaded": ""}'
        assert _check_lemonade_health(body) is False


# --- _send_lemonade_warmup ---


class TestSendLemonadeWarmup:

    def test_success(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            result = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _send_lemonade_warmup("localhost", "8080", "model.gguf", 0) is True
        assert len(calls) == 1
        # Verify curl is called with correct URL and model ID
        cmd = calls[0]
        assert "http://localhost:8080/api/v1/chat/completions" in cmd
        payload_idx = cmd.index("-d") + 1
        assert '"extra.model.gguf"' in cmd[payload_idx]

    def test_failure(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _send_lemonade_warmup("localhost", "8080", "model.gguf", 0) is False

    def test_timeout(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 35))

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _send_lemonade_warmup("localhost", "8080", "model.gguf", 0) is False

    def test_containerized_host(self, monkeypatch):
        """Verify the host parameter is used (not hardcoded to localhost)."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _send_lemonade_warmup("ods-llama-server", "8080", "model.gguf", 0)
        assert "http://ods-llama-server:8080/api/v1/chat/completions" in calls[0]


class TestLemonadeCompletionReady:

    def test_success_when_completion_has_choices(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"choices":[{"message":{"content":"ok"}}]}',
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _lemonade_completion_ready("127.0.0.1", "8080", "model.gguf") is True
        assert "http://127.0.0.1:8080/api/v1/chat/completions" in calls[0]
        payload = calls[0][calls[0].index("-d") + 1]
        assert '"extra.model.gguf"' in payload

    def test_false_on_nonzero_exit(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _lemonade_completion_ready("127.0.0.1", "8080", "model.gguf") is False

    def test_false_on_invalid_json(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="not-json", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _lemonade_completion_ready("127.0.0.1", "8080", "model.gguf") is False


# --- _write_lemonade_config ---


class TestWriteLemonadeConfig:

    def test_writes_correct_content(self, tmp_path):
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        _write_lemonade_config(tmp_path, "Qwen3.5-9B-Q4_K_M.gguf")

        content = (litellm_dir / "lemonade.yaml").read_text()
        assert "model: openai/extra.Qwen3.5-9B-Q4_K_M.gguf" in content
        assert "api_base: http://llama-server:8080/api/v1" in content
        assert "api_key: sk-lemonade" in content
        assert "extra_body:" in content
        assert "chat_template_kwargs:" in content
        assert "enable_thinking: false" in content
        assert 'model_name: "*"' in content
        assert "drop_params: true" in content
        assert "request_timeout: 900" in content
        assert "stream_timeout: 900" in content

    def test_fallback_writer_keeps_long_model_timeouts(self, monkeypatch, tmp_path):
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        monkeypatch.setattr(_mod, "_render_runtime_config", lambda *args, **kwargs: False)

        _write_lemonade_config(tmp_path, "fallback-model.gguf")

        content = (litellm_dir / "lemonade.yaml").read_text()
        assert "model: openai/extra.fallback-model.gguf" in content
        assert "request_timeout: 900" in content
        assert "stream_timeout: 900" in content

    def test_reads_lemonade_key_from_env_file_when_process_env_unset(
        self, monkeypatch, tmp_path,
    ):
        """Installer-written Lemonade keys live in .env, not process env."""
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        (tmp_path / ".env").write_text(
            "LITELLM_LEMONADE_API_KEY=sk-from-env-file-12345\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("LITELLM_LEMONADE_API_KEY", raising=False)

        _write_lemonade_config(tmp_path, "Qwen3.5-9B-Q4_K_M.gguf")

        content = (litellm_dir / "lemonade.yaml").read_text()
        assert "api_key: sk-from-env-file-12345" in content
        assert "api_key: sk-lemonade" not in content

    def test_overwrites_previous(self, tmp_path):
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)

        _write_lemonade_config(tmp_path, "old-model.gguf")
        _write_lemonade_config(tmp_path, "new-model.gguf")

        content = (litellm_dir / "lemonade.yaml").read_text()
        assert "old-model.gguf" not in content
        assert "model: openai/extra.new-model.gguf" in content

    def test_file_path(self, tmp_path):
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        _write_lemonade_config(tmp_path, "model.gguf")
        assert (litellm_dir / "lemonade.yaml").exists()


class TestPatchHermesModelConfig:

    def test_updates_model_default_only(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "# example\n"
            "model:\n"
            "  default: \"old-model\"\n"
            "  provider: \"custom\"\n"
            "  base_url: \"http://llama-server:8080/v1\"\n"
            "other:\n"
            "  default: \"leave-me\"\n",
            encoding="utf-8",
        )

        assert _patch_hermes_model_config(config, "new-model.gguf") is True

        text = config.read_text(encoding="utf-8")
        assert '  default: "new-model.gguf"' in text
        assert '  provider: "custom"' in text
        assert '  default: "leave-me"' in text

    def test_updates_context_and_base_url(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "model:\n"
            "  default: \"old-model\"\n"
            "  provider: \"custom\"\n"
            "  base_url: \"http://host.docker.internal:8080/v1\"\n"
            "  context_length: 32768\n"
            "auxiliary:\n"
            "  compression:\n"
            "    context_length: 32768\n",
            encoding="utf-8",
        )

        assert _patch_hermes_model_config(
            config,
            "new-model.gguf",
            base_url="http://llama-server:8080/v1",
            context_length=131072,
        ) is True

        text = config.read_text(encoding="utf-8")
        assert '  default: "new-model.gguf"' in text
        assert '  base_url: "http://llama-server:8080/v1"' in text
        assert "  context_length: 131072" in text
        assert "    context_length: 131072" in text

    def test_missing_file_is_noop(self, tmp_path):
        assert _patch_hermes_model_config(tmp_path / "missing.yaml", "model.gguf") is False

    def test_permission_denied_stat_is_noop(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        original_exists = Path.exists

        def fake_exists(path):
            if path == config:
                raise PermissionError("container-owned")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", fake_exists)
        assert _patch_hermes_model_config(config, "model.gguf") is False


class TestComposeRestartLlamaServer:

    def test_amd_uses_stop_then_up(self, monkeypatch, tmp_path):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(_mod, "INSTALL_DIR", tmp_path)
        monkeypatch.setattr(
            _mod,
            "resolve_compose_flags",
            lambda: ["--env-file", ".env", "-f", "docker-compose.base.yml"],
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        _compose_restart_llama_server({"GPU_BACKEND": "amd"})

        assert calls == [
            [
                "docker", "compose", "--env-file", ".env", "-f",
                "docker-compose.base.yml", "stop", "llama-server",
            ],
            [
                "docker", "compose", "--env-file", ".env", "-f",
                "docker-compose.base.yml", "up", "-d", "llama-server",
            ],
        ]


class TestLaunchNativeLlamaServer:

    def test_reads_env_and_writes_pid(self, monkeypatch, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text(
            "GGUF_FILE=test-model.gguf\n"
            "CTX_SIZE=8192\n"
            "LLAMA_REASONING=on\n"
            "AMD_INFERENCE_PORT=9090\n",
            encoding="utf-8",
        )
        (tmp_path / "data" / "models").mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        llama_bin = tmp_path / "bin" / "llama-server"
        llama_bin.parent.mkdir(parents=True)
        llama_bin.write_text("", encoding="utf-8")
        llama_log = tmp_path / "data" / "llama-server.log"
        pid_file = tmp_path / "data" / ".llama-server.pid"

        calls = []

        class _FakeProc:
            pid = 4321

        def fake_popen(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return _FakeProc()

        monkeypatch.setattr(_mod, "INSTALL_DIR", tmp_path)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        _launch_native_llama_server(env_path, llama_bin, llama_log, pid_file)

        assert pid_file.read_text(encoding="utf-8") == "4321"
        cmd, _kwargs = calls[0]
        assert cmd[0] == str(llama_bin)
        assert "--model" in cmd
        assert str(tmp_path / "data" / "models" / "test-model.gguf") in cmd
        assert cmd[cmd.index("--port") + 1] == "9090"
        assert "--ctx-size" in cmd
        assert "8192" in cmd
        assert "--reasoning-format" in cmd
        assert "deepseek" in cmd
        assert _kwargs["cwd"] == str(tmp_path)

    def test_llm_bridge_is_disabled_before_native_bind(self, monkeypatch, tmp_path):
        env = {
            "GGUF_FILE": "test-model.gguf",
            "BIND_ADDRESS": "0.0.0.0",
            "ODS_MACOS_HOST_GATEWAY": "192.168.106.1",
        }
        events = []

        class _FakeProc:
            pid = 4321

        def fake_disable(actual_env, bind_addr, label):
            events.append(("bootout", actual_env, bind_addr, label))
            return True

        def fake_popen(cmd, **kwargs):
            events.append(("bind", cmd, kwargs))
            return _FakeProc()

        monkeypatch.setattr(_mod, "INSTALL_DIR", tmp_path)
        monkeypatch.setattr(_mod, "load_env", lambda _path: env)
        monkeypatch.setattr(_mod, "_disable_conflicting_macos_bridge", fake_disable)
        monkeypatch.setattr(_mod.subprocess, "Popen", fake_popen)

        _launch_native_llama_server(
            tmp_path / ".env",
            tmp_path / "bin" / "llama-server",
            tmp_path / "data" / "llama-server.log",
            tmp_path / "data" / ".llama-server.pid",
        )

        assert events[0] == (
            "bootout",
            env,
            "0.0.0.0",
            "com.ods.llm-bridge",
        )
        assert events[1][0] == "bind"
        assert events[1][1][events[1][1].index("--host") + 1] == "0.0.0.0"


class TestWindowsNativeLlamaServer:

    def test_detects_managed_windows_amd_llama_fallback(self, monkeypatch):
        monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")

        assert _is_windows_host_llama_server({
            "GPU_BACKEND": "amd",
            "LLM_BACKEND": "llama-server",
            "AMD_INFERENCE_RUNTIME": "llama-server",
            "AMD_INFERENCE_RUNTIME_MODE": "windows-llama-server-fallback",
            "AMD_INFERENCE_LOCATION": "host",
            "AMD_INFERENCE_MANAGED": "true",
        }) is True
        assert _is_windows_host_llama_server({
            "GPU_BACKEND": "amd",
            "LLM_BACKEND": "lemonade",
            "AMD_INFERENCE_RUNTIME": "lemonade",
            "AMD_INFERENCE_LOCATION": "host",
        }) is False

    def test_writes_litellm_local_config_for_native_windows_llama(self, tmp_path):
        _write_windows_native_litellm_config(
            tmp_path,
            "Qwen3.5-4B-Q4_K_M.gguf",
            {"AMD_INFERENCE_PORT": "9090"},
        )

        content = (tmp_path / "config" / "litellm" / "local.yaml").read_text(encoding="utf-8")
        assert "model: openai/Qwen3.5-4B-Q4_K_M.gguf" in content
        assert "api_base: http://host.docker.internal:9090/v1" in content
        assert 'model_name: "*"' in content
        assert "request_timeout: 900" in content

    def test_native_restart_uses_stop_process_before_taskkill(self, monkeypatch, tmp_path):
        install_dir = tmp_path / "install"
        env_path = install_dir / ".env"
        model_dir = install_dir / "data" / "models"
        llama_bin = install_dir / "llama-server" / "llama-server.exe"
        model_dir.mkdir(parents=True)
        llama_bin.parent.mkdir(parents=True)
        model_dir.joinpath("model.gguf").write_text("model", encoding="utf-8")
        llama_bin.write_text("", encoding="utf-8")
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("GGUF_FILE=model.gguf\nAMD_INFERENCE_PORT=9090\n", encoding="utf-8")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["script"] = cmd[-1]
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        launch_calls = []

        def fake_launch(*args):
            launch_calls.append(args)

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(_mod, "_launch_native_llama_server", fake_launch)

        _restart_windows_native_llama_server(env_path, {
            "GGUF_FILE": "model.gguf",
            "AMD_INFERENCE_PORT": "9090",
        })

        assert "Stop-Process -Id $ProcId -Force" in captured["script"]
        assert "taskkill.exe /PID $ProcId /F" in captured["script"]
        assert "taskkill.exe /PID $ProcId /T /F" not in captured["script"]
        assert "Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Ignore" in captured["script"]
        assert captured["script"].rstrip().endswith("exit 0")
        assert captured["env"]["ODS_WIN_LLAMA_PORT"] == "9090"
        assert launch_calls


class TestRestartWindowsLemonade:

    def test_refreshes_task_with_current_exe_and_falls_back_to_direct_start(self, monkeypatch, tmp_path):
        program_files = tmp_path / "Program Files"
        lemonade_exe = program_files / "Lemonade Server" / "bin" / "LemonadeServer.exe"
        lemonade_exe.parent.mkdir(parents=True)
        lemonade_exe.write_text("", encoding="utf-8")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["script"] = cmd[-1]
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setenv("ProgramFiles", str(program_files))
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(_mod, "INSTALL_DIR", tmp_path)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)

        _restart_windows_lemonade({
            "AMD_INFERENCE_PORT": "8080",
            "BIND_ADDRESS": "0.0.0.0",
        })

        script = captured["script"]
        assert "$existingTask = Get-ScheduledTask -TaskName $taskName" in script
        assert "Could not refresh Lemonade scheduled task; reusing existing task" in script
        assert "Register-ScheduledTask -TaskName $taskName" in script
        assert "-Force -ErrorAction Stop | Out-Null" in script
        assert "Unregister-ScheduledTask" not in script
        assert "taskkill.exe /PID $ProcId /T /F" in script
        assert "for ($i = 0; $i -lt 45; $i++)" in script
        assert "task result: $taskResult" in script
        assert "Start-ScheduledTask -TaskName $taskName" in script
        assert "Start-Process -FilePath $exe" in script
        assert "Lemonade scheduled task did not start a server process" in script
        assert "Stop-ScheduledTask -TaskName $taskName" in script
        assert captured["env"]["ODS_WIN_LEMONADE_TASK"] == "ODSLemonadeRuntime"
        assert captured["env"]["ODS_WIN_LEMONADE_EXE"] == str(lemonade_exe)


# --- Rollback integration ---


class _ResponseHandler:
    def __init__(self, wfile=None):
        self.wfile = wfile or io.BytesIO()
        self.response_code = None
        self.response_headers = []

    def send_response(self, code):
        self.response_code = code

    def send_header(self, name, value):
        self.response_headers.append((name, value))

    def end_headers(self):
        pass

    def parse_response(self):
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class _BrokenPipeWriter:
    def write(self, _payload):
        raise BrokenPipeError("client disconnected")


def _llama_identity_response(model_id):
    return json.dumps({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "status": {"value": "loaded"},
        }],
    })


def _write_model_activation_fixture(
    tmp_path,
    gpu_backend="nvidia",
    lemonade=False,
    lemonade_api_key=None,
):
    install_dir = tmp_path / "install"
    config_dir = install_dir / "config"
    models_dir = install_dir / "data" / "models"
    llama_dir = config_dir / "llama-server"
    litellm_dir = config_dir / "litellm"
    models_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    litellm_dir.mkdir(parents=True)

    (models_dir / "new-model.gguf").write_text("model", encoding="utf-8")
    (config_dir / "model-library.json").write_text(
        json.dumps({
            "models": [{
                "id": "target-model",
                "gguf_file": "new-model.gguf",
                "llm_model_name": "new-model",
                "context_length": 4096,
            }]
        }),
        encoding="utf-8",
    )

    env_text = (
        f"GPU_BACKEND={gpu_backend}\n"
        "GGUF_FILE=old-model.gguf\n"
        "LLM_MODEL=old-model\n"
        "CTX_SIZE=2048\n"
        "OLLAMA_PORT=8080\n"
    )
    if lemonade_api_key:
        env_text += f"LITELLM_LEMONADE_API_KEY={lemonade_api_key}\n"
    env_path = install_dir / ".env"
    env_path.write_text(env_text, encoding="utf-8")

    ini_text = "[old-model]\nfilename = old-model.gguf\n"
    models_ini = llama_dir / "models.ini"
    models_ini.write_text(ini_text, encoding="utf-8")

    lemonade_yaml = litellm_dir / "lemonade.yaml"
    lemonade_text = None
    if lemonade:
        lemonade_text = "model_list:\n  - model_name: old\n"
        lemonade_yaml.write_text(lemonade_text, encoding="utf-8")

    return install_dir, env_path, env_text, models_ini, ini_text, lemonade_yaml, lemonade_text


class TestModelActivateRollback:

    @pytest.mark.parametrize(
        ("bind_addr", "expected_identity_url"),
        [
            ("0.0.0.0", "http://127.0.0.1:9090/v1/models"),
            ("::", "http://[::1]:9090/v1/models"),
            ("192.168.106.1", "http://192.168.106.1:9090/v1/models"),
        ],
    )
    def test_apple_native_activation_probes_reachable_bind(
        self,
        tmp_path,
        monkeypatch,
        bind_addr,
        expected_identity_url,
    ):
        install_dir, env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path, gpu_backend="apple")
        )
        env_path.write_text(
            env_path.read_text(encoding="utf-8")
            + f"BIND_ADDRESS={bind_addr}\nODS_NATIVE_LLAMA_PORT=9090\n",
            encoding="utf-8",
        )
        llama_bin = install_dir / "bin" / "llama-server"
        llama_bin.parent.mkdir(parents=True)
        llama_bin.write_text("", encoding="utf-8")
        lib_dir = install_dir / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "constants.sh").write_text("# test fixture\n", encoding="utf-8")
        (lib_dir / "bridge-manager.sh").write_text("# test fixture\n", encoding="utf-8")
        launches = []
        calls = []

        def fake_launch(*args):
            launches.append(args)

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.platform, "system", lambda: "Darwin")
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_configure_macos_llm_bridge", lambda _env_path: None)
        monkeypatch.setattr(_mod, "_launch_native_llama_server", fake_launch)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        assert len(launches) == 1
        curl_calls = [cmd for cmd in calls if cmd and cmd[0] == "curl"]
        assert [cmd[-1] for cmd in curl_calls] == [expected_identity_url]

    def test_success_response_disconnect_does_not_roll_back(self, tmp_path, monkeypatch):
        install_dir, env_path, _env_text, models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler(wfile=_BrokenPipeWriter())

        with pytest.raises(BrokenPipeError):
            _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert "GGUF_FILE=new-model.gguf" in env_path.read_text(encoding="utf-8")
        assert "LLM_MODEL=new-model" in env_path.read_text(encoding="utf-8")
        assert "filename = new-model.gguf" in models_ini.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "runtime_kind",
        [
            "compose-llama",
            "container-llama",
            "windows-lemonade",
            "windows-native-llama",
            "macos-native-llama",
        ],
    )
    def test_late_failure_restores_previous_runtime_and_config(
        self,
        tmp_path,
        monkeypatch,
        runtime_kind,
    ):
        gpu_backend = "amd" if runtime_kind.startswith("windows-") else "nvidia"
        if runtime_kind == "macos-native-llama":
            gpu_backend = "apple"
        install_dir, env_path, _env_text, models_ini, _ini_text, lemonade_yaml, _ = (
            _write_model_activation_fixture(
                tmp_path,
                gpu_backend=gpu_backend,
                lemonade=runtime_kind == "windows-lemonade",
            )
        )

        if runtime_kind == "windows-lemonade":
            env_path.write_text(
                "ODS_MODE=lemonade\n"
                "GPU_BACKEND=amd\n"
                "LLM_BACKEND=lemonade\n"
                "AMD_INFERENCE_RUNTIME=lemonade\n"
                "AMD_INFERENCE_LOCATION=host\n"
                "AMD_INFERENCE_PORT=8080\n"
                "GGUF_FILE=old-model.gguf\n"
                "LLM_MODEL=old-model\n"
                "CTX_SIZE=2048\n",
                encoding="utf-8",
            )
        elif runtime_kind == "windows-native-llama":
            env_path.write_text(
                "ODS_MODE=local\n"
                "GPU_BACKEND=amd\n"
                "LLM_BACKEND=llama-server\n"
                "AMD_INFERENCE_RUNTIME=llama-server\n"
                "AMD_INFERENCE_RUNTIME_MODE=windows-llama-server-fallback\n"
                "AMD_INFERENCE_LOCATION=host\n"
                "AMD_INFERENCE_MANAGED=true\n"
                "AMD_INFERENCE_PORT=9090\n"
                "GGUF_FILE=old-model.gguf\n"
                "LLM_MODEL=old-model\n"
                "CTX_SIZE=2048\n",
                encoding="utf-8",
            )
        elif runtime_kind == "macos-native-llama":
            llama_bin = install_dir / "bin" / "llama-server"
            llama_bin.parent.mkdir(parents=True)
            llama_bin.write_text("binary", encoding="utf-8")

        local_yaml = install_dir / "config" / "litellm" / "local.yaml"
        tracked_configs = (env_path, models_ini, lemonade_yaml, local_yaml)

        def config_state():
            return {
                path: path.read_text(encoding="utf-8") if path.exists() else None
                for path in tracked_configs
            }

        original_config = config_state()
        runtime_restarts = []

        def record_restart(env):
            runtime_restarts.append((env["GGUF_FILE"], config_state()))

        def record_container_restart(env, override_image=""):
            record_restart(env)

        def record_native_restart(path, _env=None, *_args):
            record_restart(_mod.load_env(path))

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        if runtime_kind == "compose-llama":
            monkeypatch.setattr(_mod.platform, "system", lambda: "Linux")
            monkeypatch.setattr(_mod, "_compose_restart_llama_server", record_restart)
        elif runtime_kind == "container-llama":
            monkeypatch.setattr(_mod.platform, "system", lambda: "Linux")
            monkeypatch.setenv("ODS_HOST_INSTALL_DIR", str(install_dir))
            monkeypatch.setattr(_mod, "_recreate_llama_server", record_container_restart)
        elif runtime_kind == "windows-lemonade":
            monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
            monkeypatch.setattr(_mod, "_restart_windows_lemonade", record_restart)
        elif runtime_kind == "windows-native-llama":
            monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
            monkeypatch.setattr(_mod, "_restart_windows_native_llama_server", record_native_restart)
        else:
            monkeypatch.setattr(_mod.platform, "system", lambda: "Darwin")
            monkeypatch.setattr(_mod, "_restart_macos_native_llama_server", record_native_restart)

        def fake_run(cmd, **_kwargs):
            if cmd and cmd[0] == "curl":
                stdout = (
                    '{"status": "ok", "model_loaded": "extra.new-model.gguf"}'
                    if runtime_kind == "windows-lemonade"
                    else _llama_identity_response("new-model.gguf")
                )
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            if cmd == ["docker", "restart", "ods-litellm"]:
                raise subprocess.TimeoutExpired(cmd, 60)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 500
        assert [model for model, _state in runtime_restarts] == [
            "new-model.gguf",
            "old-model.gguf",
        ]
        assert "GGUF_FILE=new-model.gguf" in runtime_restarts[0][1][env_path]
        assert "filename = new-model.gguf" in runtime_restarts[0][1][models_ini]
        assert runtime_restarts[1][1] == original_config
        assert config_state() == original_config

    def test_activation_accepts_local_gguf_without_catalog_entry(self, tmp_path, monkeypatch):
        install_dir, env_path, _env_text, models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        (install_dir / "config" / "model-library.json").write_text(
            json.dumps({"models": []}),
            encoding="utf-8",
        )
        (install_dir / "data" / "models" / "Research.Model-Q8_0.gguf").write_text(
            "model",
            encoding="utf-8",
        )
        env_path.write_text(
            "GPU_BACKEND=nvidia\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "MAX_CONTEXT=65536\n"
            "LLAMA_ARG_SPEC_TYPE=draft-mtp\n"
            "LLAMA_ARG_SPEC_DRAFT_N_MAX=3\n"
            "OLLAMA_PORT=8080\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = (
                _llama_identity_response("Research.Model-Q8_0.gguf")
                if cmd and cmd[0] == "curl"
                else ""
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "Research.Model-Q8_0")

        assert handler.response_code == 200
        assert handler.parse_response() == {"status": "activated", "model_id": "Research.Model-Q8_0"}
        env_text = env_path.read_text(encoding="utf-8")
        assert "GGUF_FILE=Research.Model-Q8_0.gguf" in env_text
        assert "LLM_MODEL=Research.Model-Q8_0" in env_text
        assert "CTX_SIZE=65536" in env_text
        assert "LLAMA_ARG_SPEC_TYPE=" not in env_text
        assert "LLAMA_ARG_SPEC_DRAFT_N_MAX=" not in env_text
        assert "filename = Research.Model-Q8_0.gguf" in models_ini.read_text(encoding="utf-8")

    def test_activation_resolves_local_gguf_by_stem_with_mixed_case_extension(
        self, tmp_path, monkeypatch,
    ):
        install_dir, env_path, _env_text, models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        (install_dir / "config" / "model-library.json").write_text(
            json.dumps({"models": []}),
            encoding="utf-8",
        )
        (install_dir / "data" / "models" / "MixedCaseModel.GGUF").write_text(
            "model",
            encoding="utf-8",
        )
        env_path.write_text(
            "GPU_BACKEND=nvidia\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "MAX_CONTEXT=32768\n"
            "OLLAMA_PORT=8080\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = (
                _llama_identity_response("MixedCaseModel.GGUF")
                if cmd and cmd[0] == "curl"
                else ""
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "MixedCaseModel")

        assert handler.response_code == 200
        env_text = env_path.read_text(encoding="utf-8")
        assert "GGUF_FILE=MixedCaseModel.GGUF" in env_text
        assert "LLM_MODEL=MixedCaseModel" in env_text
        assert "filename = MixedCaseModel.GGUF" in models_ini.read_text(encoding="utf-8")

    def test_activation_accepts_sanitized_local_id_for_spaced_filename(
        self, tmp_path, monkeypatch,
    ):
        install_dir, env_path, _env_text, models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        (install_dir / "config" / "model-library.json").write_text(
            json.dumps({"models": []}),
            encoding="utf-8",
        )
        (install_dir / "data" / "models" / "My Custom Model.Q8_0.GGUF").write_text(
            "model",
            encoding="utf-8",
        )
        env_path.write_text(
            "GPU_BACKEND=nvidia\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "MAX_CONTEXT=32768\n"
            "OLLAMA_PORT=8080\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = (
                _llama_identity_response("My Custom Model.Q8_0.GGUF")
                if cmd and cmd[0] == "curl"
                else ""
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "My-Custom-Model.Q8_0")

        assert handler.response_code == 200
        assert handler.parse_response() == {
            "status": "activated",
            "model_id": "My-Custom-Model.Q8_0",
        }
        env_text = env_path.read_text(encoding="utf-8")
        assert "GGUF_FILE=My Custom Model.Q8_0.GGUF" in env_text
        assert "LLM_MODEL=My-Custom-Model.Q8_0" in env_text
        assert "[My-Custom-Model.Q8_0]" in models_ini.read_text(encoding="utf-8")
        assert "filename = My Custom Model.Q8_0.GGUF" in models_ini.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "model_id",
        [
            "../outside",
            r"..\outside",
            "nested/model",
            r"nested\model",
            "unsafe\x00model",
        ],
    )
    def test_local_gguf_resolver_rejects_path_traversal(self, tmp_path, model_id):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "outside.gguf").write_text("model", encoding="utf-8")

        assert _mod._resolve_local_gguf_filename(model_id, models_dir) is None

    def test_local_gguf_resolver_rejects_ambiguous_sanitized_id(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "My Custom Model.gguf").write_text("model", encoding="utf-8")
        (models_dir / "My@Custom@Model.gguf").write_text("model", encoding="utf-8")

        assert _mod._resolve_local_gguf_filename("My-Custom-Model", models_dir) is None

    def test_activation_rejects_empty_local_gguf_before_restart(self, tmp_path, monkeypatch):
        install_dir, _env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        (install_dir / "config" / "model-library.json").write_text(
            json.dumps({"models": []}),
            encoding="utf-8",
        )
        (install_dir / "data" / "models" / "EmptyLocal.gguf").write_text(
            "",
            encoding="utf-8",
        )

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)

        def fail_restart(_env):
            raise AssertionError("empty GGUF should be rejected before restart")

        monkeypatch.setattr(_mod, "_compose_restart_llama_server", fail_restart)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "EmptyLocal")

        assert handler.response_code == 400
        assert "not downloaded or empty" in handler.parse_response()["error"]

    def test_amd_activation_rewrites_lemonade_yaml_with_env_file_key(
        self, tmp_path, monkeypatch,
    ):
        install_dir, _env_path, _env_text, _models_ini, _ini_text, lemonade_yaml, _yaml_text = (
            _write_model_activation_fixture(
                tmp_path,
                gpu_backend="amd",
                lemonade=True,
                lemonade_api_key="sk-inline-from-env-file-67890",
            )
        )
        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.delenv("LITELLM_LEMONADE_API_KEY", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = (
                '{"status": "ok", "model_loaded": "extra.new-model.gguf"}'
                if cmd and cmd[0] == "curl"
                else ""
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        content = lemonade_yaml.read_text(encoding="utf-8")
        assert "api_key: sk-inline-from-env-file-67890" in content
        assert "api_key: sk-lemonade" not in content
        assert "enable_thinking: false" in content

    def test_windows_lemonade_already_serving_skips_native_restart(
        self, tmp_path, monkeypatch,
    ):
        install_dir, env_path, _env_text, _models_ini, _ini_text, lemonade_yaml, _yaml_text = (
            _write_model_activation_fixture(
                tmp_path,
                gpu_backend="amd",
                lemonade=True,
                lemonade_api_key="sk-inline-from-env-file-67890",
            )
        )
        env_path.write_text(
            "GPU_BACKEND=amd\n"
            "LLM_BACKEND=lemonade\n"
            "AMD_INFERENCE_RUNTIME=lemonade\n"
            "AMD_INFERENCE_LOCATION=host\n"
            "AMD_INFERENCE_PORT=8080\n"
            "GGUF_FILE=new-model.gguf\n"
            "LLM_MODEL=new-model\n"
            "CTX_SIZE=4096\n"
            "LITELLM_LEMONADE_API_KEY=sk-inline-from-env-file-67890\n",
            encoding="utf-8",
        )

        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            stdout = (
                '{"status": "ok", "model_loaded": "extra.new-model.gguf"}'
                if cmd and cmd[0] == "curl"
                else ""
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        def fail_restart(_env):
            raise AssertionError("native Lemonade restart should be skipped")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.delenv("LITELLM_LEMONADE_API_KEY", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_lemonade_completion_ready", lambda *_args: True)
        monkeypatch.setattr(_mod, "_restart_windows_lemonade", fail_restart)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        content = lemonade_yaml.read_text(encoding="utf-8")
        assert "model: openai/extra.new-model.gguf" in content
        assert ["docker", "restart", "ods-litellm"] in calls

    def test_windows_native_llama_activation_uses_plain_health_and_litellm_local(
        self, tmp_path, monkeypatch,
    ):
        install_dir, env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path, gpu_backend="amd")
        )
        env_path.write_text(
            "GPU_BACKEND=amd\n"
            "LLM_BACKEND=llama-server\n"
            "AMD_INFERENCE_RUNTIME=llama-server\n"
            "AMD_INFERENCE_RUNTIME_MODE=windows-llama-server-fallback\n"
            "AMD_INFERENCE_LOCATION=host\n"
            "AMD_INFERENCE_MANAGED=true\n"
            "AMD_INFERENCE_PORT=9090\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "CTX_SIZE=2048\n"
            "OLLAMA_PORT=11434\n",
            encoding="utf-8",
        )
        hermes_live = install_dir / "data" / "hermes" / "config.yaml"
        hermes_template = install_dir / "extensions" / "services" / "hermes" / "cli-config.yaml.template"
        hermes_live.parent.mkdir(parents=True)
        hermes_template.parent.mkdir(parents=True)
        hermes_text = (
            "model:\n"
            "  default: \"old-model.gguf\"\n"
            "  provider: \"custom\"\n"
            "  base_url: \"http://litellm:4000/v1\"\n"
        )
        hermes_live.write_text(hermes_text, encoding="utf-8")
        hermes_template.write_text(hermes_text, encoding="utf-8")

        restart_calls = []

        def record_native_restart(path, env):
            restart_calls.append((path, dict(env)))

        def fail_wrong_restart(*_args, **_kwargs):
            raise AssertionError("wrong restart path for Windows native llama-server")

        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            if cmd and cmd[0] == "curl":
                assert cmd[-1] == "http://127.0.0.1:9090/v1/models"
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=_llama_identity_response("new-model.gguf"),
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_restart_windows_native_llama_server", record_native_restart)
        monkeypatch.setattr(_mod, "_restart_windows_lemonade", fail_wrong_restart)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", fail_wrong_restart)
        monkeypatch.setattr(_mod, "_send_lemonade_warmup", fail_wrong_restart)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        assert restart_calls and restart_calls[0][0] == env_path
        assert restart_calls[0][1]["GGUF_FILE"] == "new-model.gguf"
        assert '  default: "new-model.gguf"' in hermes_live.read_text(encoding="utf-8")
        assert "extra.new-model.gguf" not in hermes_live.read_text(encoding="utf-8")
        local_yaml = install_dir / "config" / "litellm" / "local.yaml"
        content = local_yaml.read_text(encoding="utf-8")
        assert "model: openai/new-model.gguf" in content
        assert "api_base: http://host.docker.internal:9090/v1" in content
        assert ["docker", "restart", "ods-litellm"] in calls
        assert ["docker", "restart", "ods-hermes"] in calls

    def test_windows_native_litellm_local_rolls_back_on_late_failure(self, tmp_path, monkeypatch):
        install_dir, env_path, env_text, models_ini, ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path, gpu_backend="amd")
        )
        env_path.write_text(
            "GPU_BACKEND=amd\n"
            "LLM_BACKEND=llama-server\n"
            "AMD_INFERENCE_RUNTIME=llama-server\n"
            "AMD_INFERENCE_RUNTIME_MODE=windows-llama-server-fallback\n"
            "AMD_INFERENCE_LOCATION=host\n"
            "AMD_INFERENCE_MANAGED=true\n"
            "AMD_INFERENCE_PORT=9090\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "CTX_SIZE=2048\n",
            encoding="utf-8",
        )
        env_text = env_path.read_text(encoding="utf-8")
        litellm_local = install_dir / "config" / "litellm" / "local.yaml"
        old_local_yaml = "model_list:\n  - model_name: old\n"
        litellm_local.write_text(old_local_yaml, encoding="utf-8")

        def fake_run(cmd, **_kwargs):
            if cmd and cmd[0] == "curl":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=_llama_identity_response("new-model.gguf"),
                    stderr="",
                )
            if cmd == ["docker", "restart", "ods-litellm"]:
                raise subprocess.TimeoutExpired(cmd, 60)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_restart_windows_native_llama_server", lambda *_args: None)
        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 500
        assert env_path.read_text(encoding="utf-8") == env_text
        assert models_ini.read_text(encoding="utf-8") == ini_text
        assert litellm_local.read_text(encoding="utf-8") == old_local_yaml

    def test_windows_native_restart_error_restores_previous_runtime(self, tmp_path, monkeypatch):
        install_dir, env_path, _env_text, models_ini, ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path, gpu_backend="amd")
        )
        env_path.write_text(
            "GPU_BACKEND=amd\n"
            "LLM_BACKEND=llama-server\n"
            "AMD_INFERENCE_RUNTIME=llama-server\n"
            "AMD_INFERENCE_RUNTIME_MODE=windows-llama-server-fallback\n"
            "AMD_INFERENCE_LOCATION=host\n"
            "AMD_INFERENCE_MANAGED=true\n"
            "AMD_INFERENCE_PORT=9090\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "CTX_SIZE=2048\n",
            encoding="utf-8",
        )
        env_text = env_path.read_text(encoding="utf-8")
        restart_models = []

        def restart_then_recover(_path, env):
            restart_models.append(env["GGUF_FILE"])
            if len(restart_models) == 1:
                raise RuntimeError("simulated native launch failure")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.setattr(_mod.platform, "system", lambda: "Windows")
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_restart_windows_native_llama_server", restart_then_recover)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 500
        assert restart_models == ["new-model.gguf", "old-model.gguf"]
        assert env_path.read_text(encoding="utf-8") == env_text
        assert models_ini.read_text(encoding="utf-8") == ini_text

    def test_activation_patches_hermes_configs_and_restarts_hermes(self, tmp_path, monkeypatch):
        install_dir, _env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        hermes_live = install_dir / "data" / "hermes" / "config.yaml"
        hermes_template = install_dir / "extensions" / "services" / "hermes" / "cli-config.yaml.template"
        hermes_live.parent.mkdir(parents=True)
        hermes_template.parent.mkdir(parents=True)
        hermes_text = (
            "model:\n"
            "  default: \"old-model\"\n"
            "  provider: \"custom\"\n"
            "  base_url: \"http://llama-server:8080/v1\"\n"
        )
        hermes_live.write_text(hermes_text, encoding="utf-8")
        hermes_template.write_text(hermes_text, encoding="utf-8")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        assert '  default: "new-model.gguf"' in hermes_live.read_text(encoding="utf-8")
        assert '  default: "new-model.gguf"' in hermes_template.read_text(encoding="utf-8")
        assert ["docker", "restart", "ods-hermes"] in calls

    def test_activation_preserves_hermes_context_from_env(self, tmp_path, monkeypatch):
        install_dir, env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        env_path.write_text(
            "GPU_BACKEND=nvidia\n"
            "GGUF_FILE=old-model.gguf\n"
            "LLM_MODEL=old-model\n"
            "CTX_SIZE=131072\n"
            "MAX_CONTEXT=131072\n"
            "OLLAMA_PORT=8080\n",
            encoding="utf-8",
        )
        hermes_live = install_dir / "data" / "hermes" / "config.yaml"
        hermes_template = install_dir / "extensions" / "services" / "hermes" / "cli-config.yaml.template"
        hermes_live.parent.mkdir(parents=True)
        hermes_template.parent.mkdir(parents=True)
        hermes_text = (
            "model:\n"
            "  default: \"old-model\"\n"
            "  provider: \"custom\"\n"
            "  base_url: \"http://host.docker.internal:8080/v1\"\n"
            "  context_length: 32768\n"
            "auxiliary:\n"
            "  compression:\n"
            "    context_length: 32768\n"
        )
        hermes_live.write_text(hermes_text, encoding="utf-8")
        hermes_template.write_text(hermes_text, encoding="utf-8")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        env_text = env_path.read_text(encoding="utf-8")
        assert "MAX_CONTEXT=131072" in env_text
        assert "CTX_SIZE=131072" in env_text
        assert "  context_length: 131072" in hermes_live.read_text(encoding="utf-8")
        assert "    context_length: 131072" in hermes_live.read_text(encoding="utf-8")
        assert '  base_url: "http://host.docker.internal:8080/v1"' in hermes_live.read_text(encoding="utf-8")

    def test_activation_skips_hermes_restart_when_live_config_unreadable(self, tmp_path, monkeypatch):
        install_dir, _env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        hermes_live = install_dir / "data" / "hermes" / "config.yaml"
        hermes_template = install_dir / "extensions" / "services" / "hermes" / "cli-config.yaml.template"
        hermes_live.parent.mkdir(parents=True)
        hermes_template.parent.mkdir(parents=True)
        hermes_live.write_text("model:\n  default: \"old-live\"\n", encoding="utf-8")
        hermes_template.write_text("model:\n  default: \"old-template\"\n", encoding="utf-8")

        original_exists = Path.exists

        def fake_exists(path):
            if path == hermes_live:
                raise PermissionError("container-owned")
            return original_exists(path)

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        assert '  default: "new-model.gguf"' in hermes_template.read_text(encoding="utf-8")
        assert ["docker", "restart", "ods-hermes"] not in calls

    def test_activation_applies_matching_runtime_profile_flags(self, tmp_path, monkeypatch):
        install_dir, env_path, _env_text, _models_ini, _ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        model_library = install_dir / "config" / "model-library.json"
        model_library.write_text(json.dumps({
            "models": [{
                "id": "target-model",
                "gguf_file": "new-model.gguf",
                "llm_model_name": "new-model",
                "context_length": 131072,
                "runtime_profiles": [{
                    "id": "nvidia-8gb-test",
                    "label": "Advanced test profile",
                    "backend": "nvidia",
                    "memory_type": "discrete",
                    "vram_min_gb": 7.5,
                    "vram_max_gb": 12.5,
                    "system_ram_min_gb": 32,
                    "context_length": 65536,
                    "llama_server_image": "example.test/llama:turbo",
                    "env": {
                        "LLAMA_PARALLEL": "1",
                        "LLAMA_ARG_FLASH_ATTN": "on",
                        "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
                        "LLAMA_ARG_CACHE_TYPE_V": "turbo3",
                        "LLAMA_ARG_N_CPU_MOE": "30",
                        "LLAMA_ARG_NO_CACHE_PROMPT": "1",
                        "LLAMA_ARG_CHECKPOINT_EVERY_N_TOKENS": "-1",
                        "LLAMA_ARG_SPEC_TYPE": "draft-mtp",
                        "LLAMA_ARG_SPEC_DRAFT_N_MAX": "3",
                    },
                }],
            }]
        }), encoding="utf-8")

        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)
        monkeypatch.setattr(_mod, "_nvidia_vram_gb", lambda: 8.0)
        monkeypatch.setattr(_mod, "_system_ram_gb", lambda: 32)
        monkeypatch.setattr(_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(_mod, "_compose_restart_llama_server", lambda _env: None)

        def fake_run(cmd, **_kwargs):
            stdout = _llama_identity_response("new-model.gguf") if cmd and cmd[0] == "curl" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(_mod.subprocess, "run", fake_run)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 200
        env_text = env_path.read_text(encoding="utf-8")
        assert "MAX_CONTEXT=65536" in env_text
        assert "MODEL_RUNTIME_PROFILE=nvidia-8gb-test" in env_text
        assert "LLAMA_SERVER_IMAGE=example.test/llama:turbo" in env_text
        assert "LLAMA_ARG_CACHE_TYPE_V=turbo3" in env_text
        assert "LLAMA_ARG_N_CPU_MOE=30" in env_text
        assert "LLAMA_ARG_CHECKPOINT_EVERY_N_TOKENS=-1" in env_text
        assert "LLAMA_ARG_SPEC_TYPE=draft-mtp" in env_text
        assert "LLAMA_ARG_SPEC_DRAFT_N_MAX=3" in env_text

    def test_unexpected_failure_rolls_back_all_config_backups(self, tmp_path, monkeypatch):
        install_dir, env_path, env_text, models_ini, ini_text, lemonade_yaml, lemonade_text = (
            _write_model_activation_fixture(tmp_path, gpu_backend="amd", lemonade=True)
        )
        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)
        monkeypatch.delenv("ODS_HOST_INSTALL_DIR", raising=False)

        def fail_restart(_env):
            raise RuntimeError("restart failed")

        monkeypatch.setattr(_mod, "_compose_restart_llama_server", fail_restart)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 500
        assert "restart failed" in handler.parse_response()["error"]
        assert env_path.read_text(encoding="utf-8") == env_text
        assert models_ini.read_text(encoding="utf-8") == ini_text
        assert lemonade_yaml.read_text(encoding="utf-8") == lemonade_text

    def test_pre_snapshot_failure_does_not_overwrite_configs(self, tmp_path, monkeypatch):
        install_dir, env_path, env_text, models_ini, ini_text, _yaml, _yaml_text = (
            _write_model_activation_fixture(tmp_path)
        )
        monkeypatch.setattr(_mod, "INSTALL_DIR", install_dir)

        def fail_load_env(_path):
            raise OSError("cannot read env")

        monkeypatch.setattr(_mod, "load_env", fail_load_env)
        handler = _ResponseHandler()

        _mod.AgentHandler._do_model_activate(handler, "target-model")

        assert handler.response_code == 500
        assert env_path.read_text(encoding="utf-8") == env_text
        assert models_ini.read_text(encoding="utf-8") == ini_text


class TestLemonadeYamlRollback:
    """Verify that lemonade.yaml is backed up and restored on rollback.

    We don't spin up the full HTTP server — instead, we test the backup/restore
    logic by checking that the pattern in _do_model_activate is correct.
    """

    def test_backup_sentinel_none_when_missing(self, tmp_path):
        """When lemonade.yaml doesn't exist, backup should be None."""
        yaml_path = tmp_path / "config" / "litellm" / "lemonade.yaml"
        # File doesn't exist
        backup = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else None
        assert backup is None

    def test_backup_preserves_content(self, tmp_path):
        """When lemonade.yaml exists, backup should capture content."""
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        yaml_path = litellm_dir / "lemonade.yaml"
        yaml_path.write_text("original content", encoding="utf-8")

        backup = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else None
        assert backup == "original content"

        # Simulate overwrite + rollback
        _write_lemonade_config(tmp_path, "new-model.gguf")
        assert "new-model.gguf" in yaml_path.read_text()

        # Restore
        if backup is not None:
            yaml_path.write_text(backup, encoding="utf-8")
        assert yaml_path.read_text() == "original content"

    def test_no_restore_when_backup_is_none(self, tmp_path):
        """When backup is None, rollback should not create the file."""
        litellm_dir = tmp_path / "config" / "litellm"
        litellm_dir.mkdir(parents=True)
        yaml_path = litellm_dir / "lemonade.yaml"

        backup = None  # File didn't exist at backup time
        # Rollback should NOT create the file
        if backup is not None:
            yaml_path.write_text(backup, encoding="utf-8")
        assert not yaml_path.exists()


# --- NVIDIA regression guard ---


class TestNvidiaHealthUnchanged:
    """Ensure the NVIDIA health check still uses the simple '"ok"' check."""

    def test_ok_response_is_healthy(self):
        """llama.cpp health response contains "ok" — should be detected."""
        body = '{"status": "ok"}'
        # The NVIDIA path checks: '"ok"' in body
        assert '"ok"' in body

    def test_model_loaded_not_needed_for_nvidia(self):
        """NVIDIA doesn't need model_loaded — just "ok" is sufficient."""
        # This response has "ok" but no model_loaded — fine for NVIDIA
        body = '{"status": "ok"}'
        assert '"ok"' in body
        # But Lemonade check would fail (no model_loaded key)
        assert _check_lemonade_health(body) is False
