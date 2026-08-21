"""instances-helper.sh contract tests (one-shot mode).

The helper is the privileged half of the Deck's engine-INSTANCES split
(INST I1): node-agent (LAN, unprivileged) writes instance-req.json into a
shared ctl dir; this helper (host, docker rights) renders the document
through the repo-owned per-kind templates (render_instance.py) and runs
`docker compose` under the SEPARATE `deck-instances` project. These tests
drive the script in --once mode with a fake `docker` on PATH so no real
daemon is ever touched.
"""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "instances-helper" / "instances-helper.sh"
TEMPLATES = Path(__file__).resolve().parents[1] / "instances-helper" / "templates"
DOC = {"resource": "agent", "kind": "hipfire", "gpu_indices": [2], "port": 11500,
       "env": {"HIPFIRE_MODEL": "qwen3.8:27b"}}


class _AnyTs:
    def __eq__(self, other):
        return isinstance(other, str) and bool(other)

    def __repr__(self):
        return "ANY_TS"


ANY_TS = _AnyTs()


def _mk_req(tmp_path, verb, doc=DOC):
    ctl = tmp_path / "ctl"; ctl.mkdir(exist_ok=True)
    (ctl / "instance-req.json").write_text(json.dumps({"verb": verb, "document": doc, "ts": 1.0}))
    return ctl


class Docker:
    """Handle for the fake docker on PATH: its argv log."""

    def __init__(self, bindir, log):
        self.bindir = bindir
        self.log = log

    def lines(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def calls(self, verb):
        return [ln for ln in self.lines() if ln.split(" ")[0] == verb]


def _mk_docker(tmp_path, compose_exit=0):
    """A fake `docker` on PATH: logs `"$*"`, exits `compose_exit` for the
    `compose` subcommand, 0 for anything else."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "docker.log"
    docker = bindir / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> {shlex.quote(str(log))}\n'
        'case "$1" in\n'
        f'  compose) exit {compose_exit} ;;\n'
        'esac\n'
        'exit 0\n'
    )
    docker.chmod(0o755)
    return Docker(bindir, log)


def _run_once(ctl, inst, ods, bindir):
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(["bash", str(HELPER), "--once", str(ctl), str(TEMPLATES), str(inst), str(ods)],
                          capture_output=True, text=True, timeout=30, env=env)


def _status(ctl, resource):
    return json.loads((ctl / f"instance-status-{resource}.json").read_text())


def test_create_renders_then_ups_in_project_deck_instances(tmp_path):
    ctl = _mk_req(tmp_path, "create"); inst = tmp_path / "inst"; inst.mkdir(); ods = tmp_path / "ods"
    docker = _mk_docker(tmp_path)
    r = _run_once(ctl, inst, ods, docker.bindir)
    assert r.returncode == 0, r.stderr
    assert not (ctl / "instance-req.json").exists()            # consumed first
    assert (inst / "agent.yaml").exists()
    assert docker.calls("compose") == [f"compose -p deck-instances -f {inst}/agent.yaml up -d"]
    assert _status(ctl, "agent") == {"resource": "agent", "verb": "create", "ok": True, "error": None, "ts": ANY_TS}


def test_remove_downs_with_the_rendered_file_then_deletes_it(tmp_path):
    ctl = _mk_req(tmp_path, "remove"); inst = tmp_path / "inst"; inst.mkdir(); ods = tmp_path / "ods"
    (inst / "agent.yaml").write_text("{}")
    docker = _mk_docker(tmp_path)
    _run_once(ctl, inst, ods, docker.bindir)
    assert docker.calls("compose") == [f"compose -p deck-instances -f {inst}/agent.yaml down"]
    assert not (inst / "agent.yaml").exists()
    assert _status(ctl, "agent")["ok"] is True


def test_remove_without_a_rendered_file_is_refused_not_guessed(tmp_path):
    ctl = _mk_req(tmp_path, "remove"); inst = tmp_path / "inst"; inst.mkdir()
    docker = _mk_docker(tmp_path)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert docker.calls("compose") == []
    assert _status(ctl, "agent") == {**_status(ctl, "agent"), "ok": False, "error": "no rendered file for 'agent' (never created here, or already removed)"}


def test_move_renders_first_then_down_old_then_up_new(tmp_path):
    ctl = _mk_req(tmp_path, "move", {**DOC, "gpu_indices": [3]}); inst = tmp_path / "inst"; inst.mkdir()
    (inst / "agent.yaml").write_text(json.dumps({"services": {"agent": {"environment": {"ROCR_VISIBLE_DEVICES": "2"}}}}))
    docker = _mk_docker(tmp_path)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert docker.calls("compose") == [f"compose -p deck-instances -f {inst}/agent.yaml down",
                                       f"compose -p deck-instances -f {inst}/agent.yaml up -d"]
    assert json.loads((inst / "agent.yaml").read_text())["services"]["deck-agent"]["environment"]["ROCR_VISIBLE_DEVICES"] == "3"


def test_unknown_kind_writes_a_refusal_and_never_runs_docker(tmp_path):
    ctl = _mk_req(tmp_path, "create", {**DOC, "kind": "nope"}); inst = tmp_path / "inst"; inst.mkdir()
    docker = _mk_docker(tmp_path)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert docker.calls("compose") == []
    assert _status(ctl, "agent")["ok"] is False and "unknown kind" in _status(ctl, "agent")["error"]


def test_unsafe_resource_name_writes_no_result_file_at_all(tmp_path):
    ctl = _mk_req(tmp_path, "create", {**DOC, "resource": "../evil"}); inst = tmp_path / "inst"; inst.mkdir()
    docker = _mk_docker(tmp_path)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert docker.calls("compose") == [] and list(ctl.glob("instance-status-*")) == []
    assert "fails name validation" in (ctl / "instances.log").read_text()


def test_compose_failure_is_reported_not_hidden(tmp_path):
    ctl = _mk_req(tmp_path, "create"); inst = tmp_path / "inst"; inst.mkdir()
    docker = _mk_docker(tmp_path, compose_exit=1)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert _status(ctl, "agent")["ok"] is False and "docker compose up failed" in _status(ctl, "agent")["error"]


def test_stale_result_is_invalidated_before_the_slow_part(tmp_path):
    ctl = _mk_req(tmp_path, "create"); inst = tmp_path / "inst"; inst.mkdir()
    (ctl / "instance-status-agent.json").write_text(json.dumps({"ok": True, "verb": "old"}))
    docker = _mk_docker(tmp_path, compose_exit=1)
    _run_once(ctl, inst, tmp_path / "ods", docker.bindir)
    assert _status(ctl, "agent")["verb"] == "create"


def test_create_stages_a_route_for_hipfire(tmp_path):
    ctl = _mk_req(tmp_path, "create"); inst = tmp_path / "inst"; inst.mkdir(); ods = tmp_path / "ods"
    docker = _mk_docker(tmp_path)
    r = _run_once(ctl, inst, ods, docker.bindir)
    assert r.returncode == 0, r.stderr
    assert _status(ctl, "agent")["ok"] is True
    routes = json.loads((ods / "config" / "litellm" / "extra-routes.json").read_text())
    assert routes == [{"model_name": "qwen3.8:27b", "model": "openai/qwen3.8:27b",
                       "api_base": "http://deck-agent:11435/v1", "_deck_instance": "agent"}]


def test_staging_failure_never_fails_the_verb(tmp_path):
    ctl = _mk_req(tmp_path, "create"); inst = tmp_path / "inst"; inst.mkdir(); ods = tmp_path / "ods"
    litellm_dir = ods / "config" / "litellm"
    litellm_dir.mkdir(parents=True)
    (litellm_dir / "extra-routes.json").mkdir()   # a directory where stage_route.py wants to write a file
    docker = _mk_docker(tmp_path)
    r = _run_once(ctl, inst, ods, docker.bindir)
    assert r.returncode == 0, r.stderr
    assert _status(ctl, "agent")["ok"] is True
    assert "gateway staging failed" in (ctl / "instances.log").read_text()


def test_lock_refuses_a_second_instance(tmp_path):
    ctl = tmp_path / "ctl"; ctl.mkdir()
    inst = tmp_path / "inst"; inst.mkdir()
    ods = tmp_path / "ods"
    docker = _mk_docker(tmp_path)
    lock = ctl / ".instances.lock"
    holder = subprocess.Popen(
        ["bash", "-c", f'exec 9>{shlex.quote(str(lock))}; flock 9; sleep 5'])
    try:
        for _ in range(50):
            if lock.exists():
                break
            time.sleep(0.1)
        time.sleep(0.2)  # give flock a moment to actually acquire
        r = _run_once(ctl, inst, ods, docker.bindir)
        assert r.returncode == 1
        assert "another instance holds the lock" in r.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=10)
