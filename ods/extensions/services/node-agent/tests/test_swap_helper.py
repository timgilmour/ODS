"""swap-helper.sh contract tests (one-shot mode).

The helper is the privileged half of the spark swap split: node-agent (LAN,
unprivileged) writes request.json into a shared ctl dir; the helper (host,
docker rights) validates the profile name against the compose-*.yaml set and
runs swap.sh. These tests drive the script in --once mode with a fake swap.sh
so no docker is involved.
"""

import json
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "swap-helper" / "swap-helper.sh"


def _mk_vllm(tmp_path, profiles=("mm27b", "laguna")):
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    for p in profiles:
        (vllm / f"compose-{p}.yaml").write_text("services: {}\n")
    calls = tmp_path / "swap-calls.log"
    swap = vllm / "swap.sh"
    swap.write_text(f"#!/bin/bash\necho \"$1\" >> {calls}\nexit 0\n")
    swap.chmod(0o755)
    return vllm, calls


def _mk_ctl(tmp_path, profile, req_id="req-1"):
    ctl = tmp_path / "ctl"
    ctl.mkdir(exist_ok=True)
    (ctl / "request.json").write_text(json.dumps({"id": req_id, "profile": profile}))
    return ctl


def _run_once(ctl, vllm):
    return subprocess.run(
        ["bash", str(HELPER), "--once", str(ctl), str(vllm)],
        capture_output=True, text=True, timeout=30,
    )


def _status(ctl):
    return json.loads((ctl / "status.json").read_text())


def test_valid_profile_runs_swap_and_reports_done(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == "laguna"
    s = _status(ctl)
    assert s["state"] == "done"
    assert s["profile"] == "laguna"
    assert s["id"] == "req-1"


def test_unknown_profile_rejected_without_touching_swap(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = _mk_ctl(tmp_path, "evil")
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    assert not calls.exists()
    s = _status(ctl)
    assert s["state"] == "error"
    assert "unknown profile" in s["message"]


def test_traversal_name_rejected_before_filesystem_lookup(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = _mk_ctl(tmp_path, "../../etc/passwd")
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    assert not calls.exists()
    assert _status(ctl)["state"] == "error"


def test_request_consumed_after_processing(tmp_path):
    vllm, _ = _mk_vllm(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")
    _run_once(ctl, vllm)
    assert not (ctl / "request.json").exists()


def test_no_request_is_a_quiet_noop(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    assert not calls.exists()
    assert not (ctl / "status.json").exists()


def test_swap_failure_reported_as_error(tmp_path):
    vllm, _ = _mk_vllm(tmp_path)
    (vllm / "swap.sh").write_text("#!/bin/bash\necho boom >&2\nexit 1\n")
    (vllm / "swap.sh").chmod(0o755)
    ctl = _mk_ctl(tmp_path, "mm27b")
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    s = _status(ctl)
    assert s["state"] == "error"
    assert s["profile"] == "mm27b"


def test_malformed_request_json_reports_error(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    (ctl / "request.json").write_text("{not json")
    r = _run_once(ctl, vllm)
    assert r.returncode == 0, r.stderr
    assert not calls.exists()
    assert _status(ctl)["state"] == "error"
    assert not (ctl / "request.json").exists()


def test_second_instance_blocked_by_lock(tmp_path):
    vllm, calls = _mk_vllm(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")
    slow = tmp_path / "slow-swap-started"
    (vllm / "swap.sh").write_text(
        f"#!/bin/bash\ntouch {slow}\necho \"$1\" >> {calls}\nsleep 3\nexit 0\n")
    (vllm / "swap.sh").chmod(0o755)
    p1 = subprocess.Popen(["bash", str(HELPER), "--once", str(ctl), str(vllm)])
    for _ in range(50):
        if slow.exists():
            break
        import time
        time.sleep(0.1)
    (ctl / "request.json").write_text(json.dumps({"id": "req-2", "profile": "mm27b"}))
    r2 = _run_once(ctl, vllm)
    p1.wait(timeout=30)
    assert r2.returncode != 0  # lock held -> refuses, does not queue-jump
    assert calls.read_text().strip().splitlines() == ["laguna"]
