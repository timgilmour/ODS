"""swap-helper.sh contract tests (one-shot mode).

The helper is the privileged half of the spark swap split: node-agent (LAN,
unprivileged) writes request.json into a shared ctl dir; the helper (host,
docker rights) validates the profile name against the compose-*.yaml set and
runs swap.sh. These tests drive the script in --once mode with a fake swap.sh
so no docker is involved.
"""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

# The real constant, not a copy: what the settings API hands back for a
# profile nobody has configured is exactly what the helper has to survive.
from settings_store import EMPTY

HELPER = Path(__file__).resolve().parents[1] / "swap-helper" / "swap-helper.sh"
PROBE = Path(__file__).resolve().parents[1] / "swap-helper" / "harvest_probe.py"

# What the fake `docker exec` prints in place of a real in-container probe.
# Sentinel-wrapped like the real thing (app.harvest._SENTINEL) so the stored
# probe_output is something app.harvest.parse_probe_output could consume.
FAKE_PROBE_STDOUT = (
    "some engine chatter on stdout\n"
    "===MODEL_DECK_HARVEST_PROBE===\n"
    '{"options": [{"flags": ["--max-model-len"], "type": "int"}]}\n'
    "===MODEL_DECK_HARVEST_PROBE==="
)

# A settings document in the Task 3 schema (settings_store.EMPTY's keys).
DOC = {"args": {"max-model-len": "131072"}, "env": {"V": "1"},
       "argv": ["serve", "/model", "--max-model-len", "131072"],
       "service": "aeon-vllm"}


def _mk_vllm(tmp_path, profiles=("mm27b", "laguna"), containers=False):
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    for p in profiles:
        body = "services: {}\n"
        if containers:
            # Teardown is derived from container_name: lines (ported from the
            # live sparky swap.sh), so the settings-owned tests need real ones.
            body = (f"services:\n  aeon-vllm:\n"
                    f"    container_name: fake-{p}-container  # note\n")
        (vllm / f"compose-{p}.yaml").write_text(body)
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


def _mk_settings(tmp_path, profile, doc):
    """Settings dir + <profile>.json. `doc` may be a dict, a raw string (to
    plant a corrupt document), or None for a dir with no document at all."""
    settings = tmp_path / "settings"
    settings.mkdir(exist_ok=True)
    if doc is not None:
        text = doc if isinstance(doc, str) else json.dumps(doc)
        (settings / f"{profile}.json").write_text(text)
    return settings


class Docker:
    """Handles for the fake docker on PATH: its argv log and exec stdin."""

    def __init__(self, bindir, log, exec_stdin, witness):
        self.bindir = bindir
        self.log = log            # one line per invocation, args space-joined
        self.exec_stdin = exec_stdin  # what `docker exec` was fed on stdin
        self.witness = witness    # status.json as of the `docker compose` call

    def lines(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def calls(self, verb):
        return [ln for ln in self.lines() if ln.split(" ")[0] == verb]


def _mk_docker(tmp_path, exec_exit=0, inspect_exit=0, image="sha256:fake"):
    """A fake `docker` on PATH.

    Real docker IS installed on the dev box, so every test that can reach the
    settings-owned launch branch must run with this shadowing it — otherwise
    `docker rm -f` would talk to the live daemon.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "docker.log"
    exec_stdin = tmp_path / "docker-exec-stdin.txt"
    witness = tmp_path / "status-at-compose.json"
    docker = bindir / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> {shlex.quote(str(log))}\n'
        'case "$1" in\n'
        # Snapshot the status file mid-launch so a test can prove the
        # settings-owned path published `swapping` before it launched.
        f'  compose) cp {shlex.quote(str(tmp_path / "ctl" / "status.json"))} '
        f'{shlex.quote(str(witness))} 2>/dev/null ;;\n'
        f'  inspect) printf "%s\\n" {shlex.quote(image)}; exit {inspect_exit} ;;\n'
        f'  exec) cat > {shlex.quote(str(exec_stdin))}; '
        f'printf "%s\\n" {shlex.quote(FAKE_PROBE_STDOUT)}; exit {exec_exit} ;;\n'
        'esac\n'
        'exit 0\n'
    )
    docker.chmod(0o755)
    return Docker(bindir, log, exec_stdin, witness)


def _run_once(ctl, vllm, settings=None, bindir=None):
    argv = ["bash", str(HELPER), "--once", str(ctl), str(vllm)]
    if settings is not None:
        argv.append(str(settings))
    env = None
    if bindir is not None:
        env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=30, env=env,
    )


def _run_once4(ctl, vllm, settings, bindir):
    """--once with the optional 4th positional (settings dir)."""
    return _run_once(ctl, vllm, settings=settings, bindir=bindir)


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


# --- settings-owned launch, probe and catalog (optional 4th positional) -----


def test_three_arg_invocation_is_byte_identical_to_today(tmp_path):
    """No 4th arg -> swap.sh delegation, no override, no probe, no catalog."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    # A settings dir exists on disk with a perfectly good document; without
    # the 4th positional the helper must not know or care.
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once(ctl, vllm, bindir=docker.bindir)

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert calls.read_text().strip() == "laguna"
    assert docker.lines() == []                       # docker never invoked
    assert not (vllm / "settings-laguna.override.yaml").exists()
    assert not (settings / "catalog-laguna.json").exists()
    s = _status(ctl)
    assert s["state"] == "done"
    assert s["profile"] == "laguna"
    assert s["id"] == "req-1"


def test_no_settings_file_delegates_to_swap_sh(tmp_path):
    """4th arg given but no <profile>.json -> swap.sh runs, docker compose
    does NOT, and any stale override file is removed."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", None)   # dir, no document
    override = vllm / "settings-laguna.override.yaml"
    override.write_text('{"services": {"aeon-vllm": {"command": ["stale"]}}}')
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert calls.read_text().strip() == "laguna"        # swap.sh owned it
    assert docker.calls("compose") == []                # helper did not launch
    assert not override.exists()                        # stale override gone
    assert _status(ctl)["state"] == "done"
    # The harvest is orthogonal to WHICH branch launched the profile: the
    # container exists either way, so a delegated launch is still catalogued.
    assert (settings / "catalog-laguna.json").exists()


@pytest.mark.parametrize("bad", [
    "{not json",                                        # unparseable
    json.dumps({"args": {}, "env": {}, "argv": "serve", "service": "x"}),
    json.dumps({"args": {}, "env": {}, "argv": ["ok"], "service": ""}),
    # argv non-empty on purpose: an empty one is the "asserts nothing" shape
    # below, which falls back BEFORE env is ever inspected.
    json.dumps({"args": {}, "env": {"K": ["list"]}, "argv": ["s"], "service": "x"}),
    json.dumps(["not", "an", "object"]),
])
def test_corrupt_settings_delegates_and_removes_stale_override(tmp_path, bad):
    """A settings bug must never break a swap."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", bad)
    override = vllm / "settings-laguna.override.yaml"
    override.write_text('{"services": {"aeon-vllm": {"command": ["stale"]}}}')
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == "laguna"        # today's exact path
    assert docker.calls("compose") == []
    assert not override.exists()
    assert _status(ctl)["state"] == "done"
    # Falling back is safe but it is not silent -- an operator has to be able
    # to see WHY a settings document stopped being honoured.
    assert "unusable settings document" in r.stderr


@pytest.mark.parametrize("doc, why", [
    (EMPTY, "settings_store.EMPTY -- what the API hands back for a profile "
            "nobody has configured yet"),
    ({**EMPTY, "service": "aeon-vllm"}, "a service, but nothing to say about it"),
    ({"args": {}, "env": {"V": "1"}, "argv": [], "service": "aeon-vllm"},
     "an env but no command"),
    ({**EMPTY, "argv": ["serve", "/model"]}, "a command but no service to put it on"),
])
def test_document_that_asserts_nothing_falls_back_silently(tmp_path, doc, why):
    """A document that asserts nothing is every profile's STARTING state, not
    a fault.

    It must not actuate: compose REPLACES `command` rather than merging it,
    so rendering `command: []` would launch the profile on the image's
    default CMD and still report `done` -- a settings document breaking a
    swap. And it must not warn: this shape arrives on every swap of an
    unconfigured profile, so a diagnostic here is noise that trains operators
    to ignore the line that does mean something."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", doc)
    override = vllm / "settings-laguna.override.yaml"
    override.write_text('{"services": {"aeon-vllm": {"command": ["stale"]}}}')
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == "laguna", why   # swap.sh owned it
    assert docker.calls("compose") == []                # the helper did NOT
    assert not override.exists()                        # stale override gone
    assert r.stderr == ""                               # and it said nothing
    assert _status(ctl)["state"] == "done"


def test_unwritable_override_falls_back_instead_of_failing_the_swap(tmp_path):
    """The document is fine but the override cannot be written (read-only
    vllm dir). That is still a settings fault, so it takes the settings
    fault's exit: swap.sh, not a failed swap."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")
    vllm.chmod(0o555)
    try:
        r = _run_once4(ctl, vllm, settings, docker.bindir)
    finally:
        vllm.chmod(0o755)

    assert r.returncode == 0, r.stderr
    assert calls.read_text().strip() == "laguna"
    assert docker.calls("compose") == []
    assert not (vllm / "settings-laguna.override.yaml").exists()
    assert _status(ctl)["state"] == "done"
    assert "cannot write override" in r.stderr


def test_valid_settings_helper_owns_the_launch(tmp_path):
    """Override written with the document's service/command/environment;
    docker compose invoked with BOTH -f files; swap.sh NOT invoked;
    teardown derived from container_name lines across compose-*.yaml."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert not calls.exists()                           # swap.sh untouched
    override = vllm / "settings-laguna.override.yaml"
    assert override.exists()
    compose = docker.calls("compose")
    assert len(compose) == 1
    assert compose[0] == (
        f"compose -f {vllm}/compose-laguna.yaml -f {override} up -d")
    # Teardown: every container_name across compose-*.yaml, comment and all
    # stripped -- never a hard-coded list.
    rm = docker.calls("rm")
    assert len(rm) == 1
    assert set(rm[0].split(" ")[2:]) == {"fake-mm27b-container",
                                         "fake-laguna-container"}
    assert _status(ctl)["state"] == "done"


def test_override_is_json_syntax_yaml_with_full_command(tmp_path):
    """json.load the override; assert
    services["aeon-vllm"]["command"] == DOC["argv"] and
    ["environment"] == DOC["env"]."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)
    assert r.returncode == 0, r.stderr

    override = json.loads((vllm / "settings-laguna.override.yaml").read_text())
    assert set(override) == {"services"}
    assert set(override["services"]) == {"aeon-vllm"}
    service = override["services"]["aeon-vllm"]
    assert service["command"] == DOC["argv"]
    assert service["environment"] == DOC["env"]
    # The override asserts the command and the environment and NOTHING else:
    # image/volumes/devices stay the node operator's, in compose-<p>.yaml.
    assert set(service) == {"command", "environment"}
    # Never a compose-*.yaml name: swapctl.list_profiles globs that pattern
    # and would list the override as a ghost profile.
    assert not list(vllm.glob("compose-laguna.override.yaml"))
    assert sorted(p.name for p in vllm.glob("compose-*.yaml")) == [
        "compose-laguna.yaml", "compose-mm27b.yaml"]


def test_vllm_profile_probe_writes_catalog(tmp_path):
    """After a successful launch of an engine=vllm profile, catalog-<p>.json
    exists with image_id sha256:fake and non-empty probe_output."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)
    assert r.returncode == 0, r.stderr

    catalog = json.loads((settings / f"catalog-laguna.json").read_text())
    assert set(catalog) == {"image_id", "harvested_ts", "engine", "probe_output"}
    assert catalog["image_id"] == "sha256:fake"
    assert catalog["engine"] == "vllm"
    assert catalog["probe_output"].strip() == FAKE_PROBE_STDOUT
    # read_newest_catalog sorts these as plain strings, so the format matters.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        catalog["harvested_ts"])
    # The probe is fed on stdin, so `docker exec` must attach it: without -i
    # the in-container python3 reads EOF and the catalog is empty forever.
    execs = docker.calls("exec")
    assert len(execs) == 1
    assert execs[0] == f"exec -i fake-laguna-container python3 -"
    assert docker.exec_stdin.read_text() == PROBE.read_text()
    # inspect ran against the same container, not a hard-coded name.
    assert docker.calls("inspect") == [
        "inspect -f {{.Image}} fake-laguna-container"]


def test_probe_output_is_stored_verbatim_however_hostile(tmp_path):
    """Probe output is arbitrary engine stdout. It reaches the catalog as a
    file path, never interpolated into the python that writes the catalog --
    so a payload carrying that python's own heredoc terminator, triple
    quotes, backslashes and shell substitutions round-trips unharmed instead
    of ending the heredoc early or executing."""
    hostile = (
        '===MODEL_DECK_HARVEST_PROBE===\n'
        '{"options": [{"help": "path like C:\\\\Users \\"quoted\\" \'single\'"}]}\n'
        'PYEOF\n'
        '"""$(touch /tmp/pwned-by-swap-helper)`id`${HOME}\n'
        'EOF\n'
        '===MODEL_DECK_HARVEST_PROBE==='
    )
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    (docker.bindir / "docker").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$*" >> {shlex.quote(str(docker.log))}\n'
        'case "$1" in\n'
        '  inspect) printf "%s\\n" "sha256:fake" ;;\n'
        f'  exec) cat > /dev/null; printf "%s\\n" {shlex.quote(hostile)} ;;\n'
        'esac\n'
        'exit 0\n')
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    catalog = json.loads((settings / "catalog-laguna.json").read_text())
    assert catalog["probe_output"] == hostile + "\n"
    assert not Path("/tmp/pwned-by-swap-helper").exists()


def test_quoted_container_name_is_unquoted_for_teardown_and_probe(tmp_path):
    """container_name: "spark-foo"  # note -- quotes and comment stripped, or
    docker rm -f gets a name that matches nothing and fails into `|| true`."""
    vllm, _ = _mk_vllm(tmp_path, profiles=("laguna",))
    (vllm / "compose-laguna.yaml").write_text(
        'services:\n  aeon-vllm:\n    container_name: "fake-quoted"   # why\n')
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert docker.calls("rm") == ["rm -f fake-quoted"]
    assert docker.calls("exec") == ["exec -i fake-quoted python3 -"]


def test_compose_file_without_container_name_warns_into_swap_log(tmp_path):
    """A profile whose container cannot be torn down is the 2026-08-04 bug
    (spark-ds4 kept holding :8000). It is called out where the status message
    already sends operators -- swap.log -- not onto the daemon's stderr."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    (vllm / "compose-mm27b.yaml").write_text("services:\n  x:\n    image: y\n")
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    log = (ctl / "swap.log").read_text()
    assert "compose-mm27b.yaml has no container_name" in log
    assert docker.calls("rm") == ["rm -f fake-laguna-container"]
    assert _status(ctl)["state"] == "done"


def test_non_vllm_profile_is_not_probed(tmp_path):
    """profiles.json marks the profile engine=ds4 -> no docker exec, no
    catalog file."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    (vllm / "profiles.json").write_text(json.dumps({
        "laguna": {"engine": "ds4"}, "mm27b": {"engine": "vllm"}}))
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert docker.calls("exec") == []
    assert docker.calls("inspect") == []
    assert not (settings / "catalog-laguna.json").exists()
    assert _status(ctl)["state"] == "done"   # the swap itself still happened
    assert len(docker.calls("compose")) == 1


def test_probe_failure_leaves_status_done(tmp_path):
    """fake docker exits 1 on exec -> status.json is still 'done'; the swap
    outcome is untouched."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path, exec_exit=1)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    s = _status(ctl)
    assert s["state"] == "done"
    assert s["message"] == "swap launched"
    assert not (settings / "catalog-laguna.json").exists()  # no half catalog
    assert "harvest: probe failed for laguna" in r.stderr


def test_probe_skipped_when_image_id_is_unavailable(tmp_path):
    """docker inspect failing (container already gone) is a no-catalog, not
    an error: an image_id-less catalog would claim a false engine identity."""
    vllm, _ = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path, inspect_exit=1, image="")
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)

    assert r.returncode == 0, r.stderr
    assert len(docker.calls("compose")) == 1   # settings-owned launch happened
    assert docker.calls("inspect") == ["inspect -f {{.Image}} fake-laguna-container"]
    assert docker.calls("exec") == []
    assert not (settings / "catalog-laguna.json").exists()
    assert _status(ctl)["state"] == "done"


def test_absent_docker_reports_error_and_survives(tmp_path):
    """A settings-owned launch with no usable docker fails the swap as an
    `error` -- and the helper still exits 0, so the --daemon loop lives.

    The stand-in exits 127 with the shell's own not-found message: to every
    caller in this script that is indistinguishable from docker being absent
    from PATH, and PATH cannot simply be emptied here (bash, sed and python3
    live in the same directory docker does).
    """
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text("#!/bin/bash\necho 'docker: command not found' >&2\nexit 127\n")
    docker.chmod(0o755)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, bindir)

    assert r.returncode == 0
    assert r.stderr == ""                       # noise goes to swap.log
    assert _status(ctl)["state"] == "error"
    assert not calls.exists()                   # no silent retry via swap.sh
    assert not (settings / "catalog-laguna.json").exists()
    assert "command not found" in (ctl / "swap.log").read_text()


def test_settings_status_states_match_todays(tmp_path):
    """The settings-owned launch writes swapping -> done, same as the
    delegated path."""
    vllm, calls = _mk_vllm(tmp_path, containers=True)
    settings = _mk_settings(tmp_path, "laguna", DOC)
    docker = _mk_docker(tmp_path)
    ctl = _mk_ctl(tmp_path, "laguna")

    r = _run_once4(ctl, vllm, settings, docker.bindir)
    assert r.returncode == 0, r.stderr
    assert json.loads(docker.witness.read_text())["state"] == "swapping"
    assert _status(ctl)["state"] == "done"
    owned = _status(ctl)

    # Same profile, same request shape, delegated path: identical states and
    # identical messages, so nothing downstream can tell the branches apart.
    witness = tmp_path / "status-at-swap.json"
    swap = vllm / "swap.sh"
    swap.write_text(f"#!/bin/bash\ncp {ctl}/status.json {witness} 2>/dev/null\n"
                    f"echo \"$1\" >> {calls}\nexit 0\n")
    swap.chmod(0o755)
    (settings / "laguna.json").unlink()
    ctl = _mk_ctl(tmp_path, "laguna", req_id="req-2")

    r = _run_once4(ctl, vllm, settings, docker.bindir)
    assert r.returncode == 0, r.stderr
    assert json.loads(witness.read_text())["state"] == "swapping"
    delegated = _status(ctl)

    assert owned["state"] == delegated["state"] == "done"
    assert owned["message"] == delegated["message"]
