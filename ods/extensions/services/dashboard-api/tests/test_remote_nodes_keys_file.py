"""ODS_REMOTE_NODE_KEYS_FILE: key delivery off the environment.

Why a file at all, when ODS_REMOTE_NODE_KEYS already works: an env var is
readable by anything that can run `docker inspect` on the dashboard host or
read /proc/<pid>/environ, and this particular var is the aggregate -- one
compromise hands over the bearer key for *every* remote node at once, not one.
A 0600 file mounted in (e.g. from /run/secrets) shrinks that surface.

Follows the shape upstream PR #1365 uses for --token-file: file beats env,
loud warning on a group/world-readable file but still proceed, and the path is
what gets logged and persisted -- never the contents.

Two deliberate departures from their CLI version, both because this runs in a
long-lived service rather than a one-shot command:

- A missing or malformed file degrades to the env map instead of hard-failing.
  Killing dashboard-api over one bad path would take the whole GPU page down,
  including every local card, to report a remote-node misconfiguration.
- Every diagnostic here is warn-once. load_remote_nodes() runs on each 5s poll
  cycle, so a plain warning writes the same line ~17k times a day.
"""
import json
import logging
import os

import remote_nodes

NODE = json.dumps([{"name": "sparky", "url": "http://sparky.test:7720",
                    "key_env": "TEST_NODE_KEY"}])
TWO_NODES = json.dumps([
    {"name": "sparky", "url": "http://sparky.test:7720"},
    {"name": "deadbox", "url": "http://deadbox.test:7720"},
])


def setup_function(_fn):
    remote_nodes._STATE.clear()
    remote_nodes._WARNED_ONCE.clear()


def _write_keys(tmp_path, payload, mode=0o600, name="keys.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    path.chmod(mode)
    return path


def test_keys_file_supplies_the_key(monkeypatch, tmp_path):
    path = _write_keys(tmp_path, {"sparky": "from-file"})
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.delenv("TEST_NODE_KEY", raising=False)
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    assert remote_nodes.load_remote_nodes()[0].key == "from-file"


def test_keys_file_beats_the_env_map_and_key_env(monkeypatch, tmp_path):
    path = _write_keys(tmp_path, {"sparky": "from-file"})
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    assert remote_nodes.load_remote_nodes()[0].key == "from-file"


def test_file_and_env_map_merge_per_node(monkeypatch, tmp_path):
    """Precedence is per node, not whole-map replacement, so an admin can move
    nodes onto the file one at a time without the others going dark."""
    path = _write_keys(tmp_path, {"sparky": "from-file"})
    monkeypatch.setenv("ODS_REMOTE_NODES", TWO_NODES)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS",
                       json.dumps({"deadbox": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    keys = {n.name: n.key for n in remote_nodes.load_remote_nodes()}
    assert keys == {"sparky": "from-file", "deadbox": "from-map"}


def test_missing_file_warns_with_the_path_and_falls_back(monkeypatch, tmp_path,
                                                         caplog):
    missing = tmp_path / "not-there.json"
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(missing))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-map"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert str(missing) in logged


def test_malformed_file_warns_and_falls_back(monkeypatch, tmp_path, caplog):
    path = _write_keys(tmp_path, "{not json at all")
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-map"
    assert str(path) in "\n".join(r.getMessage() for r in caplog.records)


def test_file_holding_a_json_list_is_treated_as_malformed(monkeypatch, tmp_path):
    path = _write_keys(tmp_path, ["sparky", "from-file"])
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    assert remote_nodes.load_remote_nodes()[0].key == "from-map"


def test_empty_file_falls_back(monkeypatch, tmp_path):
    path = _write_keys(tmp_path, "")
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    assert remote_nodes.load_remote_nodes()[0].key == "from-map"


def test_file_contents_never_reach_the_log(monkeypatch, tmp_path, caplog):
    """The failure paths log the path. A secret in a file that fails to parse
    must not ride along in the message."""
    path = _write_keys(tmp_path, '{"sparky": "super-secret-key" oops', mode=0o644)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        remote_nodes.load_remote_nodes()
    assert "super-secret-key" not in "\n".join(
        r.getMessage() for r in caplog.records)


def test_group_readable_file_warns_but_still_resolves(monkeypatch, tmp_path,
                                                      caplog):
    """Soft warning, not a hard fail: a mode nit must not take the node down."""
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o640)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-file"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert str(path) in logged and "640" in logged


def test_world_readable_file_warns(monkeypatch, tmp_path, caplog):
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o644)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        remote_nodes.load_remote_nodes()
    assert any("644" in r.getMessage() for r in caplog.records)


def test_private_file_produces_no_warning(monkeypatch, tmp_path, caplog):
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o600)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        remote_nodes.load_remote_nodes()
    assert [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING] == []


def test_tilde_in_the_path_is_expanded(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    _write_keys(home, {"sparky": "from-file"})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", "~/keys.json")
    assert remote_nodes.load_remote_nodes()[0].key == "from-file"


def test_permission_warning_is_not_repeated_on_every_poll(monkeypatch, tmp_path,
                                                          caplog):
    """load_remote_nodes() runs once per 5s poll cycle."""
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o644)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        for _ in range(5):
            remote_nodes.load_remote_nodes()
    assert len([r for r in caplog.records if "644" in r.getMessage()]) == 1


def test_tightening_the_mode_lets_a_later_regression_warn_again(monkeypatch,
                                                                tmp_path,
                                                                caplog):
    """Warn-once must key on the condition, not fire once per process: an
    operator who fixes the mode and later loosens it again needs to hear it."""
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o644)
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        remote_nodes.load_remote_nodes()
        path.chmod(0o600)
        remote_nodes.load_remote_nodes()
        caplog.clear()
        path.chmod(0o644)
        remote_nodes.load_remote_nodes()
    assert any("644" in r.getMessage() for r in caplog.records)


def test_key_rotation_is_picked_up_without_a_restart(monkeypatch, tmp_path):
    """The file is re-read per load and the secret is never cached anywhere,
    so rotating it is a file write -- no container recreate."""
    path = _write_keys(tmp_path, {"sparky": "old-key"})
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    assert remote_nodes.load_remote_nodes()[0].key == "old-key"
    _write_keys(tmp_path, {"sparky": "rotated-key"})
    assert remote_nodes.load_remote_nodes()[0].key == "rotated-key"


def test_unreadable_file_warns_and_falls_back(monkeypatch, tmp_path, caplog):
    path = _write_keys(tmp_path, {"sparky": "from-file"}, mode=0o000)
    if os.access(path, os.R_OK):  # running as root: the mode cannot bite
        path.chmod(0o600)
        return
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": "from-map"}))
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS_FILE", str(path))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-map"
    assert str(path) in "\n".join(r.getMessage() for r in caplog.records)
    path.chmod(0o600)


def test_no_key_warning_is_not_repeated_on_every_poll(monkeypatch, caplog):
    """Pre-existing warning on the same per-poll path, same spam exposure."""
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.delenv("TEST_NODE_KEY", raising=False)
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS_FILE", raising=False)
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        for _ in range(5):
            remote_nodes.load_remote_nodes()
    assert len([r for r in caplog.records
                if "sparky" in r.getMessage()]) == 1
