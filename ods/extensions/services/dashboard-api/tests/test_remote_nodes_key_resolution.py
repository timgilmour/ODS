"""ODS_REMOTE_NODE_KEYS: the preferred key-delivery mechanism.

Deploy-time problem this closes: docker-compose.base.yml's dashboard-api
service enumerates its environment explicitly, so an admin-chosen key_env
var name (e.g. ODS_NODE_KEY_SPARKY) never reaches the container without a
compose edit per node. ODS_REMOTE_NODE_KEYS is a single JSON object
{"<node name>": "<bearer key>"} that IS forwarded (see docker-compose.base.yml
passthrough) and resolves without any per-node compose change.

Resolution order per node, by name: (1) ODS_REMOTE_NODE_KEYS[name] if
present and non-empty, (2) key_env lookup (existing, must keep working),
(3) empty string. Malformed ODS_REMOTE_NODE_KEYS (bad JSON / not an object)
must log a warning and behave as if empty -- never crash, mirroring
load_remote_nodes' own malformed-config handling.

Keys must never be read from ODS_REMOTE_NODES itself -- only key_env
(a var *name*) or the ODS_REMOTE_NODE_KEYS map may supply key material.
"""
import json
import logging

import remote_nodes

NODE = json.dumps([{"name": "sparky", "url": "http://sparky.test:7720",
                    "key_env": "TEST_NODE_KEY"}])


def setup_function(_fn):
    remote_nodes._STATE.clear()


def test_map_hit_used_when_key_env_absent(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.delenv("TEST_NODE_KEY", raising=False)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS",
                       json.dumps({"sparky": "from-map"}))
    nodes = remote_nodes.load_remote_nodes()
    assert len(nodes) == 1
    assert nodes[0].key == "from-map"


def test_map_miss_falls_back_to_key_env(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS",
                       json.dumps({"someone-else": "not-sparky"}))
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-key-env"


def test_no_map_configured_falls_back_to_key_env(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-key-env"


def test_precedence_map_wins_when_both_set(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS",
                       json.dumps({"sparky": "from-map"}))
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-map"


def test_map_empty_string_value_falls_back_to_key_env(monkeypatch):
    """Present-but-empty in the map does not count as a hit."""
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps({"sparky": ""}))
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-key-env"


def test_neither_configured_yields_empty_string(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.delenv("TEST_NODE_KEY", raising=False)
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == ""


def test_malformed_map_bad_json_falls_back_to_key_env(monkeypatch, caplog):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", "{not json")
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-key-env"
    assert any("ODS_REMOTE_NODE_KEYS" in r.message for r in caplog.records)


def test_malformed_map_not_an_object_falls_back_to_key_env(monkeypatch, caplog):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.setenv("TEST_NODE_KEY", "from-key-env")
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", json.dumps(["sparky", "from-map"]))
    with caplog.at_level(logging.WARNING, logger="remote_nodes"):
        nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == "from-key-env"
    assert any("ODS_REMOTE_NODE_KEYS" in r.message for r in caplog.records)


def test_malformed_map_never_crashes_with_no_fallback_either(monkeypatch):
    monkeypatch.setenv("ODS_REMOTE_NODES", NODE)
    monkeypatch.delenv("TEST_NODE_KEY", raising=False)
    monkeypatch.setenv("ODS_REMOTE_NODE_KEYS", "not even json {{{")
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == ""


def test_keys_never_read_from_topology_json(monkeypatch):
    """A stray 'key' field inline in ODS_REMOTE_NODES must be ignored;
    only key_env (a var name) or the map may supply key material."""
    sneaky = json.dumps([{"name": "sparky", "url": "http://sparky.test:7720",
                          "key": "inline-secret-should-be-ignored"}])
    monkeypatch.setenv("ODS_REMOTE_NODES", sneaky)
    monkeypatch.delenv("ODS_REMOTE_NODE_KEYS", raising=False)
    nodes = remote_nodes.load_remote_nodes()
    assert nodes[0].key == ""
