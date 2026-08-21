import importlib.util, json, pathlib, pytest
from app.engine_kinds import (KNOWN_KINDS, INSTANCE_CONTAINER_PREFIX, INSTANCE_INTERNAL_PORT,
                              instance_env_schema, kind_instantiable)

TEMPLATES = pathlib.Path(__file__).resolve().parents[2] / "node-agent" / "instances-helper" / "templates"
pytestmark = pytest.mark.skipif(not TEMPLATES.is_dir(), reason="needs the full repo checkout")


def test_every_instantiable_kind_has_a_template_and_vice_versa():
    kinds = json.loads((TEMPLATES / "kinds.json").read_text())
    assert set(kinds) == {k for k in KNOWN_KINDS if kind_instantiable(k)}


@pytest.mark.parametrize("kind", [k for k in KNOWN_KINDS if kind_instantiable(k)])
def test_template_internal_port_and_env_allowlist_match_the_descriptor(kind):
    kinds = json.loads((TEMPLATES / "kinds.json").read_text())
    t = json.loads((TEMPLATES / kinds[kind]).read_text())
    assert t["internal_port"] == INSTANCE_INTERNAL_PORT[kind]
    assert set(t["env_allow"]) == set(instance_env_schema(kind))


def _helper_module(name):
    path = TEMPLATES.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", ["render_instance", "stage_route"])
def test_helper_container_prefix_matches_the_deck(script):
    """The compose service name the helper writes (== the container name,
    deck-<resource>) is the host every deck connection URL and staged
    litellm route dials — the two sides must spell the prefix identically."""
    assert _helper_module(script).CONTAINER_PREFIX == INSTANCE_CONTAINER_PREFIX
