"""INST I1 disruptive drill: the live create -> load -> serve -> remove round
trip for a deck-created lemonade instance, plus a boot-only drill for a
comfyui instance (D-I1-6's bet made fact: a shared, read-only ComfyUI tree
still boots).

    ./deck-drill -k instances

Skips (never fails) unless the local node-agent + instances-helper overlay
is actually deployed on this box: `control: "instances"` needs to be set on
the local registry entry, its `instance_port_range` needs a free port, and
the node-agent has to answer at all (README's "Engine instances" section
and node-agent/README.md's Deploy section — none of this is set up by a
fresh install; it is an operator opt-in overlay).

Why disruptive: each case creates and tears down a REAL container
(`deck-<resource>`) on the live box, through the full wire path — the
deck's `instances` router, the node-agent's file-protocol channel
(`POST /v1/node/instance/{resource}`), and the host-side instances-helper's
`docker compose` calls against the separate `deck-instances` project.

Left-as-found: the instance never existed before the case and does not
exist after it. The finalizer removes it unconditionally — on pass, fail,
or error alike — the same rule every other disruptive case in this suite
follows (`livetests/README.md`).
"""

import time

import httpx
import pytest

pytestmark = pytest.mark.disruptive

# Container start -> deck reads it as reachable/idle. Generous: a lemonade or
# comfyui image pull/cold-start on a box that has never run this kind before
# can take a while; a slow create is not this drill's regression to catch.
CREATE_TIMEOUT = 240.0
# A model load is the actual cold boot cost (weights into VRAM).
LOAD_TIMEOUT = 300.0
# Compose down + the deck's own declaration/intent teardown.
REMOVE_TIMEOUT = 120.0


def _local_node(deck) -> dict:
    return next(n for n in deck.get("/api/nodes").json()["nodes"] if n["agent_kind"] == "local")


def _engines(deck) -> list[dict]:
    return _local_node(deck).get("engines", [])


def _freest_gpu_index(deck) -> int:
    """Out-of-band: `/api/state`'s `world.gpus`, most free bytes — the drill
    picks a GPU the way an operator would, never GPU 0 by default (a busy
    GPU 0 would make this case fight VRAM contention instead of testing
    instances)."""
    gpus = deck.get("/api/state").json()["world"]["gpus"]
    assert gpus, "no GPUs observed on the local node"
    return max(gpus, key=lambda g: g["free"])["index"]


def _free_port(node: dict) -> bool:
    rng = node.get("instance_port_range")
    if not rng:
        return False
    taken = {e["port"] for e in node.get("engines", []) if e.get("managed")}
    return any(p not in taken for p in range(rng["start"], rng["end"] + 1))


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(2)
    pytest.fail(f"{what} — not met within {timeout_s:.0f}s")


def _create_instance(deck, body: dict, timeout_s: float = 90.0):
    """POST .../instances, retrying ONLY a 409 whose detail names the
    node-agent's capacity-one queue ("already pending"): a previous verb's
    request file is consumed by the helper within its 2 s poll, so a create
    that lands inside that window is not a refusal, just early. Any other
    status (incl. the port-exhaustion / not-managed 409s) returns at once."""
    deadline = time.monotonic() + timeout_s
    while True:
        r = deck.post("/api/nodes/local/instances", json=body)
        if r.status_code == 409 and "already pending" in r.text and time.monotonic() < deadline:
            time.sleep(1)
            continue
        return r


def _remove_instance(deck, resource: str, timeout_s: float = 90.0):
    """DELETE .../instances/{resource} with the same bounded retry on the
    node-agent's capacity-one 409 — the helper runs ONE compose verb at a
    time and a comfyui `down` alone takes ~10-15 s, so a remove queued
    right behind another verb is early, not refused."""
    deadline = time.monotonic() + timeout_s
    while True:
        r = deck.delete(f"/api/nodes/local/instances/{resource}")
        if r.status_code == 409 and "already pending" in r.text and time.monotonic() < deadline:
            time.sleep(1)
            continue
        return r


def _wait_state(deck, resource: str, wanted: set[str], timeout: float) -> None:
    key = f"local/{resource}"
    _wait_for(lambda: deck.get("/api/state").json()["lifecycle"].get(key, {}).get("status") in wanted,
              timeout, f"{key} never reached one of {sorted(wanted)}")


def _wait_absent(deck, resource: str, timeout: float) -> None:
    """No declaration, no intent x observation entry — the instance-remove
    round trip (declaration removed from the store, intent forgotten) both
    happen synchronously inside `DELETE .../instances/{resource}`, so this
    is normally immediate; it polls anyway rather than asserting once,
    matching every other teardown wait in this suite."""
    key = f"local/{resource}"

    def gone() -> bool:
        state = deck.get("/api/state").json()
        return (key not in state["lifecycle"]
                and resource not in {e["resource"] for e in _engines(deck)})

    _wait_for(gone, timeout, f"{resource} never fully disappeared (declaration + lifecycle key)")


@pytest.fixture
def instances_window(deck):
    """Preconditions for an instances drill. Skips, naming each missing
    prereq, rather than forcing any of them — none of these is set up by a
    fresh install (INST I1's node-agent instances channel + host-side
    instances-helper are an opt-in overlay, deployed separately per
    node-agent/README.md)."""
    node = _local_node(deck)
    missing = []
    if node.get("control") != "instances":
        missing.append('the local node is not control: "instances" '
                       '(set it on the Nodes screen, or PUT /api/nodes/local, '
                       'after deploying the node-agent instances overlay + instances-helper)')
    if not _free_port(node):
        missing.append("no free port in instance_port_range (unset, or the range is exhausted)")
    if missing:
        pytest.skip("instances drill prerequisites missing: " + "; ".join(missing))
    return node


def test_lemonade_instance_create_load_serve_remove(deck, drill_model, events, instances_window):
    """create -> the deck observes it idle -> load a model -> the deck reads
    it serving -> an out-of-band chat completion against the instance's OWN
    service DNS name proves it, not the local triple's -> remove -> clean.
    """
    gpu = _freest_gpu_index(deck)
    created = _create_instance(deck, {"kind": "lemonade", "gpu_indices": [gpu], "env": {}})
    assert created.status_code == 201, created.text
    entry = created.json()
    resource = entry["resource"]
    assert entry["gpu_indices"] == [gpu]
    assert entry["managed"] is True

    try:
        # No intent is written on create (D-I1-1) — a fresh instance with
        # nothing loaded and no intent reads 'idle', never 'unmanaged'
        # (that word is reserved for something ELSE having loaded a model
        # into it first).
        _wait_state(deck, resource, {"idle"}, CREATE_TIMEOUT)

        loaded = deck.post(f"/api/tenants/{resource}/load",
                           json={"model": drill_model}, timeout=240.0)
        assert loaded.status_code == 200, loaded.text
        _wait_state(deck, resource, {"serving"}, LOAD_TIMEOUT)

        # Out-of-band proof: from inside the drill container the instance is
        # reachable on ods-network at its own service DNS name — the deck
        # itself never dials the published 127.0.0.1:<port> either (D-I1-3),
        # so this is the same channel the deck's own observation uses, not a
        # side door.
        # The served id is whatever the deck observed as loaded — lemonade
        # names extra-dir files `extra.<file>` (app/state.py's _strip_prefix /
        # the lemonade adapter's id vocabulary), so never the bare filename.
        served = deck.get("/api/state").json()["world"]["tenants"][resource]["model"]
        assert served, "deck reads the instance serving but reports no model id"
        r = httpx.post(f"http://deck-{resource}:8080/api/v1/chat/completions",
                       json={"model": served,
                             "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 8},
                       timeout=60.0)
        assert r.status_code == 200, r.text
    finally:
        removed = _remove_instance(deck, resource)
        assert removed.status_code == 200, removed.text
        _wait_absent(deck, resource, REMOVE_TIMEOUT)

    # clean: no declaration, no intent x observation record for it.
    assert resource not in {e["resource"] for e in _engines(deck)}
    assert f"local/{resource}" not in deck.get("/api/state").json()["lifecycle"]

    events.expect("instance-created", 1.0)
    events.expect("instance-removed", 1.0)


def test_comfyui_instance_boots_with_a_read_only_tree(deck, events, instances_window):
    """D-I1-6's bet made fact, live: a comfyui instance's SHARED tree
    (`user/comfyui.db`, `output/`, `user/__manager/`) is mounted read-only,
    with only its own per-instance user/output/input/temp dirs writable —
    and it still boots to a healthy, idle observation. No load step:
    comfyui's only human verb is `free` — `_ComfyAdapter.human_verbs()`
    returns `frozenset({"free"})`, no load/unload at all (a model just has
    to be resident for its own frontend to use); reaching 'idle' off a
    clean create is the entire proof this case exists for. If this ever
    fails because a custom-node import needs to WRITE the shared tree,
    D-I1-6's recorded fallback (a per-instance copy of the tree) is the
    next move, not an ad-hoc permissions fix here.
    """
    gpu = _freest_gpu_index(deck)
    created = _create_instance(deck, {"kind": "comfyui", "gpu_indices": [gpu], "env": {}})
    assert created.status_code == 201, created.text
    resource = created.json()["resource"]

    try:
        # A RESIDENT comfyui with no intent record reads `unmanaged`, not
        # `idle` (app/lifecycle.py derive_status: "is loaded but the Deck
        # has no intent for it" — comfyui's observation counts a reachable,
        # resident process as loaded, exactly like a hipfire instance that
        # boots serving). `idle` would only appear for a reachable-but-not-
        # resident observation, which comfyui never reports. Either word
        # proves the boot; the tenant state is the out-of-band check.
        _wait_state(deck, resource, {"unmanaged", "idle"}, CREATE_TIMEOUT)
        tenant = deck.get("/api/state").json()["world"]["tenants"][resource]
        assert tenant["state"] in ("idle", "busy"), tenant
    finally:
        removed = _remove_instance(deck, resource)
        assert removed.status_code == 200, removed.text
        _wait_absent(deck, resource, REMOVE_TIMEOUT)

    assert resource not in {e["resource"] for e in _engines(deck)}
    assert f"local/{resource}" not in deck.get("/api/state").json()["lifecycle"]

    events.expect("instance-created", 1.0)
    events.expect("instance-removed", 1.0)
