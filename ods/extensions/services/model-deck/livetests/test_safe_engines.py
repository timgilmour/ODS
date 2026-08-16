"""Declared REMOTE engines against the live box (read-only; SAFE tier).

An engine declared on a node-agent entry (sglang-omni Task 7 — the first and
so far only `remote_capable` kind) is the fourth kind of thing the deck can
see, and the first that does not run beside it. These cases prove it is on
the board, keyed the way every other placement is keyed, and — this is the
tier's own guarantee — that LOOKING at it actuates nothing.

**Nothing here fires a verb.** Engine load/unload are not in
`test_safe_sets.py`'s FORBIDDEN_STEPS (that set names park/resume/activate,
and this kind declares neither park nor resume — see the long comment there),
so the safe tier's structural protection does not cover them: a safe-tier
engine case must simply not actuate, and prove it. The proof is the intent
record's VERB SIGNATURE — `state`/`model`/`actor`/`updated_ts`, the four
fields only a deliberate load/unload writes (`app/intent.py`'s `record()`;
`app/routers/__init__.py`'s `_settings_drift` docstring spells out why
`updated_ts` is the stable one: `note_healthy` and a plain reconciler restore
both leave it alone). A verb that fired would move it; a tick that merely
observed cannot.

**Preconditions skip, never fail.** The live deployment declares no remote
engine until an operator does it — `POST /api/nodes/{node}/engines` is a
post-deploy step, not something the seed performs (declaration is management
scope, never ownership: E1 design §6.2). The suite must be green on both
sides of that step, so every case here goes through `remote_engines`, which
skips with a reason when nothing is declared — the same discipline
`test_spark_ds4.py`'s `ds4_window` uses for an absent profile.

Kind-agnostic on purpose: these read the registry for whatever is declared
on a node-agent entry rather than naming a kind or a resource, so they keep
describing the deployment after a second engine — or a second kind — is
declared.
"""

import time

import pytest

from app.lifecycle import STATUSES   # the deck's own status vocabulary

pytestmark = pytest.mark.safe

# Long enough to span several 2 s watcher ticks AND one full observation TTL
# (app.node_clients' REMOTE_OBSERVE_TTL_S, 10 s), so the deck really does
# re-probe the engine inside the window rather than serving this case its
# cache; short enough to stay a read-only case.
WATCH_S = 12.0
READ_INTERVAL_S = 2.0


def declared_remote_engines(deck) -> list[tuple[str, dict]]:
    """Every (node id, engine declaration) pair on a non-local entry, read
    LIVE off the registry — `agent_kind` is the same predicate the world
    assembly itself walks by (app.node_clients.remote_engine_declarations
    skips the local entry)."""
    return [(node["id"], engine)
            for node in deck.get("/api/nodes").json()["nodes"]
            if node["agent_kind"] != "local"
            for engine in node.get("engines") or []]


def engine_key(node_id: str, engine: dict) -> str:
    """`<node>/<resource>` — app.observe.node_key's shape, spelled here
    rather than imported so this case describes the WIRE the UI and an
    operator read, not the function that built it."""
    return f"{node_id}/{engine['resource']}"


def _verb_signature(entry: dict) -> tuple | None:
    """The four intent fields a deliberate load/unload writes, or None when
    the deck has no intent for this key at all. See the module docstring."""
    intent = entry["intent"]
    if intent is None:
        return None
    return tuple(intent[field] for field in
                 ("state", "model", "actor", "updated_ts"))


def _signatures(deck, engines) -> dict[str, tuple | None]:
    lifecycle = deck.get("/api/state").json()["lifecycle"]
    return {engine_key(node_id, engine): _verb_signature(lifecycle[engine_key(node_id, engine)])
            for node_id, engine in engines}


@pytest.fixture
def remote_engines(deck):
    """Every declared remote engine, or a skip naming why there are none."""
    engines = declared_remote_engines(deck)
    if not engines:
        pytest.skip("no engine is declared on any node-agent entry — declaring "
                    "one (POST /api/nodes/{node}/engines) is a post-deploy "
                    "operator step, not part of the seed")
    return engines


def test_every_declared_remote_engine_is_on_the_board(deck, remote_engines):
    """A declared remote engine appears in /api/state under `<node>/<resource>`
    — the SAME key shape as every local placement and every swap slot — with a
    status from the deck's own vocabulary and an observation of its own.

    Keying is the whole point of the case: the deck's keys generalized from
    `local/<resource>` + `<node>/slot0` to `<node>/<resource>`, and a remote
    engine that landed under a bare resource name would collide with a
    same-named local one on a board that shows both.
    """
    state = deck.get("/api/state").json()
    lifecycle = state["lifecycle"]
    remote_tenants = state["world"]["remote_tenants"]

    for node_id, engine in remote_engines:
        key = engine_key(node_id, engine)
        assert key in lifecycle, (
            f"{key} is declared but absent from /api/state lifecycle; "
            f"keys present: {sorted(lifecycle)}")
        assert lifecycle[key]["status"] in STATUSES
        assert lifecycle[key]["reason"], "a status must say why"

        assert key in remote_tenants, (
            f"{key} has a lifecycle entry but no world observation")
        tenant = remote_tenants[key]
        assert (tenant["node_id"], tenant["resource"]) == (node_id,
                                                           engine["resource"])
        assert tenant["engine"] == engine["kind"]


def test_a_declared_remote_engine_gets_a_policy_row(deck, remote_engines):
    """Declaration seeds a policy row for a REMOTE engine exactly as it does
    for a local one (ruling R10), which is what makes idle-release and
    `forget` node-blind-by-key and still unambiguous — a resource name is
    unique deck-wide.

    Read-only: it asserts the row EXISTS and is shaped, never that it holds
    particular values (those are the operator's own declaration).
    """
    policy = deck.get("/api/policy").json()
    for _node_id, engine in remote_engines:
        row = policy.get(engine["resource"])
        assert row is not None, (
            f"{engine['resource']} is declared but has no policy row — "
            "idle-release skips a resource it has no row for")
        assert isinstance(row["pinned"], bool)
        assert isinstance(row["idle_ttl"], int)
        assert isinstance(row["priority"], int)


def test_watching_a_remote_engine_fires_no_verb(deck, remote_engines, events):
    """The safe tier's own guarantee, at the one surface where breaking it
    would reach across the network: reading the board must never actuate.

    A remote engine's observation costs a real node-agent request every tick,
    and the verbs live one route away on the same client — so "the read path
    called up() / down()" is a concrete failure mode here in a way it is not
    for a local GGUF. This watches across several watcher ticks and asserts
    the intent record's verb signature did not move.

    The ONE legitimate mover is the arbiter's own idle-release, which E1
    design §6.4 names as the deck's single deliberate touch-point ("it
    unloads an idle model whoever loaded it") and which records `actor:
    "deck"`. A route call — the only thing this drill could possibly have
    caused — records `actor: "operator"` (app/intent.py's `record` defaults
    that way, deliberately). So a moved signature is tolerated only when the
    arbiter owns it, and is otherwise this case's failure.
    """
    before = _signatures(deck, remote_engines)

    deadline = time.monotonic() + WATCH_S
    while time.monotonic() < deadline:
        deck.get("/api/state").raise_for_status()
        deck.get("/api/nodes").raise_for_status()
        deck.get("/api/policy").raise_for_status()
        time.sleep(READ_INTERVAL_S)

    after = _signatures(deck, remote_engines)
    assert set(after) == set(before), "an engine was declared or forgotten mid-case"
    for key, signature in before.items():
        if after[key] == signature:
            continue
        assert after[key] is not None and after[key][2] == "deck", (
            f"a verb fired on {key} during a read-only pass: "
            f"{signature} -> {after[key]}")

    # The rollback path logs when it declines to undo a speculative record —
    # its presence would mean a verb was attempted and failed
    # (app/routers/serving.py's engine_verb).
    fired = [e for e in events.new_events()
             if e["kind"] == "engine-verb-rollback-skipped"]
    assert not fired, f"a remote engine verb was attempted: {fired}"
