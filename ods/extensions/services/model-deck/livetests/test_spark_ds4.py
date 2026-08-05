"""D6: deck-driven spark swap round trip — ds4 -> heretic -> ds4.

Why this case exists. ds4 is the first non-vLLM *inference* profile on the
spark, and it arrived with two failure classes the vLLM-only machinery had
never had to survive:

* the port fight — swap.sh's teardown named vLLM's containers literally, so
  a ds4 container outlived its own swap and kept holding :8000 while the
  incoming profile tried to bind it;
* the mid-generation kill — the busy guard knew only vLLM's two gauges, so
  a swap away from ds4 scraped :8000, matched nothing, read "0 busy" and
  would have killed whatever was generating.

D6 proves both against the real box. The busy guard is asserted in BOTH
directions on purpose: a refusal on its own cannot tell a working guard
from one that refuses everything, so the same swap must be REFUSED while a
generation is genuinely in flight and ACCEPTED once it drains.

Container-level checks. The drill has no docker reach onto sparky — the
node-agent deliberately holds no docker socket (the host-helper split, see
app/engines/spark.py), so conftest's ``docker`` fixture cannot be pointed
at it. "The outgoing container is gone" is therefore asserted from the
engine side: every spark profile runs ``network_mode: host``, so :8000 has
exactly one owner, and an answer identifying the incoming model — plus the
disappearance of the outgoing engine's own gauge from /metrics — is proof
the outgoing container released the socket. The node-agent's own view
(``/api/spark/status``, a pass-through of /v1/node/profiles and
/v1/node/serving) is asserted next to it.

Teardown is unconditional AND asserted: the spark ends this case serving
ds4 with the deck's intent recording it, because ds4 is the standing intent
on this box and a drill that left the spark on heretic — or on nothing —
would recreate exactly the outage class this line of work exists to
prevent.

Runtime: expect 7-12 minutes, and budget for more. The heretic leg alone is
~5 min warm since the 2026-08-02 cache mounts and up to ~15 min if
FlashInfer autotunes cold (D5, the other spark round trip, measured 901.5 s
end to end); the busy leg holds a real ~1 min generation open; the ds4 leg
boots in ~30 s.

The strict-mode third path — a 200 scrape with no matching gauge, which
must refuse rather than read as idle — is deliberately NOT staged here:
faking it live would mean breaking ds4's metrics on purpose. It is covered
by unit rows over SparkClient.busy_requests; this case covers the two paths
only the live box can show.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.observe import SPARK_SLOT_KEY  # the deck's own key for this slot

pytestmark = pytest.mark.disruptive

DS4 = "ds4"
# The round-trip partner. A vllm profile on purpose: leg 2 then swaps away
# from vLLM, so one run exercises both sides of the engine-aware guard.
PARTNER = "heretic"

# vllm boot-back: ~5 min warm, ~13-15 min if FlashInfer autotunes from cold.
# Same budget D5 uses, and the same reason.
SWAP_TIMEOUT = 900
# ~22 tok/s containerized => ~45 s of generation. Long enough that the swap
# attempt lands unambiguously mid-flight, short enough not to dominate the
# run; the case waits this out before the accepted leg.
BUSY_TOKENS = 1024
BUSY_PROMPT = ("Write an exhaustive, section-by-section technical history of "
               "GPU memory architectures. Be thorough; do not summarise and "
               "do not stop early.")
DRAIN_TIMEOUT = 600
# The deck derives the slot's status from a cached observation refreshed on
# a tick, so "serving" trails a successful swap by a few seconds.
DERIVE_TIMEOUT = 120

# Helpers are local, as they are in every other disruptive module (D5 keeps
# its own identical trio; the suite's modules are self-contained).


def _status(deck) -> dict:
    return deck.get("/api/spark/status").json()


def _slot(deck) -> dict:
    return deck.get("/api/state").json()["lifecycle"][SPARK_SLOT_KEY]


def _swap(deck, profile: str) -> dict:
    r = deck.post("/api/spark/swap", json={"profile": profile})
    assert r.status_code == 200, f"swap to {profile!r} refused: {r.status_code} {r.text}"
    return r.json()


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(2)
    pytest.fail(f"{what} — not met within {timeout_s:.0f}s")


def _wait_serving(deck, profile: str, timeout_s: float = SWAP_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = _status(deck)
        if s["serving"]["endpoint_ok"] and s["serving"]["model"] == profile:
            return s
        time.sleep(10)
    pytest.fail(f"{profile} not serving within {timeout_s}s: {_status(deck)}")


@pytest.fixture
def ds4_window(deck, litellm_direct, spark_serving):
    """Preconditions for the round trip; skips rather than forces any of them.

    Returns the model id ds4 answers /v1/models with, so the teardown can
    assert the box came back to the same SERVED identity instead of to a
    literal that could drift.
    """
    status = deck.get("/api/spark/status")
    if status.status_code == 503:
        pytest.skip("no spark configured on this deck")
    status.raise_for_status()
    body = status.json()

    profiles = {p["name"]: p for p in body["profiles"] if isinstance(p, dict)}
    missing = [n for n in (DS4, PARTNER) if n not in profiles]
    if missing:
        pytest.skip(f"profile(s) {missing} absent from the node — nothing to round-trip")
    if profiles[DS4].get("engine") != "ds4":
        pytest.skip(f"ds4 profile self-reports engine {profiles[DS4].get('engine')!r}; "
                    "the strict busy guard keys on 'ds4'")

    serving = body["serving"]
    if not (serving["endpoint_ok"] and serving["model"] == DS4):
        # Not a failure: D6 both starts and ends on ds4, and must not move a
        # box it did not find there — the teardown's known-good state would
        # otherwise be a state change nobody asked for.
        pytest.skip(f"spark is serving {serving['model']!r} "
                    f"(endpoint_ok={serving['endpoint_ok']}) — D6 runs only from ds4")

    if spark_serving.host in litellm_direct.default_api_base():
        pytest.skip("litellm's default route targets the spark serving endpoint — "
                    "the swap guard refuses that (force-proof), and swapping would "
                    "break every default-route caller")

    inflight = spark_serving.inflight()
    if inflight is None:
        pytest.skip("ds4's in-flight gauge is absent from the serving port's /metrics "
                    "— something other than ds4 owns :8000")
    if inflight != 0:
        pytest.skip(f"busy-skip: ds4 already has {inflight} in-flight request(s)")

    served = spark_serving.served_model()
    assert served, "ds4 reads as serving but /v1/models lists no model"
    return served


@pytest.fixture
def restore_ds4(deck, spark_serving, ds4_window):
    """UNCONDITIONAL teardown: the spark ends serving ds4, and says so.

    Runs on pass, fail and error alike. It depends on ``ds4_window``, so a
    skipped precondition never arms it — a skip cannot move the box.

    Restoring through the deck's own swap route rather than sparky's
    swap.sh is deliberate, for the same reason ``restore_hipfire`` resumes
    through the deck: that is the action that records intent 'loaded' for
    the slot, so the reconciler is left with a truthful record instead of
    one naming a profile nobody is running.
    """
    served = ds4_window
    yield

    serving = _status(deck)["serving"]
    if not (serving["endpoint_ok"] and serving["model"] == DS4):
        r = deck.post("/api/spark/swap", json={"profile": DS4})
        if r.status_code == 409:
            # Guard-refused: a boot this drill interrupted, or a request it
            # left in flight. force is the documented recovery path for
            # both, and teardown is the only place it belongs — the body
            # must never force, or it would stop testing the guard.
            r = deck.post("/api/spark/swap", json={"profile": DS4, "force": True})
        assert r.status_code == 200, f"TEARDOWN FAILED: swap back to ds4: {r.text}"
        _wait_serving(deck, DS4)

    assert spark_serving.served_model() == served, \
        f"TEARDOWN FAILED: :8000 serves {spark_serving.served_model()!r}, not {served!r}"
    _wait_for(lambda: _slot(deck)["status"] == "serving", DERIVE_TIMEOUT,
              f"TEARDOWN FAILED: the deck does not read {SPARK_SLOT_KEY} as serving")
    intent = _slot(deck)["intent"]
    assert intent["model"] == DS4, f"TEARDOWN FAILED: intent is {intent}"


def test_d6_spark_ds4_round_trip(deck, spark_serving, ds4_window, lemonade_guard,
                                 restore_ds4):
    """ds4 -> heretic -> ds4 through the deck, with the busy guard proven
    refusing mid-generation and allowing once idle, and :8000 handed over
    cleanly on both legs."""
    # lemonade_guard: the heretic leg alone can outlast lemonade's 900 s
    # idle TTL, so the deck's idle-release watcher may legitimately unload
    # lemonade mid-drill and trip box_bookend's left-as-found check (found
    # live on D5, 2026-07-31). Function-scoped teardown runs before the
    # session bookend's closing snapshot, so the box is left as found.
    served = ds4_window
    ds4_profile = next(p for p in _status(deck)["profiles"] if p["name"] == DS4)
    assert ds4_profile["container"], \
        "node-agent lists no container for ds4 — swap.sh's teardown derives " \
        "what to remove from compose-ds4.yaml's container_name: (this field " \
        "is a parallel agreeing copy, not the source), and a null here means " \
        "the two have drifted apart, which is how the port fight happens"

    # --- the guard REFUSES while a generation is in flight ----------------
    with ThreadPoolExecutor(max_workers=1) as pool:
        generation = pool.submit(spark_serving.completion, served, BUSY_PROMPT,
                                 BUSY_TOKENS)
        _wait_for(lambda: (spark_serving.inflight() or 0) >= 1, 120,
                  "ds4 never reported the drill's request as in flight")

        refused = deck.post("/api/spark/swap", json={"profile": PARTNER})
        assert refused.status_code == 409, \
            f"swap away from a BUSY ds4 was not refused: {refused.status_code} {refused.text}"
        assert "in-flight" in refused.json()["detail"], refused.text
        # A refusal must be a refusal: nothing may have started moving.
        assert spark_serving.served_model() == served, "the refused swap moved the box"
        assert _status(deck)["serving"]["model"] == DS4
    # Leaving the pool waits for the generation, so the accepted leg below
    # starts from a genuinely idle engine rather than a lucky race.

    body = generation.result()
    assert body["usage"]["completion_tokens"] > 0, \
        "the drill's 'in-flight' generation produced no tokens — the 409 above " \
        "proves nothing"
    _wait_for(lambda: spark_serving.inflight() == 0, DRAIN_TIMEOUT,
              "ds4 never went idle again")

    # --- the SAME swap is ACCEPTED once ds4 is idle ------------------------
    _swap(deck, PARTNER)
    _wait_serving(deck, PARTNER)
    assert spark_serving.served_model() == PARTNER, \
        "the spark's serving port does not answer as the incoming profile"
    assert spark_serving.inflight() is None, \
        "ds4's gauge is still on :8000 — the outgoing container kept the port"
    assert _status(deck)["swap_status"]["profile"] == PARTNER

    # --- and back ----------------------------------------------------------
    _swap(deck, DS4)
    _wait_serving(deck, DS4)
    assert spark_serving.served_model() == served
    assert spark_serving.inflight() == 0, \
        "ds4 serves again but its in-flight gauge is missing — the strict busy " \
        "guard would refuse every future swap away from it"
    assert _status(deck)["swap_status"]["profile"] == DS4
