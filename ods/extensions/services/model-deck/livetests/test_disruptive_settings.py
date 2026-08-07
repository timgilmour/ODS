"""D11 + D12: remote settings apply and adopt-then-own, against sparky.

Disruptive: D11 performs FOUR REAL swaps (~5m17s warm each, same
SWAP_TIMEOUT budget test_spark_ds4.py and test_spark_comfyui.py both use),
so budget ~25 min for this module end to end:

  1. swap the vLLM profile in,
  2. reload — ships the drill's override, re-swaps, engine serves it,
  3. reload again after the unset — ships the CLEANED argv (see
     "Leaving no residue" below),
  4. the restore fixture swaps the pre-test profile (ds4 today) back.

The restore fixture swaps back UNCONDITIONALLY, force=True: a live test
that leaves sparky misconfigured would recreate the outage class this
project exists to remove (same posture as test_spark_ds4.restore_ds4 and
conftest.restore_hipfire).

Leaving no residue takes TWO writes, not one (final branch review,
2026-08-07). Unsetting the override in the Deck's settings store cleans the
DECK only; the NODE keeps the document step 2 shipped, and
swap-helper.sh reads that document on EVERY subsequent swap of the
profile. Without step 3 sparky would serve half its compose context
indefinitely — and invisibly, since step 2's re-swap already re-recorded
intent and cleared settings_drift. Step 3 must also run BEFORE the restore
fixture's swap-back, which is why it lives at the end of the test body
rather than in a fixture of its own.

D12 (adopt) is defined FIRST and therefore runs first (pytest's default
order is file order; this suite has no randomization plugin — see
livetests/requirements.txt): app.routers.spark.spark_reload requires an
adopted ``profile_identities`` entry for the profile it is reloading
(app/routers/spark.py:142-146, 409 "has no adopted identity" otherwise),
and — more importantly — D11's reload ships an argv built ONLY from
DECLARED settings-store layers (app/routers/settings.py's
``_declared_only`` / design decision 3). Without D12 having imported
heretic's full compose command into that same scope first, D11's PUT would
be the ONLY key in it, and reload would ship an argv containing nothing but
``--max-model-len 131072`` — which reload now refuses outright (the
positional guard, 409 naming adopt) rather than shipping a launch the
engine cannot boot. The two cases are one spine, not two independent
drills.
"""

import os
import time

import pytest

from app.argline import POSITIONAL_KEY

pytestmark = pytest.mark.disruptive

VLLM_PROFILE = os.environ.get("DECK_DRILL_SETTINGS_PROFILE", "heretic")
# app.compose_import's derived identity for VLLM_PROFILE's /model mount --
# tests/fixtures/spark-profiles/compose-heretic.yaml, pinned live by
# tests/test_compose_import.py:60 and tests/test_api.py:2306-2307,2317.
IDENTITY = "Qwen3.6-35B-A3B-heretic-NVFP4"
ENGINE_MODEL_KEY = f"sparky/vllm|{IDENTITY}"
SETTING_KEY = "max-model-len"
SETTING_VALUE = "131072"   # heretic's compose says "262144" -- a real, different value

# Same budget test_spark_ds4.SWAP_TIMEOUT / test_spark_comfyui.SWAP_TIMEOUT
# use: vLLM warm boot ~5 min, up to ~15 min if FlashInfer autotunes cold.
SWAP_TIMEOUT = 900

# Helpers are local, as they are in every other disruptive module here
# (test_spark_ds4.py's docstring: "D5 keeps its own identical trio; the
# suite's modules are self-contained").


def _status(deck) -> dict:
    return deck.get("/api/spark/status").json()


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        time.sleep(5)
    pytest.fail(f"{what} -- not met within {timeout_s:.0f}s")


def _wait_serving(deck, profile: str, timeout_s: float = SWAP_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = _status(deck)
        if s["serving"]["endpoint_ok"] and s["serving"]["model"] == profile:
            return s
        time.sleep(10)
    pytest.fail(f"{profile} not serving within {timeout_s}s: {_status(deck)}")


def _swap_with_patience(deck, profile: str, attempts: int = 6, pause_s: float = 5.0):
    """Swap, tolerating a TRANSIENT busy 409.

    ds4 is a _STRICT_BUSY_ENGINES member and counts EVERY http request as
    in-flight, including the deck's own 300s derive pass probing /v1/models
    (proven live 2026-08-07: first D11 run collided with it and 409'd on
    attempt one). The guard is correct — the drill retries briefly rather
    than forcing through it; a PERSISTENT 409 still fails the drill."""
    for remaining in range(attempts - 1, -1, -1):
        resp = deck.post("/api/spark/swap", json={"profile": profile})
        if resp.status_code != 409 or remaining == 0:
            resp.raise_for_status()
            return resp
        time.sleep(pause_s)


@pytest.fixture()
def settings_window(deck, litellm_direct, spark_serving):
    """Skip D11 rather than force through a guard that exists to protect
    default-route callers.

    ``SparkClient.swap``'s litellm-default-route guard is force-proof
    (app/engines/spark.py's ``swap()``: "force does not override this") --
    the same precondition test_spark_ds4.ds4_window checks before its own
    round trip, for the same reason: every swap in this module would 409
    on the very first call if litellm's default route currently targets
    the spark serving endpoint. Skipping is the correct response to a live
    state the drill did not create and must not fight; it is not this
    module's job to move the default route out of the way.
    """
    if spark_serving.host in litellm_direct.default_api_base():
        pytest.skip("litellm's default route targets the spark serving endpoint -- "
                    "SparkClient.swap's guard refuses every swap unconditionally")
    if not any(p["name"] == VLLM_PROFILE for p in _status(deck)["profiles"]):
        pytest.skip(f"profile {VLLM_PROFILE!r} absent from the node -- set "
                    "DECK_DRILL_SETTINGS_PROFILE to a real vllm profile")


@pytest.fixture()
def restore_spark_profile(deck):
    """Record what is serving before the drill touches anything; swap it
    back on teardown, UNCONDITIONALLY and force=True.

    Runs on pass, fail and error alike -- same posture as
    test_spark_ds4.restore_ds4 and conftest.restore_hipfire. Restoring
    through the deck's own swap route (rather than sparky's swap.sh
    directly) is deliberate: that is the action that records intent
    ``loaded`` for the slot (app/routers/spark.py's ``_swap_and_record``),
    so the reconciler is left with a truthful record instead of one naming
    a profile nobody actuated through it.
    """
    before = _status(deck)
    prior = (before.get("swap_status") or {}).get("profile")
    yield prior
    if not prior:
        return
    r = deck.post("/api/spark/swap", json={"profile": prior, "force": True})
    assert r.status_code == 200, f"TEARDOWN FAILED: swap back to {prior!r}: {r.text}"
    _wait_serving(deck, prior)


def test_d12_adopt_imports_without_touching_the_node(deck):
    """Adopt sweeps vLLM profiles, keeps ds4 out, records the profile ->
    identity map, and changes NOTHING on sparky -- no swap or reload call
    appears anywhere in this test."""
    lifecycle = deck.get("/api/state").json()["lifecycle"]
    # settings scope node id and lifecycle node id share one source
    # (app.observe.spark_node_id, derived from SPARK_SLOT_KEY): assert the
    # sparky slot key exists before using its prefix.
    assert any(k.startswith("sparky/") for k in lifecycle)

    report = deck.post("/api/settings/adopt/sparky/vllm").json()

    # {"adopted": [...], "kept": [...], "skipped": [...]} -- app/routers/
    # settings.py adopt(), the route's return statement.
    assert report["adopted"] or report["kept"], report
    assert any(s["profile"] == "ds4" for s in report["skipped"]), report

    scope = deck.get(f"/api/settings/engine_models/{ENGINE_MODEL_KEY}").json()
    # kv-cache-dtype and the positional [serve, /model] are values D11 never
    # touches (its PUT writes only SETTING_KEY) -- unlike max-model-len,
    # asserting on THESE stays a valid re-run check no matter how many times
    # this module has already swapped D11's override on and off on this box
    # (values pinned live: tests/test_compose_import.py:27-41; tests/
    # test_api.py:2319-2322).
    assert scope["args"]["kv-cache-dtype"] == "fp8_e4m3"
    assert scope["args"][POSITIONAL_KEY] == ["serve", "/model"]
    # Notes are written only on the profile's ORIGINAL adopt: a PUT never
    # carries a `note`, so SettingsStore.put's `if note is not None` guard
    # (app/settings_store.py) leaves an existing note untouched -- stable
    # indefinitely, unlike max-model-len above.
    assert "modelopt" in (scope.get("notes", {}).get("args") or "")


def test_d11_save_flags_drift_reload_applies(deck, spark_serving, settings_window,
                                             restore_spark_profile):
    """The whole C2 spine: save -> drift (nothing restarts) -> reload ->
    the engine serves the new value -> drift clears -> unset + reload
    again, so neither the Deck NOR the node keeps the drill's override."""
    _swap_with_patience(deck, VLLM_PROFILE)
    _wait_serving(deck, VLLM_PROFILE)

    deck.put(
        f"/api/settings/engine_models/{ENGINE_MODEL_KEY}",
        json={"namespace": "args", "values": {SETTING_KEY: SETTING_VALUE}},
    ).raise_for_status()

    entry = deck.get("/api/state").json()["lifecycle"]["sparky/slot0"]
    assert entry["settings_drift"] is not None, entry
    # Qualified "namespace:key" form -- app/routers/__init__.py's
    # _settings_drift docstring: 'changed entries are qualified
    # "namespace:key"'. MEMBERSHIP, not exact equality with a one-element
    # list: within a namespace whose stamp postdates baseline, EVERY
    # current key of that namespace is reported, not just the key(s) this
    # PUT touched (app/routers/__init__.py:186-190 -- accepted C1
    # approximation, this store keeps no per-key write history to diff
    # against). heretic's engine_models scope carries ~15 args keys after
    # D12's adopt, so "args:max-model-len" is one entry among many, not
    # the whole list.
    assert f"args:{SETTING_KEY}" in entry["settings_drift"]["changed"]
    # app/configure.py's contract: "Nothing here restarts anything. A save
    # changes intent; the reload that applies launch-class settings is
    # always a human click." The PUT above never called apply_settings.
    assert spark_serving.reachable(), "saving must not restart anything"

    deck.post("/api/spark/reload", json={}).raise_for_status()
    _wait_for(lambda: spark_serving.max_model_len() == int(SETTING_VALUE), SWAP_TIMEOUT,
              "spark never reported the reloaded max-model-len")

    entry = deck.get("/api/state").json()["lifecycle"]["sparky/slot0"]
    # app/routers/spark.py's spark_reload docstring: "The re-swap's intent
    # record is what clears settings_drift -- the drift flag's baseline IS
    # the intent's updated_ts ... so re-recording it is the entire
    # 'clearing' mechanism."
    assert entry["settings_drift"] is None, "reload re-records intent; drift clears"

    # D11 leaves no residue, part 1 of 2 -- the DECK store: unset the
    # override (None unsets at the ladder -- app/ladder.py:14-18, "None at a
    # higher layer unsets a lower one") so a FUTURE reload of this profile
    # no longer ships this drill's 131072 override; the engine's own default
    # takes over again for anyone who reloads it next.
    deck.put(
        f"/api/settings/engine_models/{ENGINE_MODEL_KEY}",
        json={"namespace": "args", "values": {SETTING_KEY: None}},
    ).raise_for_status()

    # Part 2 of 2 -- the NODE document (see this module's docstring,
    # "Leaving no residue"). The PUT above touched nothing on sparky:
    # node-agent still holds the document the reload above shipped, and
    # swap-helper.sh renders it into the compose override on EVERY swap of
    # this profile. One more reload ships the cleaned argv and is the only
    # thing that retires the override; it must happen BEFORE
    # restore_spark_profile's teardown swap, so it lives here, not in a
    # fixture. End state: the node holds heretic's ADOPTED argv minus this
    # one flag (the override replaces compose's command outright, so the
    # engine's own default covers it, not compose's 262144) -- asserted as
    # "not the drill's value" rather than a guessed replacement.
    deck.post("/api/spark/reload",
              json={"profile": VLLM_PROFILE}).raise_for_status()
    _wait_serving(deck, VLLM_PROFILE)
    _wait_for(lambda: spark_serving.max_model_len() != int(SETTING_VALUE),
              SWAP_TIMEOUT,
              "the node's document still carries the drill's max-model-len")
