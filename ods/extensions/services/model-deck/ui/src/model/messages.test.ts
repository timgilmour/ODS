import { describe, expect, it } from "vitest";
import { humanizeAge, labels, messages, stateTone } from "./messages";

describe("messages", () => {
  it("gives an unreachable node a tone of danger and an age", () => {
    const m = messages.nodeUnreachable("sparky", "26h");
    expect(m.tone).toBe("danger");
    expect(m.body).toContain("26h");
    expect(m.action?.label).toBe("Retry");
  });

  it("says nothing about age when the age is unknown", () => {
    const m = messages.nodeUnreachable("sparky", null);
    expect(m.body).not.toContain("null");
    expect(m.body).not.toContain("undefined");
  });

  it("reports a down node with the backend's own detail and a retry", () => {
    const m = messages.nodeDown("sparky", "swap-helper: container exited (1)");
    expect(m.tone).toBe("danger");
    expect(m.body).toContain("sparky");
    expect(m.body).toContain("swap-helper: container exited (1)");
    expect(m.action?.label).toBe("Retry");
  });

  it("does not trail into a dangling detail when there is none", () => {
    const m = messages.nodeDown("sparky", null);
    expect(m.body).not.toContain("null");
    expect(m.body).not.toContain("undefined");
    expect(m.body?.trimEnd().endsWith("—")).toBe(false);
  });

  it("treats warming as neutral, not a warning", () => {
    // Decision 5: amber is reserved for things wanting a decision. A first
    // boot that is proceeding normally is not one of those.
    expect(messages.warmingFirstBoot().tone).toBe("neutral");
  });

  it("names the node and says a failed fetch leaves stale data on screen", () => {
    // Deliberately NOT nodeUnreachable: that reports what the backend says
    // about a node. This is the page's OWN fetch failing while the backend
    // still believes the node is fine — the case that would otherwise show
    // minutes-old data under a confident status pill.
    const m = messages.nodeFetchFailed("sparky", "500 Internal Server Error");
    expect(m.tone).toBe("danger");
    expect(m.title).toContain("sparky");
    expect(m.body).toContain("500 Internal Server Error");
    expect(m.body).toContain("out of date");
  });

  it("surfaces a node misconfiguration as a warning carrying the agent's own sentence", () => {
    // lifecycle-node-misconfigured (app/arbiter.py _node_observations); the
    // body is the node-agent's own warning text, passed through verbatim.
    const m = messages.nodeMisconfigured(
      "sparky",
      "vllm profiles configured but NODE_SERVING_PROBE_URL is unset — serving detection is blind",
    );
    expect(m.tone).toBe("warning");
    expect(m.title).toContain("sparky");
    expect(m.body).toContain("NODE_SERVING_PROBE_URL");
  });

  it("passes a guard refusal through verbatim as the body", () => {
    const m = messages.guardRefused("busy — 2 requests in flight");
    expect(m.tone).toBe("danger");
    expect(m.title).toBe("Refused");
    expect(m.body).toBe("busy — 2 requests in flight");
  });

  it("treats force-confirm as warning, not danger, because it asks for a decision", () => {
    // Warning, not danger: this is asking the operator for a decision, which
    // is what warning means here. Nothing has failed or been refused — the
    // red-outlined button above it is what carries the danger.
    const m = messages.forceConfirm();
    expect(m.tone).toBe("warning");
    expect(m.title).toBe("Click again to confirm");
  });

  it("offers the cold pull as a warning carrying the size and its own action", () => {
    // Warning, not danger: nothing failed — the operator is being asked
    // whether to spend the copy. The action label is what arms the retry.
    const m = messages.modelIsCold("18.5");
    expect(m.tone).toBe("warning");
    expect(m.body).toContain("18.5");
    expect(m.action?.label).toBe("Pull + load");
  });

  it("treats an in-flight pull as neutral, with no action", () => {
    const m = messages.pullingFromCold();
    expect(m.tone).toBe("neutral");
    expect(m.action).toBeUndefined();
  });

  it("reports state refresh failure with danger tone", () => {
    const m = messages.stateRefreshFailed("connection timeout");
    expect(m.tone).toBe("danger");
    expect(m.title).toBe("State refresh failed");
    expect(m.body).toBe("connection timeout");
  });

  it("reports an events fetch failure as danger, carrying the detail", () => {
    const m = messages.eventsFetchFailed("503 Service Unavailable");
    expect(m.tone).toBe("danger");
    expect(m.body).toBe("503 Service Unavailable");
  });

  it("presents no events with neutral tone", () => {
    const m = messages.noEvents();
    expect(m.tone).toBe("neutral");
    expect(m.title).toBe("no events yet");
  });

  it("presents last known with neutral tone", () => {
    const m = messages.lastKnown();
    expect(m.tone).toBe("neutral");
    expect(m.title).toBe("last known");
  });
});

describe("engines editor messages (E1 Task 12)", () => {
  it("reports an engines-load failure as danger, with a retry action", () => {
    const m = messages.enginesLoadFailed("500 Internal Server Error");
    expect(m.tone).toBe("danger");
    expect(m.body).toBe("500 Internal Server Error");
    expect(m.action?.label).toBe("Retry");
  });

  it("treats an empty declaration as neutral, not a failure", () => {
    // Task 3/11's "zero, three, or five" starting point (spec §5) — an
    // empty engines[] is a legal, expected state, not an error.
    expect(messages.enginesEmpty().tone).toBe("neutral");
    expect(messages.enginesEmpty().title).toBe("no engines declared");
  });

  it("states that Forget only removes the declaration — a running engine is untouched", () => {
    // The armed-confirm copy this Message backs: DELETE /api/nodes/local/
    // engines/{resource} = forget_engine (app/routers/nodes.py:273-321) is
    // bookkeeping-only and never calls the engine.
    const m = messages.forgetEngineConfirm();
    expect(m.tone).toBe("neutral");
    expect(m.title.toLowerCase()).toContain("declaration only");
    expect(m.title.toLowerCase()).toContain("keeps running");
  });

  it("explains what the consent checkbox grants, per entry", () => {
    const m = messages.engineConsentNote();
    expect(m.tone).toBe("neutral");
    expect(m.body).toContain("stop/start");
    expect(m.body).toContain("409");
  });
});

describe("model detail drawer messages", () => {
  it("can honestly say nothing was written when a declared save fails", () => {
    // Unlike settingsSaveFailed's multi-scope walk: one declared edit is one
    // PUT of one field, written atomically (app/declared.py's docstring —
    // "a rejected put leaves the file untouched").
    const m = messages.declaredSaveFailed("422 unprocessable");
    expect(m.tone).toBe("danger");
    expect(m.body).toContain("422 unprocessable");
    expect(m.body).toContain("nothing was written");
  });

  it("treats a vanished placement as neutral news, not a failure", () => {
    // Polling never stopped, so this IS current information — the model was
    // unloaded/parked/swapped away, which is not an error.
    expect(messages.placementGone().tone).toBe("neutral");
  });

  it("passes a facts or reload failure's detail through as the body", () => {
    expect(messages.factsLoadFailed("500").body).toBe("500");
    expect(messages.reloadFailed("nothing is serving; name a profile").body).toBe(
      "nothing is serving; name a profile",
    );
  });
});

describe("labels", () => {
  it("exports a non-empty dismiss label", () => {
    expect(labels.dismiss).toEqual(expect.any(String));
    expect(labels.dismiss.length).toBeGreaterThan(0);
  });

  // Replaces the old `messages.emptySlot()` assertion. That entry existed
  // only to caption an empty resource and was consumed as a bare `.title`,
  // which threw away the tone that made it a Message in the first place —
  // and its text named the spark's slot, so an empty GPU 0 with hipfire
  // parked read "GPU 0 / Serving slot". The caption is a label now, and what
  // matters about it is that it names no particular kind of resource.
  it("captions an empty resource without naming what kind it is", () => {
    expect(labels.nothingPlaced).toEqual(expect.any(String));
    expect(labels.nothingPlaced.length).toBeGreaterThan(0);
    expect(labels.nothingPlaced.toLowerCase()).not.toContain("slot");
    expect(labels.nothingPlaced.toLowerCase()).not.toContain("gpu");
  });

  it("names every tab, so no tab is left as a literal beside a catalogued one", () => {
    // The Events tab used to be the only one reading from here. A rule
    // applied to one of five siblings is the rule dying quietly.
    for (const tab of [labels.deck, labels.setBuilder, labels.storage, labels.nodes, labels.events]) {
      expect(tab).toEqual(expect.any(String));
      expect(tab.length).toBeGreaterThan(0);
    }
  });

  it("labels a swap profile with its engine only when it has a non-default one", () => {
    expect(labels.swapOption("heretic", null)).toBe("heretic");
    expect(labels.swapOption("ds4", "ds4")).toBe("ds4 (ds4)");
  });

  it("marks a cold model with its size in the load picker", () => {
    expect(labels.coldOption("qwen.gguf", "18.5")).toBe("❄ qwen.gguf (18.5 GB)");
  });

  it("formats an eviction priority", () => {
    expect(labels.priority(3)).toBe("P3");
  });

  it("formats a queue depth, and says so rather than lying when it is unknown", () => {
    expect(labels.queue(0)).toBe("queue 0");
    expect(labels.queue(3)).toBe("queue 3");
    expect(labels.queue(null)).toBe("queue —");
  });

  it("rounds an idle time to whole seconds", () => {
    expect(labels.idle(0)).toBe("idle 0 s");
    expect(labels.idle(12.4)).toBe("idle 12 s");
  });

  it("names a fact's origin by its own vocabulary, quoting the payload's source", () => {
    // Facts have two ORIGINS (app/facts.py:resolve_facts), not the settings
    // ladder's five layers — the drawer borrows the dot's colours but must
    // never borrow layerName's sentence, which would claim a fact came from
    // an engine probe.
    expect(labels.factOrigin("derived", "config.json")).toContain("config.json");
    expect(labels.factOrigin("derived", "config.json")).toContain("derived");
    expect(labels.factOrigin("declared", "declared.json")).toContain("declared");
    expect(labels.factOrigin("declared", "declared.json")).not.toContain("engine");
  });

  it("states drift as expected-vs-actual with the backend's own severity word", () => {
    expect(labels.driftDetail("131072", "262144", "mismatch")).toBe(
      "expected 131072 · actual 262144 · mismatch",
    );
  });

  it("explains the pin and the in-use badge in their tooltips", () => {
    // Both are glyph/two-word badges; the tooltip is the only place their
    // meaning is written down, and a tooltip is read out loud.
    expect(labels.pinnedTitle).toContain("evict");
    expect(labels.inUseTitle).toContain("force");
  });

  it("translates a warn-step reason code into a resource-tagged sentence", () => {
    // app/sets.py:695-710 — the free-verb branch, so ONLY a comfyui-kind
    // resource can ever pair with this reason (comfyui is the only kind
    // with "free" in human_verbs() today) — "img" here, never a
    // hipfire-kind resource like "agent" (an impossible pairing: hipfire
    // has no free verb at all).
    expect(labels.stepWarnReason("busy-skipped", "img")).toBe(
      "img skipped — queue not confirmed empty",
    );
    // app/sets.py:758 — resource-tagged.
    expect(labels.stepWarnReason("no-model-to-load", "gguf-a")).toBe(
      "gguf-a has no model to load",
    );
    // app/sets.py:653 — box-wide, no resource.
    expect(labels.stepWarnReason("durable-revert-unavailable")).toBe(
      "durable revert unavailable — no catalog id to re-activate the previous model",
    );
  });

  it("degrades an unrecognized warn reason to the raw code rather than inventing a sentence", () => {
    expect(labels.stepWarnReason("some-future-reason")).toBe("some-future-reason");
  });

  it("falls back to a plain verb when a step label is called with no resource", () => {
    expect(labels.stepUnload(null)).toBe("Unload");
    expect(labels.stepLoad("gguf-a")).toBe("Load — gguf-a");
  });

  it("derives a connection field's label from its own key, never a per-field literal", () => {
    // spec §5: field IDENTITY is never a UI literal — this only makes an
    // underscore-joined payload key readable.
    expect(labels.engineFieldLabel("metrics_url")).toBe("metrics url");
    expect(labels.engineFieldLabel("url")).toBe("url");
  });

  it("names a missing engine-form field, mirroring engine_kinds.py's requiredness", () => {
    expect(labels.engineConnectionFieldRequired("container")).toBe(
      "container is required for this kind",
    );
  });
});

describe("humanizeAge", () => {
  const now = Date.parse("2026-08-05T12:00:00Z");

  it("returns null with no timestamp", () => {
    expect(humanizeAge(null, now)).toBeNull();
  });

  it("returns null for an unparseable timestamp", () => {
    expect(humanizeAge("not a date", now)).toBeNull();
  });

  it("counts hours", () => {
    expect(humanizeAge("2026-08-04T10:00:00Z", now)).toBe("26h");
  });

  it("counts days", () => {
    expect(humanizeAge("2026-08-01T12:00:00Z", now)).toBe("4d");
  });

  it("stays in hours right up to the 48h boundary", () => {
    expect(humanizeAge("2026-08-03T13:00:00Z", now)).toBe("47h");
  });

  it("switches to days exactly at 48h", () => {
    expect(humanizeAge("2026-08-03T12:00:00Z", now)).toBe("2d");
  });

  it("never reports a negative age from clock skew", () => {
    expect(humanizeAge("2026-08-05T12:00:30Z", now)).toBe("0s");
  });
});

describe("invalidWatermark", () => {
  it("refuses in danger tone and NAMES the offending locations", () => {
    // The modal edits several drives at once, so "a watermark is invalid"
    // would leave the operator hunting. danger, because the save was
    // REFUSED and nothing was written [max-review #15].
    const m = messages.invalidWatermark(["hot", "scratch"]);
    expect(m.tone).toBe("danger");
    expect(m.body).toContain("hot");
    expect(m.body).toContain("scratch");
    // Says what to do instead — including that empty is a legal answer,
    // which is the distinction the whole fix rests on.
    expect(m.body).toContain("empty");
  });
});

describe("declared remote engine presentation (Task 10b)", () => {
  it("labels a verb with the same word the local controls use, and passes an unknown one through", () => {
    // The vocabulary is the KIND's (`human_verbs`), so this formats whatever
    // it is handed rather than gating on a closed list — a kind whose verb
    // this UI has no word for still gets a button that says what it does.
    expect(labels.engineVerb("load")).toBe(labels.load);
    expect(labels.engineVerb("unload")).toBe(labels.unload);
    expect(labels.engineVerb("polish")).toBe("polish");
  });

  it("tones the engine's own state word, sharing one map with the local controls", () => {
    expect(stateTone("busy")).toBe("busy");
    expect(stateTone("idle")).toBe("off");
    expect(stateTone("loaded")).toBe("good");
    expect(stateTone("running")).toBe("good");
    // "we failed to look" is a failure to report, not a state to be calm
    // about (app/engine_kinds.py's unknown()).
    expect(stateTone("unknown")).toBe("bad");
    // "down" is grey, not red: the ENGINE's word cannot tell a deliberate
    // unload from a death — the chip's lifecycle pill (parked vs down) is
    // what carries that, and two red things saying different reasons is how
    // a board stops being trusted.
    expect(stateTone("down")).toBe("off");
    expect(stateTone("something-new")).toBe("off");
  });

  it("says the engine catalog failed, with the backend's own detail and a retry", () => {
    const m = messages.engineKindsFailed("502 Bad Gateway");
    expect(m.tone).toBe("danger");
    expect(m.body).toContain("502 Bad Gateway");
    expect(m.action?.label).toBe("Retry");
  });
});

describe("idle-TTL affordances", () => {
  it("renders 0 as off, not as a number", () => {
    // Every kind's idle_action gates on `policy["idle_ttl"] > 0`
    // (app/engine_kinds.py — lemonade/comfyui/sglang-omni idle rules), so 0
    // is a MODE, not a duration.
    expect(messages.ttlValue(0)).toBe("off");
  });

  it("renders seconds with a human duration beside them", () => {
    expect(messages.ttlValue(900)).toBe("900 s — 15 min");
    expect(messages.ttlValue(300)).toBe("300 s — 5 min");
    expect(messages.ttlValue(45)).toBe("45 s");
  });

  it("says nothing happens when the TTL is off", () => {
    expect(messages.ttlConsequence(0, true, true)).toBe("never released automatically");
    expect(messages.ttlConsequence(0, false, true)).toBe("never released automatically");
  });

  it("distinguishes an invisible release from a one-way one", () => {
    // THE point of the whole feature: same number, opposite meaning.
    expect(messages.ttlConsequence(900, true, true)).toBe(
      "released after 15 min idle — the next request reloads it automatically");
    expect(messages.ttlConsequence(900, false, true)).toBe(
      "released after 15 min idle — reload is MANUAL, nothing brings it back");
  });

  it("declines to claim a consequence it cannot know", () => {
    // The kinds catalog failed to load: say so rather than guessing a
    // reversibility the operator would rely on.
    expect(messages.ttlConsequence(900, null, null)).toBe(
      "released after 15 min idle — reload behaviour unknown (kind catalog unavailable)");
  });

  it("says NOTHING happens when the kind has no idle rule at all — never the false MANUAL sentence", () => {
    // hipfire: arbiter_verbs() is empty, idle_action is unconditionally
    // None (app/engine_kinds.py's _HipfireAdapter). A nonzero TTL on it is
    // a no-op, not a one-way release — the old text ("reload is MANUAL,
    // nothing brings it back") falsely implied a rule that fires and simply
    // never reloads.
    expect(messages.ttlConsequence(900, false, false)).toBe(
      "never released automatically (this kind has no idle rule)");
  });

  it("the no-idle-rule sentence wins regardless of the TTL value, including 0", () => {
    expect(messages.ttlConsequence(0, false, false)).toBe(
      "never released automatically (this kind has no idle rule)");
  });
});

describe("humanDuration (via ttlValue) at half-hour boundaries", () => {
  it("does not misstate 90 minutes as a flat 2 hours", () => {
    // Regression: Math.round(90 / 60) rounded straight to 2 h, a 30-minute
    // misstatement. 5400 s = 90 min = 1.5 h.
    expect(messages.ttlValue(5400)).toBe("5400 s — 90 min");
  });

  it("still renders a whole number of hours cleanly at 2 h and above", () => {
    expect(messages.ttlValue(7200)).toBe("7200 s — 2 h");
  });
});
