import { describe, expect, it } from "vitest";
import { humanizeAge, labels, messages } from "./messages";

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
