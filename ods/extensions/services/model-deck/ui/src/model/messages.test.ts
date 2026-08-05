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
    for (const tab of [labels.deck, labels.setBuilder, labels.storage, labels.events]) {
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
