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

  it("presents no events with neutral tone", () => {
    const m = messages.noEvents();
    expect(m.tone).toBe("neutral");
    expect(m.title).toBe("no events yet");
  });

  it("presents empty slot with neutral tone", () => {
    const m = messages.emptySlot();
    expect(m.tone).toBe("neutral");
    expect(m.title).toBe("Serving slot");
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

  it("never reports a negative age from clock skew", () => {
    expect(humanizeAge("2026-08-05T12:00:30Z", now)).toBe("0s");
  });
});
