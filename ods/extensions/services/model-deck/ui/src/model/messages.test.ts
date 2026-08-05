import { describe, expect, it } from "vitest";
import { messages } from "./messages";

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
