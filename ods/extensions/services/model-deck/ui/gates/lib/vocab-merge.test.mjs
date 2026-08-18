import { describe, it, expect } from "vitest";
import { unionVocabulary, shrinkDelta } from "./vocab-merge.mjs";

describe("unionVocabulary", () => {
  it("keeps an existing token a fresh (narrower) observation didn't reproduce", () => {
    const existing = { shape: {}, tokens: { status: ["idle", "unreachable"] } };
    const observed = { shape: {}, tokens: { status: ["idle"] } };
    expect(unionVocabulary(existing, observed).tokens.status).toEqual(["idle", "unreachable"]);
  });

  it("keeps an existing shape path a fresh observation never returned this run", () => {
    const existing = { shape: { "/api/state.lifecycle.sparky/omni": ["intent", "observed", "status"] }, tokens: {} };
    const observed = { shape: {}, tokens: {} };
    expect(unionVocabulary(existing, observed).shape["/api/state.lifecycle.sparky/omni"]).toEqual([
      "intent",
      "observed",
      "status",
    ]);
  });

  it("adds a genuinely new token/path observed live that existing never had", () => {
    const existing = { shape: {}, tokens: { eventKind: ["apply-start"] } };
    const observed = { shape: { "/api/facts": ["origin"] }, tokens: { eventKind: ["apply-end"] } };
    const out = unionVocabulary(existing, observed);
    expect(out.tokens.eventKind).toEqual(["apply-end", "apply-start"]);
    expect(out.shape["/api/facts"]).toEqual(["origin"]);
  });

  it("does not mutate either input", () => {
    const existing = { shape: {}, tokens: { status: ["idle"] } };
    const observed = { shape: {}, tokens: { status: ["serving"] } };
    unionVocabulary(existing, observed);
    expect(existing.tokens.status).toEqual(["idle"]);
    expect(observed.tokens.status).toEqual(["serving"]);
  });

  // R17's actual guarantee, as a property rather than one hand-picked case:
  // no matter what `observed` is (even empty, even degenerate), unioning it
  // against `existing` can never produce something `shrinkDelta` calls a
  // shrink relative to `existing`. This is the test the review explicitly
  // asked for: a pin that `--capture` (which calls exactly this function on
  // its default, non---allow-shrink path) cannot shrink the committed
  // fixture.
  it("--capture cannot shrink: unioning against ANY observation never drops a token or path existing already had", () => {
    const existing = {
      shape: {
        "/api/state": ["lifecycle", "node"],
        "/api/state.node": ["id", "label"],
      },
      tokens: {
        status: ["idle", "parked", "serving", "unreachable"],
        eventKind: ["apply-vetoed", "load-failed", "move_failed"],
      },
    };

    const candidates = [
      { shape: {}, tokens: {} }, // sparky down: a much narrower live snapshot
      { shape: { "/api/state": ["node"] }, tokens: { status: ["idle"] } }, // partially overlapping
      { shape: { "/api/state": [] }, tokens: { status: [] } }, // degenerate but present
      existing, // capturing the exact same thing twice
    ];

    for (const observed of candidates) {
      const unioned = unionVocabulary(existing, observed);
      expect(shrinkDelta(existing, unioned)).toEqual([]);
    }
  });
});

describe("shrinkDelta", () => {
  it("reports nothing when next is a superset of existing", () => {
    const existing = { shape: {}, tokens: { status: ["idle"] } };
    const next = { shape: {}, tokens: { status: ["idle", "serving"] } };
    expect(shrinkDelta(existing, next)).toEqual([]);
  });

  it("reports exactly what a raw overwrite (--allow-shrink) would drop", () => {
    const existing = { shape: {}, tokens: { status: ["idle", "unreachable"] } };
    const next = { shape: {}, tokens: { status: ["idle"] } }; // a healthy-box re-capture
    const delta = shrinkDelta(existing, next);
    expect(delta).toHaveLength(1);
    expect(delta[0]).toContain("unreachable");
  });

  it("does not flag a brand-new key/family next has that existing never declared", () => {
    const existing = { shape: {}, tokens: { status: ["idle"] } };
    const next = { shape: {}, tokens: { status: ["idle"], kind: ["hipfire"] } };
    expect(shrinkDelta(existing, next)).toEqual([]);
  });
});
