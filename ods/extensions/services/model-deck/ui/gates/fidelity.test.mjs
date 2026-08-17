import { describe, it, expect } from "vitest";
import { compare } from "./fidelity.gate.mjs";

const committed = {
  shape: { "/api/state": ["lifecycle", "node"] },
  tokens: { status: ["serving", "unreachable"] },
};

describe("compare", () => {
  // Task 11 brief, Step 1 — the three tests the plan itself specifies.

  it("fails on a token the fixtures have never seen", () => {
    const live = { shape: committed.shape, tokens: { status: ["serving", "quarantined"] } };
    const bad = compare(live, committed).filter((r) => !r.ok);
    expect(bad).toHaveLength(1);
    expect(bad[0].detail).toContain("quarantined");
  });

  it("passes when live shows FEWER tokens than the fixtures know", () => {
    // One-directional on purpose: sparky being down changes the payload
    // wholesale, and a gate that reddens when a box reboots is a gate
    // people stop reading. The deck makes this same distinction itself
    // (unreachable != down).
    const live = { shape: committed.shape, tokens: { status: ["serving"] } };
    expect(compare(live, committed).every((r) => r.ok)).toBe(true);
  });

  it("fails on an unknown key in an unconditional section", () => {
    const live = {
      shape: { "/api/state": ["lifecycle", "node", "surprise"] },
      tokens: committed.tokens,
    };
    expect(compare(live, committed).some((r) => !r.ok)).toBe(true);
  });

  // Beyond the brief's three — covering the rest of what the docstring
  // claims `compare` does.

  it("is pure: repeated calls with the same inputs return equal, independent results", () => {
    const live = { shape: committed.shape, tokens: { status: ["serving"] } };
    const a = compare(live, committed);
    const b = compare(live, committed);
    expect(a).toEqual(b);
    expect(a).not.toBe(b);
  });

  it("never touches the network or the filesystem — importing and calling it needs neither", () => {
    // No live deck, no fixture on disk, nothing awaited: if this module had
    // top-level I/O or `compare` itself did any, this test would need a
    // stub server or a temp file to even run. It needs neither.
    const live = { shape: {}, tokens: {} };
    const empty = { shape: {}, tokens: {} };
    expect(() => compare(live, empty)).not.toThrow();
  });

  it("passes cleanly when live and committed are identical", () => {
    const live = { shape: committed.shape, tokens: committed.tokens };
    const rows = compare(live, committed);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((r) => r.ok)).toBe(true);
  });

  it("ignores a shape path committed knows but live never returned this run (topology, not drift)", () => {
    const committedWithExtraPath = {
      shape: { ...committed.shape, "/api/state.lifecycle.sparky/omni": ["intent", "observed", "status"] },
      tokens: committed.tokens,
    };
    const live = { shape: committed.shape, tokens: { status: ["serving"] } };
    expect(compare(live, committedWithExtraPath).every((r) => r.ok)).toBe(true);
  });

  it("does NOT catch a field removed from a node-dependent (conditional) section — documented limitation", () => {
    const committedWithLifecycleEntry = {
      shape: {
        ...committed.shape,
        "/api/state.lifecycle.local/hipfire": ["intent", "observed", "reason", "status"],
      },
      tokens: committed.tokens,
    };
    // live's entry lost "reason" — a real regression — but the entry is
    // still present with fewer keys, which one-directional treats the same
    // as "this run just didn't have it". This is the limitation the module
    // doc names explicitly; the test pins that it is real, not accidental.
    const live = {
      shape: {
        ...committed.shape,
        "/api/state.lifecycle.local/hipfire": ["intent", "observed", "status"],
      },
      tokens: { status: ["serving"] },
    };
    expect(compare(live, committedWithLifecycleEntry).every((r) => r.ok)).toBe(true);
  });

  it("DOES catch a field removed from the unconditional node section (claw-back)", () => {
    const committedWithNode = {
      shape: { ...committed.shape, "/api/state.node": ["id", "label"] },
      tokens: committed.tokens,
    };
    const live = {
      shape: { ...committed.shape, "/api/state.node": ["id"] },
      tokens: { status: ["serving"] },
    };
    const bad = compare(live, committedWithNode).filter((r) => !r.ok);
    expect(bad.length).toBeGreaterThan(0);
    expect(bad.some((r) => r.detail.includes("label"))).toBe(true);
  });

  it("fails on an unknown token in a family the fixture has never populated", () => {
    const committedWithEmptyKind = { shape: committed.shape, tokens: { ...committed.tokens, kind: [] } };
    const live = { shape: committed.shape, tokens: { status: ["serving"], kind: ["hipfire"] } };
    const bad = compare(live, committedWithEmptyKind).filter((r) => !r.ok);
    expect(bad).toHaveLength(1);
    expect(bad[0].name).toContain("kind");
    expect(bad[0].detail).toContain("hipfire");
  });
});
