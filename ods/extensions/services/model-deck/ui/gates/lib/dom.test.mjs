import { describe, it, expect } from "vitest";
import { disabledExpression, makeAssertUnique } from "./dom.mjs";

describe("disabledExpression", () => {
  it("asks the CSS engine, not the DOM property", () => {
    // THE trap, cost a false Critical on 2026-08-06: an element disabled by
    // an ancestor <fieldset disabled> reports .disabled === false, because
    // the PROPERTY reflects the attribute on that element alone. Only
    // matches(':disabled') sees the inherited state. Set Builder's 409
    // lockdown is exactly an ancestor fieldset, so E1 item 10 depends on
    // this being right.
    const expr = disabledExpression(".engine-form-actions .primary");
    expect(expr).toContain(":disabled");
    expect(expr).toContain("matches");
    expect(expr).not.toMatch(/\.disabled\b/);
  });

  it("returns false rather than throwing when the element is absent", () => {
    // An absent button is a real gate failure, but it must surface as
    // "not disabled" on a named check, not as an unhandled rejection that
    // kills the run and loses every later item.
    expect(disabledExpression(".nope")).toContain("if (!el) return false");
  });
});

describe("makeAssertUnique", () => {
  // Fake `page`: only `.locator(selector).count()` is exercised by
  // assertUnique, so that is the only surface faked here.
  function fakePage(count) {
    return { locator: () => ({ count: async () => count }) };
  }

  it("resolves without throwing when the selector matches exactly one element", async () => {
    const assertUnique = makeAssertUnique("some-gate");
    await expect(assertUnique(fakePage(1), ".sel", "the thing")).resolves.toBeUndefined();
  });

  it("throws, naming the calling gate, when the selector matches nothing", async () => {
    // Parameterised on gate name (M1): two gates hoisting the SAME helper
    // must still be distinguishable in a thrown message, since that message
    // is what a future author reads to find which gate's selector broke.
    const assertUnique = makeAssertUnique("e1-board");
    await expect(assertUnique(fakePage(0), ".sel", "the thing")).rejects.toThrow(
      /e1-board gate: expected exactly 1 the thing \(selector \.sel\), found 0/,
    );
  });

  it("throws, naming the calling gate, when the selector matches more than one element", async () => {
    const assertUnique = makeAssertUnique("e1-engines");
    await expect(assertUnique(fakePage(3), ".sel", "the thing")).rejects.toThrow(
      /e1-engines gate: expected exactly 1 the thing \(selector \.sel\), found 3/,
    );
  });

  it("two instances close over their own gate name independently", async () => {
    const boardAssertUnique = makeAssertUnique("e1-board");
    const enginesAssertUnique = makeAssertUnique("e1-engines");
    await expect(boardAssertUnique(fakePage(0), ".sel", "x")).rejects.toThrow(/^e1-board gate:/);
    await expect(enginesAssertUnique(fakePage(0), ".sel", "x")).rejects.toThrow(/^e1-engines gate:/);
  });
});
