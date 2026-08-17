import { describe, it, expect } from "vitest";
import { disabledExpression } from "./dom.mjs";

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
