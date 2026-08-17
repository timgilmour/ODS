import { describe, it, expect } from "vitest";
import { createResults } from "./check.mjs";

describe("createResults", () => {
  it("records a row per check and counts only failures", () => {
    const r = createResults();
    r.check("a", true);
    r.check("b", false, "got 3 want 4");
    expect(r.rows()).toEqual([
      { name: "a", ok: true, detail: "" },
      { name: "b", ok: false, detail: "got 3 want 4" },
    ]);
    expect(r.failed()).toBe(1);
  });

  it("refuses a duplicate check name", () => {
    // Two checks with one name make a report where a FAIL can hide behind a
    // PASS of the same label. Refuse, never coerce.
    const r = createResults();
    r.check("a", true);
    expect(() => r.check("a", true)).toThrow(/duplicate check name: a/);
  });

  it("treats a non-boolean ok as a defect, not as truthiness", () => {
    // `check("x", await page.$(".sel"))` is the easy mistake: an ElementHandle
    // is truthy and null is falsy, so it would silently work until it didn't.
    const r = createResults();
    expect(() => r.check("x", "yes")).toThrow(/ok must be a boolean/);
  });
});
