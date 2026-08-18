import { describe, it, expect } from "vitest";
import { createResults, reportingRun } from "./check.mjs";

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

describe("reportingRun", () => {
  it("returns the results object on a clean run", async () => {
    const results = await reportingRun(async (r) => {
      r.check("a", true);
      r.check("b", true);
    });
    expect(results.rows()).toEqual([
      { name: "a", ok: true, detail: "" },
      { name: "b", ok: true, detail: "" },
    ]);
  });

  it("attaches the rows gathered so far to a thrown error, then rethrows it (R15)", async () => {
    // The exact gap R15 exists to close: a gate throwing mid-run must not
    // lose the checks that already printed PASS/FAIL — run.mjs recovers
    // them off `err.partialRows`.
    let thrown;
    try {
      await reportingRun(async (r) => {
        r.check("a", true);
        r.check("b", false, "went wrong");
        throw new Error("mid-run failure");
      });
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(Error);
    expect(thrown.message).toBe("mid-run failure");
    expect(thrown.partialRows).toEqual([
      { name: "a", ok: true, detail: "" },
      { name: "b", ok: false, detail: "went wrong" },
    ]);
  });

  it("attaches an empty partialRows array when the throw happens before any check runs", async () => {
    let thrown;
    try {
      await reportingRun(async () => {
        throw new Error("no checks ran");
      });
    } catch (err) {
      thrown = err;
    }
    expect(thrown.partialRows).toEqual([]);
  });
});
