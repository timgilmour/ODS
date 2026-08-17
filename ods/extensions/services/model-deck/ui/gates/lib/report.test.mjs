import { describe, it, expect } from "vitest";
import { renderMarkdown, renderJson } from "./report.mjs";

const META = { tier: "fixture", startedIso: "2026-08-17T12:00:00Z", target: "stub" };

describe("report", () => {
  it("puts failures first so a long green run cannot bury one red line", () => {
    const rows = [
      { name: "a", ok: true, detail: "" },
      { name: "b", ok: false, detail: "boom" },
    ];
    const md = renderMarkdown(rows, META);
    expect(md.indexOf("FAIL b")).toBeLessThan(md.indexOf("PASS a"));
  });

  it("states the totals in the heading", () => {
    const rows = [
      { name: "a", ok: true, detail: "" },
      { name: "b", ok: false, detail: "boom" },
    ];
    expect(renderMarkdown(rows, META)).toContain("1 passed / 1 failed");
  });

  it("emits machine-readable json carrying the same counts", () => {
    const rows = [{ name: "a", ok: false, detail: "boom" }];
    const parsed = JSON.parse(renderJson(rows, META));
    expect(parsed.tier).toBe("fixture");
    expect(parsed.passed).toBe(0);
    expect(parsed.failed).toBe(1);
    expect(parsed.rows[0]).toEqual({ name: "a", ok: false, detail: "boom" });
  });
});
