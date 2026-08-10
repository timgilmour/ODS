import { describe, expect, it } from "vitest";
import { parseReserveGb } from "./reserveGb";

describe("parseReserveGb", () => {
  it("returns null for an empty field instead of snapping to a number", () => {
    // The bug [max-review c35]: `Math.max(1, Math.round(Number("") || 0))`
    // is 1, so clearing the field rewrote it to "1" under the cursor.
    // null means "not a value yet" — the caller leaves the draft alone.
    expect(parseReserveGb("")).toBeNull();
    expect(parseReserveGb("   ")).toBeNull();
  });

  it("commits a whole number", () => {
    expect(parseReserveGb("30")).toBe(30);
    expect(parseReserveGb(" 30 ")).toBe(30);
    expect(parseReserveGb("1")).toBe(1);
  });

  it("refuses fractional input rather than rounding it", () => {
    // Rounding would commit a number the operator did not type; the server
    // 422s on fractional GB anyway.
    expect(parseReserveGb("2.5")).toBeNull();
  });

  it("refuses zero and negatives", () => {
    expect(parseReserveGb("0")).toBeNull();
    expect(parseReserveGb("-5")).toBeNull();
  });

  it("refuses non-numeric input", () => {
    expect(parseReserveGb("abc")).toBeNull();
    expect(parseReserveGb("30GB")).toBeNull();
  });
});
