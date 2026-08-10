import { describe, expect, it } from "vitest";
import { parseWatermark } from "./watermark";

describe("parseWatermark", () => {
  it("REFUSES a non-numeric watermark instead of disabling the guard", () => {
    // The whole point [max-review #15]. The old inline version returned null
    // for anything unparseable, and `watermark_gb: null` is a LEGAL value the
    // backend reads as "no watermark on this drive" — so a typo silently
    // turned auto-archiving OFF for a hot location and reported success.
    expect(parseWatermark("50 GB")).toBe("invalid");
    expect(parseWatermark("abc")).toBe("invalid");
    expect(parseWatermark("50,5")).toBe("invalid");
  });

  it("treats an empty field as an explicit 'no watermark'", () => {
    // Distinct from invalid, and the distinction is the fix: this one is a
    // deliberate operator choice and must still save.
    expect(parseWatermark("")).toBeNull();
    expect(parseWatermark("   ")).toBeNull();
    expect(parseWatermark(undefined)).toBeNull();
    expect(parseWatermark(null)).toBeNull();
  });

  it("parses a number, with or without surrounding whitespace", () => {
    expect(parseWatermark("50")).toBe(50);
    expect(parseWatermark(" 50 ")).toBe(50);
    expect(parseWatermark("0")).toBe(0);
    expect(parseWatermark("12.5")).toBe(12.5);
  });

  it("refuses a negative threshold rather than clamping it", () => {
    // Clamping would silently write a number the operator never typed.
    expect(parseWatermark("-1")).toBe("invalid");
  });

  it("refuses the non-finite spellings Number() otherwise accepts", () => {
    expect(parseWatermark("Infinity")).toBe("invalid");
    expect(parseWatermark("NaN")).toBe("invalid");
  });
});
