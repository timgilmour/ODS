import { describe, expect, it } from "vitest";
import { formatPower, formatTemp, formatUtil, tempTone, utilFillClass } from "./gpuStats";

describe("tempTone", () => {
  it("is na for a null reading", () => {
    expect(tempTone(null)).toBe("na");
  });

  it("is good below the amber threshold", () => {
    expect(tempTone(69)).toBe("good");
  });

  it("is warn at the amber boundary (>=70)", () => {
    expect(tempTone(70)).toBe("warn");
  });

  it("stays warn just under the red boundary", () => {
    expect(tempTone(84)).toBe("warn");
  });

  it("is bad at the red boundary (>=85)", () => {
    expect(tempTone(85)).toBe("bad");
  });

  it("stays bad above the red boundary", () => {
    expect(tempTone(91)).toBe("bad");
  });
});

describe("utilFillClass", () => {
  it("reuses the neutral meter fill class for a null reading", () => {
    expect(utilFillClass(null)).toBe("meter-fill meter-neutral");
  });

  it("is neutral below the amber threshold", () => {
    expect(utilFillClass(69)).toBe("meter-fill meter-neutral");
  });

  it("stays neutral exactly AT 70 — the threshold is strictly-greater-than", () => {
    expect(utilFillClass(70)).toBe("meter-fill meter-neutral");
  });

  it("is amber just above the amber threshold", () => {
    expect(utilFillClass(71)).toBe("meter-fill meter-amber");
  });

  it("is amber at 85", () => {
    expect(utilFillClass(85)).toBe("meter-fill meter-amber");
  });

  it("stays amber exactly AT 90 — the red threshold is strictly-greater-than", () => {
    expect(utilFillClass(90)).toBe("meter-fill meter-amber");
  });

  it("is red just above the red threshold (91)", () => {
    expect(utilFillClass(91)).toBe("meter-fill meter-red");
  });
});

describe("formatUtil", () => {
  it("renders an em dash for a null reading", () => {
    expect(formatUtil(null)).toBe("—");
  });

  it("formats a percent", () => {
    expect(formatUtil(3)).toBe("3%");
  });

  it("rounds a fractional percent", () => {
    expect(formatUtil(70.6)).toBe("71%");
  });
});

describe("formatTemp", () => {
  it("renders an em dash for a null reading", () => {
    expect(formatTemp(null)).toBe("—");
  });

  it("formats a temperature in celsius", () => {
    expect(formatTemp(29)).toBe("29°C");
  });

  it("rounds a fractional reading", () => {
    expect(formatTemp(84.6)).toBe("85°C");
  });
});

describe("formatPower", () => {
  it("renders an em dash for a null reading", () => {
    expect(formatPower(null)).toBe("—");
  });

  it("formats a whole watt reading with one decimal", () => {
    expect(formatPower(15)).toBe("15.0W");
  });

  it("keeps a fractional watt reading to one decimal", () => {
    expect(formatPower(214.87)).toBe("214.9W");
  });
});
