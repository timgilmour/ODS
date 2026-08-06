import { describe, expect, it } from "vitest";
import type { ApplyReport } from "../api";
import { previewRows, reportRows, stepRow } from "./applySteps";

describe("stepRow", () => {
  it("maps every kind sets.py emits to its label and detail", () => {
    expect(stepRow({ step: "unload_lemonade", model: "m.gguf" }, 0))
      .toEqual({ key: "0-unload_lemonade", label: "Unload", detail: "m.gguf" });
    expect(stepRow({ step: "load_lemonade", model: "m.gguf" }, 1).detail).toBe("m.gguf");
    expect(stepRow({ step: "free_comfyui" }, 0).label).toBe("Free ComfyUI VRAM");
    expect(stepRow({ step: "park_hipfire" }, 0).label).toBe("Park hipfire");
    expect(stepRow({ step: "resume_hipfire" }, 0).label).toBe("Resume hipfire");
    expect(stepRow({ step: "activate", model_id: "cat-1" }, 0).detail).toBe("cat-1");
    expect(stepRow({ step: "policy_patch", policies: {} }, 0).label)
      .toBe("Apply policy overrides");
    expect(stepRow({ step: "warn", reason: "comfyui-busy-skipped" }, 0))
      .toMatchObject({ label: "Warning", detail: "comfyui-busy-skipped" });
  });

  it("renders an unknown future kind verbatim rather than crashing", () => {
    expect(stepRow({ step: "defrag_vram" }, 3))
      .toEqual({ key: "3-defrag_vram", label: "defrag_vram", detail: null });
  });

  it("ignores a non-string detail field", () => {
    expect(stepRow({ step: "unload_lemonade", model: 7 }, 0).detail).toBeNull();
  });
});

describe("previewRows / reportRows", () => {
  it("preview rows are all pending", () => {
    const rows = previewRows([{ step: "park_hipfire" }, { step: "activate", model_id: "x" }]);
    expect(rows.map((r) => r.state)).toEqual(["pending", "pending"]);
  });

  it("report rows are the steps that actually ran — completed then failed", () => {
    const report: ApplyReport = {
      completed: [{ step: "unload_lemonade", model: "a.gguf" }],
      failed: { step: "activate", model_id: "x" },
      error: "boom",
      warnings: [],
    };
    const rows = reportRows(report);
    expect(rows.map((r) => [r.label, r.state])).toEqual([
      ["Unload", "done"],
      ["Activate catalog model", "failed"],
    ]);
  });

  it("a clean report has no failed row", () => {
    const report: ApplyReport = {
      completed: [{ step: "park_hipfire" }], failed: null, error: null, warnings: [],
    };
    expect(reportRows(report).map((r) => r.state)).toEqual(["done"]);
  });
});
