import { describe, expect, it } from "vitest";
import type { ApplyReport } from "../api";
import { previewRows, reportRows, stepRow } from "./applySteps";

describe("stepRow", () => {
  it("maps every verb-generic, resource-tagged kind sets.py emits to its label and detail", () => {
    // E1 Task 8: verb-generic + resource-tagged (unload/load/free/park/
    // resume, each carrying its own "resource" field) — the old
    // unload_lemonade/free_comfyui/park_hipfire/resume_hipfire names are
    // gone from the wire.
    expect(stepRow({ step: "unload", model: "m.gguf", resource: "gguf-a" }, 0))
      .toEqual({ key: "0-unload", label: "Unload — gguf-a", detail: "m.gguf" });
    expect(stepRow({ step: "load", model: "m.gguf", resource: "gguf-a" }, 1).detail).toBe("m.gguf");
    expect(stepRow({ step: "free", resource: "img" }, 0).label).toBe("Free — img");
    expect(stepRow({ step: "park", resource: "agent" }, 0).label).toBe("Park — agent");
    expect(stepRow({ step: "resume", resource: "agent" }, 0).label).toBe("Resume — agent");
    expect(stepRow({ step: "activate", model_id: "cat-1" }, 0).detail).toBe("cat-1");
    expect(stepRow({ step: "policy_patch", policies: {} }, 0).label)
      .toBe("Apply policy overrides");
    // New in E1 Task 9 — must not fall through to the verbatim default
    // branch below.
    expect(stepRow({ step: "restore_settings", settings: {} }, 0).label).toBe("Restore settings");
    // "comfyui-busy-skipped" (the pre-Task-8 kind-baked reason) is DEAD —
    // T8 review I3 renamed it to the resource-tagged "busy-skipped". Only a
    // comfyui-kind resource can ever pair with it (app/sets.py:695-710's
    // free-verb branch — hipfire has no free verb, so "agent" would be an
    // impossible pairing here).
    expect(stepRow({ step: "warn", reason: "busy-skipped", resource: "img" }, 0))
      .toMatchObject({ label: "Warning", detail: "img skipped — queue not confirmed empty" });
  });

  it("renders a step's label without a resource tag when the step carries none", () => {
    // activate/policy_patch/restore_settings are box-wide (no "resource"
    // field, app/sets.py); a resource-tagged verb step without one (a
    // malformed/future payload) must still degrade to the plain label
    // rather than crash.
    expect(stepRow({ step: "unload", model: "m.gguf" }, 0).label).toBe("Unload");
    expect(stepRow({ step: "warn", reason: "durable-revert-unavailable" }, 0).detail).toBe(
      "durable revert unavailable — no catalog id to re-activate the previous model",
    );
  });

  it("translates a resource-less warn reason too", () => {
    expect(stepRow({ step: "warn", reason: "no-model-to-load" }, 0).detail).toBe(
      "no model to load",
    );
  });

  it("renders a model-mismatch warn step with declared and resident models extracted from the step (ruling #2-C, app/sets.py:762-771)", () => {
    const step = { step: "warn", reason: "model-mismatch", resource: "gguf-a", declared: "new.gguf", resident: "old.gguf" };
    expect(stepRow(step, 0))
      .toMatchObject({ label: "Warning", detail: "gguf-a resident old.gguf, declared new.gguf — apply will not swap" });
  });

  it("renders an unrecognized warn reason verbatim rather than crashing", () => {
    expect(stepRow({ step: "warn", reason: "some-future-reason" }, 0).detail).toBe(
      "some-future-reason",
    );
  });

  it("renders an unknown future kind verbatim rather than crashing", () => {
    expect(stepRow({ step: "defrag_vram" }, 3))
      .toEqual({ key: "3-defrag_vram", label: "defrag_vram", detail: null });
  });

  it("ignores a non-string detail field", () => {
    expect(stepRow({ step: "unload", model: 7, resource: "gguf-a" }, 0).detail).toBeNull();
  });
});

describe("previewRows / reportRows", () => {
  it("preview rows are all pending", () => {
    const rows = previewRows([
      { step: "park", resource: "agent" },
      { step: "activate", model_id: "x" },
    ]);
    expect(rows.map((r) => r.state)).toEqual(["pending", "pending"]);
  });

  it("report rows are the steps that actually ran — completed then failed", () => {
    const report: ApplyReport = {
      completed: [{ step: "unload", model: "a.gguf", resource: "gguf-a" }],
      failed: { step: "activate", model_id: "x" },
      error: "boom",
      warnings: [],
    };
    const rows = reportRows(report);
    expect(rows.map((r) => [r.label, r.state])).toEqual([
      ["Unload — gguf-a", "done"],
      ["Activate catalog model", "failed"],
    ]);
  });

  it("a clean report has no failed row", () => {
    const report: ApplyReport = {
      completed: [{ step: "park", resource: "agent" }], failed: null, error: null, warnings: [],
    };
    expect(reportRows(report).map((r) => r.state)).toEqual(["done"]);
  });
});
