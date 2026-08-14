import type { ApplyReport, Step } from "../api";
import { labels } from "./messages";

/** One rendered row of the Activate step list. */
export interface StepRow {
  key: string;
  label: string;
  detail: string | null;
}

export type StepItemState = "pending" | "done" | "failed";
export type StepItem = StepRow & { state: StepItemState };

function str(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

/** Maps one plan step to display copy. The vocabulary is app/sets.py
 * plan_apply()'s, verbatim: unload, load, free, park, resume, activate,
 * policy_patch, restore_settings, warn — verb-generic and resource-tagged
 * since E1 Task 8 (the old kind-suffixed unload_lemonade/free_comfyui/
 * park_hipfire/resume_hipfire names are gone from the wire). (Deliberately
 * named rather than line-cited — these citations have drifted three times;
 * grep plan_apply for the emit sites.)
 * Unknown kinds render verbatim — a future backend step must degrade to
 * ugly-but-true, never to a crash or a silent drop. */
export function stepRow(step: Step, index: number): StepRow {
  const key = `${index}-${step.step}`;
  // Every per-resource verb step carries "resource" (app/sets.py); the
  // box-wide ones (activate/policy_patch/restore_settings) and a
  // resource-less warn (durable-revert-unavailable) do not — `str` folds
  // that absence to null rather than throwing on a missing field.
  const resource = str(step.resource);
  switch (step.step) {
    case "unload":
      return { key, label: labels.stepUnload(resource), detail: str(step.model) };
    case "load":
      return { key, label: labels.stepLoad(resource), detail: str(step.model) };
    case "free":
      return { key, label: labels.stepFree(resource), detail: null };
    case "park":
      return { key, label: labels.stepPark(resource), detail: null };
    case "resume":
      return { key, label: labels.stepResume(resource), detail: null };
    case "activate":
      return { key, label: labels.stepActivate, detail: str(step.model_id) };
    case "policy_patch":
      return { key, label: labels.stepPolicyPatch, detail: null };
    case "restore_settings":
      return { key, label: labels.stepRestoreSettings, detail: null };
    case "warn": {
      const reason = str(step.reason);
      return {
        key,
        label: labels.stepWarn,
        detail: reason !== null ? labels.stepWarnReason(reason, resource) : null,
      };
    }
    default:
      return { key, label: step.step, detail: null };
  }
}

/** The confirm phase renders the PLAN: all pending. */
export function previewRows(steps: Step[]): StepItem[] {
  return steps.map((s, i) => ({ ...stepRow(s, i), state: "pending" }));
}

/** The result phase renders what actually RAN, from the report itself —
 * never by index-mapping the report back onto the preview. apply re-plans
 * server-side (app/sets.py: apply route calls plan_apply again), so the
 * executed list can legitimately differ from the previewed one; rendering
 * the report is honest under drift, mapping is not. */
export function reportRows(report: ApplyReport): StepItem[] {
  const rows: StepItem[] = report.completed.map((s, i) => ({
    ...stepRow(s, i),
    state: "done",
  }));
  if (report.failed !== null) {
    rows.push({ ...stepRow(report.failed, rows.length), state: "failed" });
  }
  return rows;
}
