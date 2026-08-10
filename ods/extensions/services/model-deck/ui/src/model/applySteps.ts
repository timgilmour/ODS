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
 * plan_apply()'s, verbatim: unload_lemonade, free_comfyui, warn,
 * park_hipfire, activate, resume_hipfire, load_lemonade, policy_patch.
 * (Deliberately named rather than line-cited — these citations have drifted
 * three times; grep plan_apply for the emit sites.)
 * Unknown kinds render verbatim — a future backend step must degrade to
 * ugly-but-true, never to a crash or a silent drop. */
export function stepRow(step: Step, index: number): StepRow {
  const key = `${index}-${step.step}`;
  switch (step.step) {
    case "unload_lemonade":
      return { key, label: labels.stepUnload, detail: str(step.model) };
    case "load_lemonade":
      return { key, label: labels.stepLoad, detail: str(step.model) };
    case "free_comfyui":
      return { key, label: labels.stepFreeComfyui, detail: null };
    case "park_hipfire":
      return { key, label: labels.stepParkHipfire, detail: null };
    case "resume_hipfire":
      return { key, label: labels.stepResumeHipfire, detail: null };
    case "activate":
      return { key, label: labels.stepActivate, detail: str(step.model_id) };
    case "policy_patch":
      return { key, label: labels.stepPolicyPatch, detail: null };
    case "warn":
      return { key, label: labels.stepWarn, detail: str(step.reason) };
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
