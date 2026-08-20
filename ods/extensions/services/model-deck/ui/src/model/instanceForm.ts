/**
 * Instance create-form logic (INST I1) — pure, componentless, same "logic
 * inline in a component is logic no test can reach" rule `engineForm.ts`'s
 * docstring names, and the sibling module to it: an instance is a
 * free-standing declared engine (`POST /api/nodes/{id}/instances`,
 * app/routers/instances.py) rather than the node's one pre-seeded entry, so
 * its form needs its own kind filter (only kinds `instance: true` marks) and
 * its own env buffer (`EngineKindDef.instance_env`, not `connection`) — but
 * the GPU-claim rule is identical, which is exactly why `toggleInstanceGpu`
 * calls `engineForm.ts`'s shared `toggleIndex` rather than re-deriving it
 * (CONTROLLER RULING, INST I1 pre-flight).
 *
 * Every per-kind fact — which kinds are instantiable, which env keys exist
 * and which are required, the GPU cap — comes from `GET /api/engine-kinds`
 * (app/routers/nodes.py's `list_engine_kinds`), NEVER a UI literal (spec
 * §5, same posture `engineForm.ts` takes for `connection`).
 */

import type { EngineKindDef, EngineKindsResponse, InstanceCreateBody } from "../api";
import { kindsFor, maxGpusFor, toggleIndex } from "./engineForm";
import { labels } from "./messages";

export interface InstanceFormState {
  kind: string;
  gpuIndices: number[];
  env: Record<string, string>;
  /** The selected kind's REQUIRED env-field names, baked in at construction
   * (`emptyInstanceForm`/`withInstanceKind`) — same "the requiredness check
   * travels with the form buffer" reasoning `EngineFormState.
   * requiredConnectionFields` documents. */
  requiredEnv: string[];
}

function envSchema(kinds: EngineKindsResponse, kind: string): Record<string, { required: boolean }> {
  return kinds.kinds.find((k) => k.kind === kind)?.instance_env ?? {};
}

/** Which of `kinds.kinds` may be declared as a free-standing INSTANCE on the
 * node currently being edited — `kindsFor`'s own remote/local capability
 * filter (engineForm.ts), further narrowed to `instance: true` (a kind can be
 * capable on this target and still be a fixed single-entry kind with no
 * instance route at all — sglang-omni today). The Create-instance kind
 * `<select>` maps over this function's OUTPUT, never `kinds.kinds` directly,
 * same no-logic-inline-in-components rule `kindsFor`'s own doc names. */
export function instanceKindsFor(kinds: EngineKindsResponse, isRemote: boolean): EngineKindDef[] {
  return kindsFor(kinds, isRemote).filter((k) => k.instance);
}

/** A freshly-opened create form: the seed GPU (usually the board card the
 * operator clicked "+ Create instance" from) as the starting claim, and a
 * blank env buffer built from the kind's own `instance_env` schema — every
 * field "", required ones tracked separately for `instanceFormErrors`. */
export function emptyInstanceForm(
  kinds: EngineKindsResponse, kind: string, seedGpu: number | null,
): InstanceFormState {
  const schema = envSchema(kinds, kind);
  return {
    kind,
    gpuIndices: seedGpu === null ? [] : [seedGpu],
    env: Object.fromEntries(Object.keys(schema).map((n) => [n, ""])),
    requiredEnv: Object.entries(schema).filter(([, s]) => s.required).map(([n]) => n),
  };
}

/** Kind switch: rebuilds `env`/`requiredEnv` for the new kind's own schema
 * (nothing carries across — same reasoning `engineForm.ts`'s `connectionFor`
 * gives for `withKind`, since two kinds' env keys sharing a name would still
 * not be the same fact), and trims the GPU claim down to the new kind's
 * `max_gpus` when it is narrower than what was already picked — a claim
 * `withKind` need not consider since a declared engine's own kind switch has
 * no comparable GPU-cap change today. */
export function withInstanceKind(
  form: InstanceFormState, kinds: EngineKindsResponse, kind: string,
): InstanceFormState {
  const next = emptyInstanceForm(kinds, kind, null);
  const max = maxGpusFor(kinds.kinds, kind);
  const gpuIndices =
    max !== null && form.gpuIndices.length > max ? form.gpuIndices.slice(0, max) : form.gpuIndices;
  return { ...next, gpuIndices };
}

/** Sets ONE env field — the instance form's counterpart to `engineForm.ts`'s
 * `setField`, over `env` instead of `connection`. */
export function setInstanceEnv(form: InstanceFormState, name: string, value: string): InstanceFormState {
  return { ...form, env: { ...form.env, [name]: value } };
}

/** Toggles `index` in/out of the form's GPU claim — delegates to
 * `engineForm.ts`'s shared `toggleIndex` (CONTROLLER RULING) rather than
 * carrying its own copy of the add/remove/replace rule. */
export function toggleInstanceGpu(
  form: InstanceFormState, index: number, max: number | null,
): InstanceFormState {
  return { ...form, gpuIndices: toggleIndex(form.gpuIndices, index, max) };
}

/** The full Create-gate: at least one GPU claimed, and every required env
 * field filled — mirrors `app.engine_kinds.validate_engines`'s per-field
 * check (same requiredness-only posture `engineForm.ts`'s `formErrors`
 * documents; the backend 422 stays authoritative for shape/collision
 * refusals this does not re-implement). */
export function instanceFormErrors(form: InstanceFormState): string[] {
  const errors: string[] = [];
  if (form.gpuIndices.length === 0) errors.push(labels.instanceGpusRequired);
  for (const name of form.requiredEnv) {
    if (!form.env[name]?.trim()) errors.push(labels.instanceEnvRequired(name));
  }
  return errors;
}

/** The POST /api/nodes/{id}/instances body — exactly `InstanceCreateBody`'s
 * shape (app/routers/instances.py's `InstanceCreate`). Empty optional env
 * values are OMITTED, never sent as `""`: `validate_engines`' env loop
 * refuses an empty-string value the same way it refuses an empty-string
 * connection field (app/engine_kinds.py), so shipping one would trade a
 * clear "X is required" for an opaque 422. Only meaningful once
 * `instanceFormErrors(form)` is empty, same posture `engineForm.ts`'s
 * `toPayload` takes for its own gate. */
export function toInstancePayload(form: InstanceFormState): InstanceCreateBody {
  const env = Object.fromEntries(Object.entries(form.env).filter(([, v]) => v.trim() !== ""));
  return { kind: form.kind, gpu_indices: form.gpuIndices, env };
}
