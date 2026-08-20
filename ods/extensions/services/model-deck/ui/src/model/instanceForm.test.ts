import { describe, expect, it } from "vitest";
import type { EngineKindsResponse } from "../api";
import {
  emptyInstanceForm,
  instanceFormErrors,
  instanceKindsFor,
  sameClaim,
  setInstanceEnv,
  toInstancePayload,
  toggleInstanceGpu,
  withInstanceKind,
} from "./instanceForm";

// Fixture rule: kinds from a hand-built catalog, GPUs 2/3/4 (never 0/1) —
// this file's own generalization fixture, matching the fixture rule
// engineForm.test.ts's header comment and nodes.test.ts's header both state.
const KINDS: EngineKindsResponse = {
  kinds: [
    {
      kind: "hipfire", connection: { container: { required: true }, gateway_host: { required: false } },
      remote_capable: false, local_capable: true, demand: false, human_verbs: ["park", "resume"], idle_release: false,
      max_gpus: 1, instance: true,
      instance_env: { HIPFIRE_MODEL: { required: true }, HIPFIRE_IDLE_TIMEOUT: { required: false } },
    },
    {
      kind: "lemonade", connection: {}, remote_capable: false, local_capable: true, demand: true, human_verbs: [],
      idle_release: true, max_gpus: null, instance: true, instance_env: {},
    },
    {
      kind: "sglang-omni", connection: { url: { required: true } }, remote_capable: true, local_capable: false,
      demand: false, human_verbs: [], idle_release: true, max_gpus: 1, instance: false, instance_env: {},
    },
  ],
};

describe("instanceForm", () => {
  it("offers only instantiable kinds the node can run", () => {
    expect(instanceKindsFor(KINDS, false).map((k) => k.kind)).toEqual(["hipfire", "lemonade"]);
    expect(instanceKindsFor(KINDS, true)).toEqual([]); // sglang-omni is remote but not instantiable
  });

  it("seeds the GPU from the board card and the env from the descriptor", () => {
    const f = emptyInstanceForm(KINDS, "hipfire", 4);
    expect(f.gpuIndices).toEqual([4]);
    expect(f.env).toEqual({ HIPFIRE_MODEL: "", HIPFIRE_IDLE_TIMEOUT: "" });
    expect(f.requiredEnv).toEqual(["HIPFIRE_MODEL"]);
    expect(instanceFormErrors(f)).toEqual(["HIPFIRE_MODEL is required"]);
  });

  it("max_gpus 1 REPLACES on toggle; null accumulates and stays sorted", () => {
    const one = toggleInstanceGpu(emptyInstanceForm(KINDS, "hipfire", 2), 4, 1);
    expect(one.gpuIndices).toEqual([4]);
    const many = toggleInstanceGpu(toggleInstanceGpu(emptyInstanceForm(KINDS, "lemonade", 4), 2, null), 3, null);
    expect(many.gpuIndices).toEqual([2, 3, 4]);
    expect(toggleInstanceGpu(many, 3, null).gpuIndices).toEqual([2, 4]);
  });

  it("kind switch rebuilds env and trims the claim to the new max", () => {
    const lem = toggleInstanceGpu(emptyInstanceForm(KINDS, "lemonade", 2), 3, null);
    const hip = withInstanceKind(lem, KINDS, "hipfire");
    expect(hip.gpuIndices).toEqual([2]); // min kept
    expect(Object.keys(hip.env)).toEqual(["HIPFIRE_MODEL", "HIPFIRE_IDLE_TIMEOUT"]);
  });

  it("payload drops empty optional env and is exactly the route's body", () => {
    const f = setInstanceEnv(emptyInstanceForm(KINDS, "hipfire", 4), "HIPFIRE_MODEL", "qwen3.8:27b");
    expect(instanceFormErrors(f)).toEqual([]);
    expect(toInstancePayload(f)).toEqual({ kind: "hipfire", gpu_indices: [4], env: { HIPFIRE_MODEL: "qwen3.8:27b" } });
  });

  it("sameClaim ignores order but not membership", () => {
    expect(sameClaim([2, 3], [3, 2])).toBe(true);
    expect(sameClaim([2, 3], [2, 4])).toBe(false);
    expect(sameClaim([2, 3], [2])).toBe(false);
    expect(sameClaim([], [])).toBe(true);
  });
});
