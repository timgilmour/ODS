import { describe, expect, it } from "vitest";
import type { ResolvedEntry } from "../api";
import {
  buildChips,
  bufferRemove,
  bufferSet,
  emptyBuffer,
  isDirty,
  LAYER_FOR_KIND,
  mergedArgsForPreview,
  scopeKeys,
  toPuts,
} from "./settingsView";

// One key resolved at "engine" (declared, editable), one at "engine_defaults"
// (derived — a harvested engine default, never editable, never shipped).
const resolved: Record<string, ResolvedEntry> = {
  "max-model-len": { value: "262144", origin: "declared", layer: "engine" },
  seed: { value: "1", origin: "derived", layer: "engine_defaults" },
};

// A second fixture where the winner sits at the MOST specific layer, for the
// "less specific pending set doesn't override" case.
const specific: Record<string, ResolvedEntry> = {
  "max-model-len": { value: "131072", origin: "declared", layer: "engine_model" },
};

describe("scopeKeys", () => {
  it("builds the three store keys — mirrors app/routers/settings.py:_resolve", () => {
    expect(scopeKeys("sparky", "vllm", "m")).toEqual({
      engines: "sparky/vllm",
      models: "m",
      engine_models: "sparky/vllm|m",
    });
  });

  it("leaves models and engine_models null with no model in scope", () => {
    const keys = scopeKeys("sparky", "vllm", null);
    expect(keys.models).toBeNull();
    expect(keys.engine_models).toBeNull();
    // The engine scope never depends on a model — always present.
    expect(keys.engines).toBe("sparky/vllm");
  });
});

describe("LAYER_FOR_KIND", () => {
  it("maps each write kind to its declared layer", () => {
    expect(LAYER_FOR_KIND.engines).toBe("engine");
    expect(LAYER_FOR_KIND.models).toBe("model");
    expect(LAYER_FOR_KIND.engine_models).toBe("engine_model");
  });
});

describe("buffer primitives", () => {
  it("emptyBuffer starts clean", () => {
    expect(isDirty(emptyBuffer)).toBe(false);
  });

  it("bufferSet marks the buffer dirty", () => {
    const b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    expect(isDirty(b)).toBe(true);
    expect(b.sets.engines?.["max-model-len"]).toBe("8192");
  });

  it("bufferSet does not mutate its input", () => {
    const b1 = emptyBuffer;
    const b2 = bufferSet(b1, "engines", "max-model-len", "8192");
    expect(b1).toEqual(emptyBuffer);
    expect(b2).not.toBe(b1);
  });

  it("bufferSet clears a pending remove of the same key at the same kind", () => {
    let b = bufferRemove(emptyBuffer, "engines", "max-model-len");
    expect(b.removes.engines).toContain("max-model-len");
    b = bufferSet(b, "engines", "max-model-len", "8192");
    expect(b.removes.engines ?? []).not.toContain("max-model-len");
    expect(b.sets.engines?.["max-model-len"]).toBe("8192");
  });

  it("bufferRemove drops a pending set and marks pendingRemove", () => {
    // Dropping a pending set: the edit never reached the server, so there is
    // nothing to tell it to remove — no removes entry appears.
    let b = bufferSet(emptyBuffer, "engine_models", "temperature", "0.7");
    b = bufferRemove(b, "engine_models", "temperature");
    expect(b.sets.engine_models?.temperature).toBeUndefined();
    expect(b.removes.engine_models ?? []).not.toContain("temperature");
    expect(isDirty(b)).toBe(false);

    // A genuine remove — no pending set to drop — DOES mark pendingRemove.
    const b2 = bufferRemove(emptyBuffer, "engine_models", "max-model-len");
    expect(b2.removes.engine_models).toContain("max-model-len");
    expect(isDirty(b2)).toBe(true);
  });

  it("isDirty is false once every set/remove list is empty again", () => {
    let b = bufferSet(emptyBuffer, "engines", "x", "1");
    b = bufferRemove(b, "engines", "x");
    expect(isDirty(b)).toBe(false);
  });
});

describe("buildChips", () => {
  it("splits declared from applied", () => {
    const { declared, applied } = buildChips(resolved, emptyBuffer, "engines");
    expect(declared.map((c) => c.name)).toEqual(["max-model-len"]);
    expect(applied.map((c) => c.name)).toEqual(["seed"]);
    expect(applied[0].origin).toBe("derived");
    expect(applied[0].setAtKind).toBe(false);
    expect(applied[0].pendingSet).toBe(false);
  });

  it("a pending set at a more specific kind overrides the shown value", () => {
    // resolved winner sits at "engine" (via kind "engines"); a pending set
    // at "engine_models" (layer "engine_model") outranks it per LAYERS
    // (app/ladder.py:48).
    const b = bufferSet(emptyBuffer, "engine_models", "max-model-len", "8192");
    const { declared } = buildChips(resolved, b, "engine_models");
    const chip = declared.find((c) => c.name === "max-model-len")!;
    expect(chip.value).toBe("8192");
    expect(chip.layer).toBe("engine_model");
    expect(chip.setAtKind).toBe(true);
    expect(chip.pendingSet).toBe(true);
    expect(chip.pendingRemove).toBe(false);
  });

  it("a pending set at a LESS specific kind does not override", () => {
    // resolved winner sits at "engine_model" (most specific); a pending set
    // at "engines" (layer "engine") ranks below it and must not change the
    // shown value — but the edit is real, so pendingSet still reads true.
    const b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    const { declared } = buildChips(specific, b, "engines");
    const chip = declared.find((c) => c.name === "max-model-len")!;
    expect(chip.value).toBe("131072");
    expect(chip.layer).toBe("engine_model");
    expect(chip.pendingSet).toBe(true);
    // setAtKind covers "or a pending set on it" even when shadowed.
    expect(chip.setAtKind).toBe(true);
  });

  it("a pending set on a key absent from resolved creates a new declared chip", () => {
    const b = bufferSet(emptyBuffer, "engines", "gpu-memory-utilization", "0.9");
    const { declared } = buildChips(resolved, b, "engines");
    const chip = declared.find((c) => c.name === "gpu-memory-utilization")!;
    expect(chip).toBeDefined();
    expect(chip.value).toBe("0.9");
    expect(chip.layer).toBe("engine");
    expect(chip.origin).toBe("declared");
    expect(chip.setAtKind).toBe(true);
    expect(chip.pendingSet).toBe(true);
  });

  it("keeps a pendingRemove chip showing the current resolved value", () => {
    // The client cannot know what re-emerges from a lower layer after a
    // remove — `resolved` is already the collapsed, single-winner view, with
    // no lower-layer value left to inspect. Display-only: exactness comes
    // from the server refetch after Save.
    const b = bufferRemove(emptyBuffer, "engines", "max-model-len");
    const { declared } = buildChips(resolved, b, "engines");
    const chip = declared.find((c) => c.name === "max-model-len")!;
    expect(chip.pendingRemove).toBe(true);
    expect(chip.value).toBe("262144");
    expect(chip.pendingSet).toBe(false);
  });

  it("declared chips are sorted by name", () => {
    let b = bufferSet(emptyBuffer, "engines", "zzz-flag", "1");
    b = bufferSet(b, "engines", "aaa-flag", "1");
    const { declared } = buildChips(resolved, b, "engines");
    expect(declared.map((c) => c.name)).toEqual(["aaa-flag", "max-model-len", "zzz-flag"]);
  });

  it("carries a bare boolean flag as a first-class pending-set value", () => {
    const b = bufferSet(emptyBuffer, "engines", "enable-chunked-prefill", true);
    const { declared } = buildChips(resolved, b, "engines");
    const chip = declared.find((c) => c.name === "enable-chunked-prefill")!;
    expect(chip.value).toBe(true);
    expect(chip.pendingSet).toBe(true);
  });
});

describe("toPuts", () => {
  it("emits one put per touched kind and refuses a null key", () => {
    let b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    b = bufferSet(b, "engine_models", "seed", "42");
    b = bufferRemove(b, "engines", "gpu-memory-utilization");
    const keys = scopeKeys("sparky", "vllm", "m");
    const puts = toPuts(b, keys);

    expect(puts).toHaveLength(2);
    expect(puts.find((p) => p.kind === "engines")).toEqual({
      kind: "engines",
      key: "sparky/vllm",
      values: { "max-model-len": "8192" },
      remove: ["gpu-memory-utilization"],
    });
    expect(puts.find((p) => p.kind === "engine_models")).toEqual({
      kind: "engine_models",
      key: "sparky/vllm|m",
      values: { seed: "42" },
      remove: [],
    });
  });

  it("refuses a model-scoped kind when the panel has no model", () => {
    const bModel = bufferSet(emptyBuffer, "models", "max-model-len", "8192");
    const noModelKeys = scopeKeys("sparky", "vllm", null);
    expect(() => toPuts(bModel, noModelKeys)).toThrow();
  });

  it("emits nothing for an untouched buffer", () => {
    const keys = scopeKeys("sparky", "vllm", "m");
    expect(toPuts(emptyBuffer, keys)).toEqual([]);
  });
});

describe("mergedArgsForPreview", () => {
  it("excludes derived layers", () => {
    const merged = mergedArgsForPreview(resolved, emptyBuffer);
    expect(merged).toEqual({ "max-model-len": "262144" });
    expect(merged.seed).toBeUndefined();
  });

  it("folds pending sets in by layer rank, highest-rank kind wins", () => {
    let b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    b = bufferSet(b, "engine_models", "max-model-len", "4096");
    const merged = mergedArgsForPreview(resolved, b);
    expect(merged["max-model-len"]).toBe("4096");
  });

  it("deletes a pendingRemove'd key from the preview", () => {
    const b = bufferRemove(emptyBuffer, "engines", "max-model-len");
    const merged = mergedArgsForPreview(resolved, b);
    expect(merged["max-model-len"]).toBeUndefined();
  });

  it("carries a bare boolean flag through into the merged preview", () => {
    const b = bufferSet(emptyBuffer, "engines", "enable-chunked-prefill", true);
    const merged = mergedArgsForPreview(resolved, b);
    expect(merged["enable-chunked-prefill"]).toBe(true);
  });
});
