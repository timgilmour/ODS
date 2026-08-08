import { describe, expect, it } from "vitest";
import type { FactsMap, ResolvedEntry } from "../api";
import {
  buildChips,
  bufferRemove,
  bufferSet,
  discardPendingAdd,
  displayValue,
  emptyBuffer,
  isDirty,
  isListEdit,
  LAYER_FOR_KIND,
  mergedArgsForPreview,
  parseValueText,
  POSITIONAL_KEY,
  scopeKeys,
  settingsIdentityFor,
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

describe("discardPendingAdd", () => {
  // F1, final branch review 2026-08-07. "+ Add option" has to buffer a
  // starting value before the chip (and so the editor) can render, and for
  // the 166-of-274 live vLLM options with no catalog default that value is
  // `""` (catalogFilter's startingValueFor). Abandoning that editor —
  // Escape, an empty blur, opening the all-options list again, switching
  // write scope — must leave nothing behind, or the next Save ships
  // `--flag ''`.
  it("an abandoned brand-new add leaves the buffer clean", () => {
    const added = bufferSet(emptyBuffer, "engines", "cpu-offload-params", "");
    expect(isDirty(added)).toBe(true);

    const cancelled = discardPendingAdd(added, {
      name: "cpu-offload-params",
      kind: "engines",
    });

    // It was the only edit, so the panel is undirtied entirely: Save is
    // disabled again and no PUT is produced for the scope.
    expect(isDirty(cancelled)).toBe(false);
    expect(toPuts(cancelled, scopeKeys("sparky", "vllm", "m"))).toEqual([]);
  });

  it("records no removal for the abandoned key", () => {
    // The server never saw this set, so telling it to REMOVE the key would
    // name one that scope never had — the PUT SettingsModal's remove guard
    // exists to prevent.
    const b = discardPendingAdd(
      bufferSet(emptyBuffer, "engine_models", "offload-params", ""),
      { name: "offload-params", kind: "engine_models" },
    );
    expect(b.sets.engine_models?.["offload-params"]).toBeUndefined();
    expect(b.removes.engine_models ?? []).not.toContain("offload-params");
  });

  it("keeps every other pending edit, including the same name at another kind", () => {
    let b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    b = bufferSet(b, "engine_models", "max-model-len", "");
    b = bufferRemove(b, "models", "seed");

    // PendingAdd carries the kind it was buffered at because the scope
    // control can move `kind` on while the editor is open.
    b = discardPendingAdd(b, { name: "max-model-len", kind: "engine_models" });

    expect(b.sets.engine_models?.["max-model-len"]).toBeUndefined();
    expect(b.sets.engines?.["max-model-len"]).toBe("8192");
    expect(b.removes.models).toContain("seed");
    expect(isDirty(b)).toBe(true);
  });

  it("is a no-op with no add pending — every non-add cancel path", () => {
    const b = bufferSet(emptyBuffer, "engines", "max-model-len", "8192");
    expect(discardPendingAdd(b, null)).toBe(b);
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

  it("a pending set on a currently-derived key adds a NEW declared chip, leaving the applied chip untouched", () => {
    // Critical fix: overriding a harvested engine default (`seed`, derived
    // origin) is the settings ladder's core use case. Before the fix the
    // pending set produced NO chip anywhere — `seen` swallowed the key in
    // the derived branch and the "new chip" loop skipped it as already
    // seen — while toPuts/mergedArgsForPreview shipped the edit on Save
    // with zero visual feedback along the way.
    const b = bufferSet(emptyBuffer, "engines", "seed", "42");
    const { declared, applied } = buildChips(resolved, b, "engines");

    const appliedSeed = applied.find((c) => c.name === "seed")!;
    expect(appliedSeed).toBeDefined();
    expect(appliedSeed.value).toBe("1");
    expect(appliedSeed.origin).toBe("derived");
    expect(appliedSeed.pendingSet).toBe(false);

    const declaredSeed = declared.find((c) => c.name === "seed")!;
    expect(declaredSeed).toBeDefined();
    expect(declaredSeed.value).toBe("42");
    expect(declaredSeed.layer).toBe("engine");
    expect(declaredSeed.origin).toBe("declared");
    expect(declaredSeed.setAtKind).toBe(true);
    expect(declaredSeed.pendingSet).toBe(true);
    expect(declaredSeed.pendingRemove).toBe(false);

    // toPuts and mergedArgsForPreview must agree with what the declared
    // chip shows — they already read the buffer directly, so this also
    // guards against a future regression drifting the three apart.
    const merged = mergedArgsForPreview(resolved, b);
    expect(merged.seed).toBe("42");

    const puts = toPuts(b, scopeKeys("sparky", "vllm", "m"));
    expect(puts.find((p) => p.kind === "engines")?.values.seed).toBe("42");
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

describe("displayValue / parseValueText", () => {
  it("renders a bare flag as empty text — the flag's presence IS the value", () => {
    // app/argline.py renders `value is True` as the flag alone; showing
    // "true" next to it would invent a CLI argument that never gets sent.
    expect(displayValue(true)).toBe("");
  });

  it("joins a list on spaces, matching render_argline's multi-value form", () => {
    expect(displayValue(["a", "b", "c"])).toBe("a b c");
  });

  it("passes a scalar through verbatim", () => {
    expect(displayValue("262144")).toBe("262144");
  });

  it("splits typed text on any run of whitespace", () => {
    expect(parseValueText("served-model-name", "  a   b\tc ")).toEqual(["a", "b", "c"]);
  });

  it("collapses a one-token list to a scalar, as the store would on write", () => {
    expect(parseValueText("served-model-name", "solo")).toBe("solo");
  });

  it("keeps _positional list-shaped even for a single token", () => {
    // app/argline.py's F1 fix: a scalar _positional makes render_argline
    // iterate the string character by character ("s e r v e").
    expect(parseValueText(POSITIONAL_KEY, "serve")).toEqual(["serve"]);
  });

  it("round-trips a multi-token positional", () => {
    expect(parseValueText(POSITIONAL_KEY, "serve /model")).toEqual(["serve", "/model"]);
  });

  it("makes displayValue and parseValueText inverses for a multi-value flag", () => {
    const value = ["a", "b"];
    expect(parseValueText("served-model-name", displayValue(value))).toEqual(value);
  });
});

describe("isListEdit", () => {
  it("keeps a list value a list even when the widget degraded to text", () => {
    // The defect this exists for: `widget` falls back to "text" for an
    // uncatalogued key, and for EVERY key when the pair has no catalog at
    // all. Opening a six-value --served-model-name and blurring would then
    // commit one space-containing scalar, which render_argline quotes into a
    // single argument — a different command line, produced by merely looking.
    expect(isListEdit("text", ["a", "b"])).toBe(true);
  });

  it("honours the catalog's list widget for a scalar value", () => {
    expect(isListEdit("list", "solo")).toBe(true);
  });

  it("leaves a scalar under a scalar widget alone", () => {
    expect(isListEdit("text", "262144")).toBe(false);
    expect(isListEdit("number", "8192")).toBe(false);
  });

  it("does not treat a bare flag as a list", () => {
    expect(isListEdit("toggle", true)).toBe(false);
  });

  it("round-trips a multi-value flag through display and parse under a text widget", () => {
    const value = ["a", "b", "c"];
    const parsed = isListEdit("text", value)
      ? parseValueText("served-model-name", displayValue(value))
      : displayValue(value);
    expect(parsed).toEqual(value);
  });
});

describe("settingsIdentityFor", () => {
  // The real shape the adopt sweep writes (app/routers/settings.py:319) under
  // the engine-scoped facts key.
  const facts: FactsMap = {
    "engine/sparky/vllm": {
      profile_identities: {
        value: {
          heretic: {
            identity: "Qwen3.6-35B-A3B-heretic-NVFP4",
            service: "vllm",
            container_name: "vllm-heretic",
          },
        },
        origin: "derived",
        source: "compose import",
        derived_ts: "2026-08-07T12:00:00Z",
      },
    },
  };

  it("translates a spark PROFILE into the checkpoint identity settings live under", () => {
    // Untranslated, a PUT would land on `sparky/vllm|heretic` — a scope key
    // nothing resolves (the D11 defect, app/routers/__init__.py:160-215).
    expect(settingsIdentityFor(facts, "sparky", "vllm", "heretic")).toBe(
      "Qwen3.6-35B-A3B-heretic-NVFP4",
    );
  });

  it("falls back to the placement name for a profile that was never adopted", () => {
    expect(settingsIdentityFor(facts, "sparky", "vllm", "ornith")).toBe("ornith");
  });

  it("falls back for a node/engine pair with no facts at all — a local tenant", () => {
    expect(settingsIdentityFor(facts, "local", "hipfire", "Qwen3-heretic")).toBe("Qwen3-heretic");
  });

  it("falls back on an empty facts map rather than throwing", () => {
    expect(settingsIdentityFor({}, "sparky", "vllm", "heretic")).toBe("heretic");
  });

  it("refuses a malformed identity map instead of reading through it", () => {
    // FactEntry.value is `unknown` by contract, so every shape has to be
    // survivable: a scalar, a list, and a null identity are all "no
    // translation", never a crash mid-render.
    const entry = { origin: "derived" as const, source: "s", derived_ts: null };
    expect(
      settingsIdentityFor(
        { "engine/sparky/vllm": { profile_identities: { ...entry, value: "heretic" } } },
        "sparky", "vllm", "heretic",
      ),
    ).toBe("heretic");
    expect(
      settingsIdentityFor(
        { "engine/sparky/vllm": { profile_identities: { ...entry, value: ["heretic"] } } },
        "sparky", "vllm", "heretic",
      ),
    ).toBe("heretic");
    expect(
      settingsIdentityFor(
        {
          "engine/sparky/vllm": {
            profile_identities: { ...entry, value: { heretic: { identity: null } } },
          },
        },
        "sparky", "vllm", "heretic",
      ),
    ).toBe("heretic");
  });
});
