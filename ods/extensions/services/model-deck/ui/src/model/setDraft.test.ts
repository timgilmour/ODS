import { describe, expect, it } from "vitest";
import type { ConfigSet, ResourceTenant } from "../api";
import {
  buildDraft,
  derivePlacedModel,
  draftChoiceFor,
  draftedFootprintBytes,
  draftEquals,
  fieldsFromSet,
  KIND_DRAFT_SPEC,
  pickLoadResource,
  setResourceChoice,
  undeclaredResources,
} from "./setDraft";

// Resources named away from live topology (gguf-a/img/agent), per the
// design's fixture rule — a set keyed by lemonade/comfyui/hipfire could not
// tell "resource name" from "kind name" apart, since they used to coincide.
const base: ConfigSet = {
  name: "Chat mode",
  notes: "notes",
  durable: { default_route_model: "extra.m.gguf", activate_model_id: null },
  ephemeral: { resources: { "gguf-a": { desired: "loaded" } } },
  policy_overrides: null,
};

describe("fieldsFromSet → buildDraft round-trip", () => {
  it("CRITICAL 2: a set that never mentioned a resource keeps it absent from resources", () => {
    const round = buildDraft(fieldsFromSet(base, false));
    expect(round.ephemeral?.resources.agent).toBeUndefined();
    expect("agent" in (round.ephemeral?.resources ?? {})).toBe(false);
  });

  it("carries policy_overrides verbatim through load → save", () => {
    const withOverrides: ConfigSet = {
      ...base,
      policy_overrides: { "gguf-a": { priority: 100, pinned: true, idle_ttl: 0 } },
    };
    const round = buildDraft(fieldsFromSet(withOverrides, false));
    expect(round.policy_overrides).toEqual(withOverrides.policy_overrides);
  });

  it("clearName empties the name (Duplicate) without touching anything else", () => {
    const f = fieldsFromSet(base, true);
    expect(f.name).toBe("");
    expect(f.notes).toBe("notes");
  });

  it("buildDraft trims the name", () => {
    const f = fieldsFromSet(base, false);
    expect(buildDraft({ ...f, name: "  padded  " }).name).toBe("padded");
  });

  it("round-trips an empty declaration (nothing declared, nothing drafted)", () => {
    const empty: ConfigSet = { ...base, ephemeral: { resources: {} } };
    const round = buildDraft(fieldsFromSet(empty, false));
    expect(round.ephemeral?.resources).toEqual({});
  });
});

describe("derivePlacedModel", () => {
  it("derives the file from an extra.-prefixed route with a loaded goal somewhere in the set", () => {
    expect(derivePlacedModel(base)).toBe("m.gguf");
  });

  it("returns null for a non-extra route (cannot be traced to a library file)", () => {
    const other: ConfigSet = {
      ...base,
      durable: { default_route_model: "gpt-oss-120b", activate_model_id: null },
    };
    expect(derivePlacedModel(other)).toBeNull();
  });

  it("returns null when nothing in the set is drafted loaded", () => {
    const unloaded: ConfigSet = {
      ...base,
      ephemeral: { resources: { "gguf-a": { desired: "unloaded" } } },
    };
    expect(derivePlacedModel(unloaded)).toBeNull();
  });

  it("returns null when durable is absent", () => {
    expect(derivePlacedModel({ ...base, durable: null })).toBeNull();
  });

  it("returns null when ephemeral is absent entirely", () => {
    expect(derivePlacedModel({ ...base, ephemeral: null })).toBeNull();
  });
});

describe("draftEquals", () => {
  it("equal for identical content", () => {
    expect(draftEquals(buildDraft(fieldsFromSet(base, false)), {
      ...base,
    })).toBe(true);
  });

  it("unequal when any field differs", () => {
    expect(draftEquals(base, { ...base, notes: "changed" })).toBe(false);
  });

  it("equal regardless of resources' key insertion order — a dynamic Record has no fixed order to rely on", () => {
    const orderA: ConfigSet = {
      ...base,
      ephemeral: {
        resources: { "gguf-a": { desired: "loaded" }, img: { desired: "freed" } },
      },
    };
    const orderB: ConfigSet = {
      ...base,
      ephemeral: {
        resources: { img: { desired: "freed" }, "gguf-a": { desired: "loaded" } },
      },
    };
    expect(draftEquals(orderA, orderB)).toBe(true);
  });

  it("still unequal when the resources actually differ, despite the order-tolerant comparison", () => {
    const a: ConfigSet = { ...base, ephemeral: { resources: { "gguf-a": { desired: "loaded" } } } };
    const b: ConfigSet = { ...base, ephemeral: { resources: { "gguf-a": { desired: "unloaded" } } } };
    expect(draftEquals(a, b)).toBe(false);
  });
});

describe("KIND_DRAFT_SPEC — mirrors app/sets.py's _DESIRED_VERBS crossed with human_verbs()", () => {
  it("gives a lemonade-kind (load-verb) resource loaded/unloaded and model support", () => {
    expect(KIND_DRAFT_SPEC.lemonade.options).toEqual(["loaded", "unloaded"]);
    expect(KIND_DRAFT_SPEC.lemonade.supportsModel).toBe(true);
  });

  it("gives a hipfire-kind resource loaded (resume)/parked and no model support", () => {
    expect(KIND_DRAFT_SPEC.hipfire.options).toEqual(["loaded", "parked"]);
    expect(KIND_DRAFT_SPEC.hipfire.supportsModel).toBe(false);
  });

  it("gives a comfyui-kind resource only freed, no model support", () => {
    expect(KIND_DRAFT_SPEC.comfyui.options).toEqual(["freed"]);
    expect(KIND_DRAFT_SPEC.comfyui.supportsModel).toBe(false);
  });

  it("labels the SAME 'loaded' token differently per kind — never one global table", () => {
    // Fix-loop finding: a single desired->label table read "Running" for
    // every kind offering "loaded", which is wrong for a lemonade-kind
    // resource (its own "loaded" is a Load, not a resume).
    expect(KIND_DRAFT_SPEC.lemonade.optionLabels.loaded).toBe("Load");
    expect(KIND_DRAFT_SPEC.hipfire.optionLabels.loaded).toBe("Running");
  });

  it("labels every kind's own options, nothing more", () => {
    expect(KIND_DRAFT_SPEC.lemonade.optionLabels).toEqual({ loaded: "Load", unloaded: "Unload" });
    expect(KIND_DRAFT_SPEC.hipfire.optionLabels).toEqual({ loaded: "Running", parked: "Parked" });
    expect(KIND_DRAFT_SPEC.comfyui.optionLabels).toEqual({ freed: "Free" });
  });
});

describe("pickLoadResource", () => {
  const gguf_a: ResourceTenant = { engine: "lemonade", gpu_index: 2, state: "unloaded" };
  const gguf_b: ResourceTenant = { engine: "lemonade", gpu_index: 2, state: "unloaded" };
  const img: ResourceTenant = { engine: "comfyui", gpu_index: 3, state: "idle" };
  const agent: ResourceTenant = { engine: "hipfire", gpu_index: 3, state: "parked" };

  it("returns null when nothing declared supports a model", () => {
    expect(pickLoadResource({ img, agent })).toBeNull();
  });

  it("picks the sole load-verb-kind resource", () => {
    expect(pickLoadResource({ "gguf-a": gguf_a, img, agent })).toBe("gguf-a");
  });

  it("with two lemonade-kind resources on the SAME gpu_index, picks deterministically by name — never nondeterministic object-key order", () => {
    // Both directions of declaration order must agree: this is the bug the
    // fix-loop caught (a plain unsorted Object.entries().find picked
    // whichever key JS happened to enumerate first).
    expect(pickLoadResource({ "gguf-a": gguf_a, "gguf-b": gguf_b })).toBe("gguf-a");
    expect(pickLoadResource({ "gguf-b": gguf_b, "gguf-a": gguf_a })).toBe("gguf-a");
  });

  it("with two lemonade-kind resources on DIFFERENT gpu_index, picks the lower gpu_index first — same order the board's cards use", () => {
    const gguf_b_gpu1: ResourceTenant = { ...gguf_b, gpu_index: 1 };
    expect(pickLoadResource({ "gguf-a": gguf_a, "gguf-b": gguf_b_gpu1 })).toBe("gguf-b");
  });
});

describe("draftedFootprintBytes", () => {
  const lemonade = { engine: "lemonade", gpu_index: 3, state: "unloaded", footprint: 5_000_000_000 };
  const hipfire = { engine: "hipfire", gpu_index: 2, state: "running", footprint: 21_000_000_000 };
  const comfy = { engine: "comfyui", gpu_index: 3, state: "idle" };

  it("shows the picked file's size when a lemonade-kind resource is drafted loaded", () => {
    expect(draftedFootprintBytes(lemonade, "loaded", 9_000_000_000, 33_000_000_000)).toBe(9_000_000_000);
  });

  it("shows zero when a lemonade-kind resource is drafted unloaded", () => {
    expect(draftedFootprintBytes(lemonade, "unloaded", 9_000_000_000, 33_000_000_000)).toBe(0);
  });

  it("shows the live footprint for a lemonade-kind resource left at don't-touch", () => {
    expect(draftedFootprintBytes(lemonade, "none", 9_000_000_000, 33_000_000_000)).toBe(5_000_000_000);
  });

  it("uses the fixed hipfire budget when drafted loaded (resume)", () => {
    expect(draftedFootprintBytes(hipfire, "loaded", 0, 33_000_000_000)).toBe(33_000_000_000);
  });

  it("shows zero when a hipfire-kind resource is drafted parked", () => {
    expect(draftedFootprintBytes(hipfire, "parked", 0, 33_000_000_000)).toBe(0);
  });

  it("shows the live footprint for a hipfire-kind resource left at don't-touch", () => {
    expect(draftedFootprintBytes(hipfire, "none", 0, 33_000_000_000)).toBe(21_000_000_000);
  });

  it("models no drafted delta for a kind with no footprint concept (comfyui) — always the live reading", () => {
    expect(draftedFootprintBytes(comfy, "freed", 0, 33_000_000_000)).toBe(0);
    expect(draftedFootprintBytes(comfy, "none", 0, 33_000_000_000)).toBe(0);
  });
});

describe("draftChoiceFor / setResourceChoice", () => {
  it("reads 'none' for a resource absent from the map", () => {
    expect(draftChoiceFor({}, "gguf-a")).toBe("none");
  });

  it("reads back exactly what setResourceChoice wrote", () => {
    const withChoice = setResourceChoice({}, "gguf-a", "loaded");
    expect(draftChoiceFor(withChoice, "gguf-a")).toBe("loaded");
    expect(withChoice["gguf-a"]).toEqual({ desired: "loaded" });
  });

  it("'none' DELETES the key rather than writing a present-but-null value", () => {
    // CRITICAL 2, at the primitive level: "don't touch" must be
    // indistinguishable from "never mentioned".
    const withChoice = setResourceChoice({}, "gguf-a", "loaded");
    const cleared = setResourceChoice(withChoice, "gguf-a", "none");
    expect("gguf-a" in cleared).toBe(false);
  });

  it("never mutates the input map (immutable update)", () => {
    const original = { "gguf-a": { desired: "loaded" as const } };
    const frozen = { ...original };
    setResourceChoice(original, "gguf-a", "unloaded");
    expect(original).toEqual(frozen);
  });

  it("leaves sibling resources untouched", () => {
    const map = { "gguf-a": { desired: "loaded" as const }, img: { desired: "freed" as const } };
    const next = setResourceChoice(map, "gguf-a", "unloaded");
    expect(next.img).toEqual({ desired: "freed" });
  });
});

describe("undeclaredResources", () => {
  it("finds a resource named in the draft that is no longer declared", () => {
    // The saved-set-outlives-forget case: app/sets.py:197's Ephemeral
    // validator 422s on this at Save time with no UI hint otherwise.
    const resources = { "gguf-a": { desired: "loaded" as const }, ghost: { desired: "unloaded" as const } };
    expect(undeclaredResources(resources, ["gguf-a", "img"])).toEqual(["ghost"]);
  });

  it("finds none when every drafted resource is still declared", () => {
    const resources = { "gguf-a": { desired: "loaded" as const } };
    expect(undeclaredResources(resources, ["gguf-a", "img"])).toEqual([]);
  });

  it("returns an empty list for an empty draft", () => {
    expect(undeclaredResources({}, ["gguf-a"])).toEqual([]);
  });

  it("sorts multiple undeclared names for deterministic rendering", () => {
    const resources = {
      zeta: { desired: "unloaded" as const },
      alpha: { desired: "unloaded" as const },
    };
    expect(undeclaredResources(resources, [])).toEqual(["alpha", "zeta"]);
  });
});
