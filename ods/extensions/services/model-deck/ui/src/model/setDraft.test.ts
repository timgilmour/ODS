import { describe, expect, it } from "vitest";
import type { ConfigSet } from "../api";
import {
  buildDraft,
  derivePlacedModel,
  draftEquals,
  fieldsFromSet,
} from "./setDraft";

const base: ConfigSet = {
  name: "Chat mode",
  notes: "notes",
  durable: { default_route_model: "extra.m.gguf", activate_model_id: null },
  ephemeral: {
    lemonade: { state: "loaded" },
    comfyui: null,
    hipfire: null,
  },
  policy_overrides: null,
};

describe("fieldsFromSet → buildDraft round-trip", () => {
  it("CRITICAL 2: a set that never mentioned hipfire keeps hipfire null", () => {
    const round = buildDraft(fieldsFromSet(base, false));
    expect(round.ephemeral?.hipfire).toBeNull();
  });

  it("carries policy_overrides verbatim through load → save", () => {
    const withOverrides: ConfigSet = {
      ...base,
      policy_overrides: { lemonade: { priority: 100, pinned: true, idle_ttl: 0 } },
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
});

describe("derivePlacedModel", () => {
  it("derives the file from an extra.-prefixed route with a loaded intent", () => {
    expect(derivePlacedModel(base)).toBe("m.gguf");
  });

  it("returns null for a non-extra route (cannot be traced to a library file)", () => {
    const other: ConfigSet = {
      ...base,
      durable: { default_route_model: "gpt-oss-120b", activate_model_id: null },
    };
    expect(derivePlacedModel(other)).toBeNull();
  });

  it("returns null when lemonade is not being loaded", () => {
    const unloaded: ConfigSet = {
      ...base,
      ephemeral: { ...base.ephemeral!, lemonade: { state: "unloaded" } },
    };
    expect(derivePlacedModel(unloaded)).toBeNull();
  });

  it("returns null when durable is absent", () => {
    expect(derivePlacedModel({ ...base, durable: null })).toBeNull();
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
});
