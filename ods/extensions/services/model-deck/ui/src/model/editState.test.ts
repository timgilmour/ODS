import { describe, expect, it } from "vitest";
import {
  abandonEdit,
  addOption,
  applyParsed,
  beginEdit,
  commitEdit,
  emptyEdit,
  forSave,
  hasChanges,
  removeChip,
} from "./editState";
import { scopeKeys, toPuts } from "./settingsView";

const keys = scopeKeys("sparky", "vllm", "Qwen3-30B");

/** What the panel would actually PUT from this state. */
function shipped(s: Parameters<typeof forSave>[0]) {
  return toPuts(forSave(s), keys);
}

// A brand-new add of an option the catalog has no default for. Measured
// against the live sparky/vllm catalog (274 options, 2026-08-08): 95 start
// at `""` (catalogFilter's startingValueFor), five of them `select` widgets
// — collect-detailed-traces, gdn-prefill-backend, mm-encoder-attn-dtype,
// safetensors-load-strategy, spec-method. Those five are the one-click leak:
// `select` has no onBlur, so nothing cancels the add on the way out. The
// starting value exists purely so the chip, and therefore its editor, can
// render at all.
const provisional = addOption(emptyEdit, "engines", "spec-method", "");

describe("addOption", () => {
  it("marks an empty starting value provisional — it is a placeholder, not a value", () => {
    expect(provisional.pendingAdd).toEqual({ name: "spec-method", kind: "engines" });
  });

  it("does NOT mark a toggle's `true` provisional — that IS the value", () => {
    // startingValueFor returns `true` for a toggle, and a checked box is the
    // complete meaning of a bare flag. Treating it as provisional would make
    // "add --enforce-eager, click away" silently do nothing.
    const s = addOption(emptyEdit, "engines", "enforce-eager", true);
    expect(s.pendingAdd).toBeNull();
    expect(shipped(s)).toEqual([
      { kind: "engines", key: "sparky/vllm", values: { "enforce-eager": true }, remove: [] },
    ]);
  });

  it("does NOT mark a real catalog default provisional", () => {
    // Declaring the current default explicitly is a genuine choice: it pins
    // the value against a future upstream default change.
    const s = addOption(emptyEdit, "engines", "kv-cache-dtype", "auto");
    expect(s.pendingAdd).toBeNull();
    expect(shipped(s)[0].values).toEqual({ "kv-cache-dtype": "auto" });
  });

  it("abandons an earlier provisional add rather than stacking two", () => {
    const s = addOption(provisional, "engines", "gdn-prefill-backend", "");
    expect(s.pendingAdd).toEqual({ name: "gdn-prefill-backend", kind: "engines" });
    expect(s.buffer.sets.engines?.["spec-method"]).toBeUndefined();
  });
});

describe("forSave", () => {
  // THE defect this module exists for. Save is an exit from an open editor
  // just as much as Escape is, and it is the one no cancel path covers: the
  // `select` and `toggle` editors have no onBlur, so clicking Save with a
  // never-committed add on screen used to ship `--spec-method ''`.
  it("ships nothing for an add whose editor never committed", () => {
    expect(shipped(provisional)).toEqual([]);
  });

  it("ships the operator's other edits while dropping the provisional add", () => {
    const s = commitEdit(provisional, "engines", "max-model-len", "8192");
    // commitEdit on a DIFFERENT key: the operator moved on, so the add goes.
    expect(shipped(s)).toEqual([
      { kind: "engines", key: "sparky/vllm", values: { "max-model-len": "8192" }, remove: [] },
    ]);
  });

  it("never invents a removal for the dropped add", () => {
    // The server never saw the add, so naming it in `remove` would tell the
    // scope to delete a key it never had — the PUT SettingsModal's remove
    // guard exists to prevent.
    const s = commitEdit(provisional, "engines", "max-model-len", "8192");
    expect(shipped(s)[0].remove).toEqual([]);
  });
});

describe("hasChanges", () => {
  it("is false when the only pending thing is a provisional add", () => {
    // Save must not be offered for an add with no value chosen: there is
    // genuinely nothing to write, and an enabled Save that ships an empty
    // PUT list would close the panel as if it had done something.
    expect(hasChanges(provisional)).toBe(false);
  });

  it("is true once that add commits a value", () => {
    expect(hasChanges(commitEdit(provisional, "engines", "spec-method", "eagle"))).toBe(true);
  });
});

describe("beginEdit", () => {
  it("drops a provisional add when the operator opens another chip's editor", () => {
    const s = beginEdit(provisional, "max-model-len");
    expect(s.pendingAdd).toBeNull();
    expect(shipped(s)).toEqual([]);
  });

  it("keeps the add armed when its own editor is reopened", () => {
    // Reopening the same editor commits nothing, so the add is still
    // provisional and every later exit must still be able to take it back out.
    const s = beginEdit(provisional, "spec-method");
    expect(s.pendingAdd).toEqual({ name: "spec-method", kind: "engines" });
    expect(shipped(s)).toEqual([]);
  });
});

describe("commitEdit", () => {
  it("makes the add the operator's — it survives every later exit", () => {
    const s = commitEdit(provisional, "engines", "spec-method", "eagle");
    expect(s.pendingAdd).toBeNull();
    expect(shipped(abandonEdit(s))[0].values).toEqual({ "spec-method": "eagle" });
  });
});

describe("removeChip", () => {
  it("undoes the add itself without recording a removal", () => {
    const s = removeChip(provisional, "engines", "spec-method");
    expect(s.pendingAdd).toBeNull();
    expect(shipped(s)).toEqual([]);
  });

  it("drops a provisional add when a DIFFERENT chip is removed", () => {
    const s = removeChip(provisional, "engines", "max-model-len");
    expect(s.buffer.sets.engines?.["spec-method"]).toBeUndefined();
    expect(shipped(s)).toEqual([
      { kind: "engines", key: "sparky/vllm", values: {}, remove: ["max-model-len"] },
    ]);
  });
});

describe("the write scope moving under an open editor", () => {
  // The scope segmented control can change `kind` while an add's editor is
  // open, which is why PendingAdd carries the kind it was buffered at. These
  // are the states where the add's NAME matches the incoming action but its
  // KIND does not — matching on name alone would strand the `""` at the old
  // kind while writing at the new one.
  const atModels = addOption(emptyEdit, "engine_models", "spec-method", "");

  it("commits at the new scope without stranding the add at the old one", () => {
    const s = commitEdit(atModels, "engines", "spec-method", "eagle");
    expect(s.buffer.sets.engine_models?.["spec-method"]).toBeUndefined();
    expect(shipped(s)).toEqual([
      { kind: "engines", key: "sparky/vllm", values: { "spec-method": "eagle" }, remove: [] },
    ]);
  });

  it("removes at the new scope without stranding the add at the old one", () => {
    const s = removeChip(atModels, "engines", "spec-method");
    expect(s.buffer.sets.engine_models?.["spec-method"]).toBeUndefined();
    expect(shipped(s)).toEqual([
      { kind: "engines", key: "sparky/vllm", values: {}, remove: ["spec-method"] },
    ]);
  });

  it("still reopens the add's own editor — beginEdit has no scope of its own", () => {
    expect(beginEdit(atModels, "spec-method").pendingAdd).toEqual({
      name: "spec-method",
      kind: "engine_models",
    });
  });
});

describe("abandonEdit", () => {
  it("takes the provisional add back out", () => {
    expect(shipped(abandonEdit(provisional))).toEqual([]);
    expect(abandonEdit(provisional).pendingAdd).toBeNull();
  });

  it("leaves a committed edit alone", () => {
    const s = commitEdit(emptyEdit, "engines", "max-model-len", "8192");
    expect(abandonEdit(s)).toEqual(s);
  });
});

describe("applyParsed", () => {
  it("keeps an imported value for the key an add was still provisional on", () => {
    // Import is a real operator value arriving for a key the panel still has
    // flagged as scaffolding. Leaving the flag armed would let the next
    // Escape — or the next click on any other chip — silently delete it.
    const s = applyParsed(provisional, "engines", { "spec-method": "eagle", seed: "1" });
    expect(s.pendingAdd).toBeNull();
    expect(shipped(s)[0].values).toEqual({ "spec-method": "eagle", seed: "1" });
  });

  it("drops a provisional add the import did not name", () => {
    const s = applyParsed(provisional, "engines", { seed: "1" });
    expect(shipped(s)[0].values).toEqual({ seed: "1" });
  });
});

describe("the walk-away paths, end to end", () => {
  // Each of these is a way out of a never-committed `select` add. Every one
  // of them shipped `--spec-method ''` at some point in this feature's
  // history; the point of the module is that they are enumerable here.
  const exits: Array<[string, (s: typeof provisional) => typeof provisional]> = [
    ["Escape / empty blur", abandonEdit],
    ["opening another chip's editor", (s) => beginEdit(s, "max-model-len")],
    ["removing another chip", (s) => removeChip(s, "engines", "max-model-len")],
    ["adding a second option", (s) => addOption(s, "engines", "gdn-prefill-backend", "")],
    ["importing an argline", (s) => applyParsed(s, "engines", { seed: "1" })],
  ];

  for (const [name, exit] of exits) {
    it(`${name} never ships --spec-method ''`, () => {
      const puts = shipped(exit(provisional));
      for (const put of puts) {
        expect(put.values).not.toHaveProperty("spec-method");
        expect(put.remove).not.toContain("spec-method");
      }
    });
  }

  it("clicking Save with the editor still open never ships it either", () => {
    const puts = shipped(provisional);
    expect(puts).toEqual([]);
  });
});
