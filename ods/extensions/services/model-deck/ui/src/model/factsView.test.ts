import { describe, expect, it } from "vitest";
import type { FactEntry, FactsDriftItem } from "../api";
import {
  declaredTags,
  declaredText,
  factRows,
  factValueText,
  partitionDrift,
  tagsSettled,
  tagsWith,
} from "./factsView";

function fact(value: unknown, origin: FactEntry["origin"] = "derived"): FactEntry {
  return { value, origin, source: "config.json", derived_ts: "2026-08-07T12:00:00Z" };
}

// The real drift shape — app/facts.py:detect_drift appends exactly these six
// fields, and `field` is the RUNTIME field name, never the fact's.
function drift(field: string, severity = "mismatch"): FactsDriftItem {
  return {
    field,
    expected: 131072,
    actual: 262144,
    expected_source: "config.json",
    actual_source: "runtime config",
    severity,
  };
}

describe("factValueText", () => {
  it("passes a string through verbatim — a checkpoint id is its identity", () => {
    expect(factValueText("compressed-tensors")).toBe("compressed-tensors");
  });

  it("renders numbers and booleans as themselves", () => {
    expect(factValueText(131072)).toBe("131072");
    expect(factValueText(true)).toBe("true");
  });

  it("JSON-renders an object rather than letting it stringify to [object Object]", () => {
    expect(factValueText({ heretic: { identity: "Qwen3", service: "vllm" } })).toBe(
      '{"heretic":{"identity":"Qwen3","service":"vllm"}}',
    );
  });

  it("JSON-renders a list rather than losing the brackets to comma-joining", () => {
    expect(factValueText(["a", "b"])).toBe('["a","b"]');
  });

  it("prints null as null — a fact that is null is still a recorded fact", () => {
    expect(factValueText(null)).toBe("null");
  });

  it("renders an absent value as empty, never the word undefined", () => {
    expect(factValueText(undefined)).toBe("");
  });
});

describe("declaredText", () => {
  it("takes a declared string as the draft", () => {
    expect(declaredText("the fast one")).toBe("the fast one");
  });

  it("refuses a non-string rather than coercing a derived value into the form", () => {
    // A shadowed DERIVED value under the same name is not something a human
    // typed, so the editor must not open pre-filled with it.
    expect(declaredText(42)).toBe("");
    expect(declaredText(null)).toBe("");
    expect(declaredText(undefined)).toBe("");
    expect(declaredText({ a: 1 })).toBe("");
  });
});

describe("declaredTags", () => {
  it("takes a list of strings — app/declared.py's validator shape", () => {
    expect(declaredTags(["fast", "deep"])).toEqual(["fast", "deep"]);
  });

  it("is empty for anything that is not a list", () => {
    expect(declaredTags(undefined)).toEqual([]);
    expect(declaredTags("fast")).toEqual([]);
    expect(declaredTags({ 0: "fast" })).toEqual([]);
  });

  it("drops a non-string element instead of discarding the whole list", () => {
    expect(declaredTags(["fast", 3, null, "deep"])).toEqual(["fast", "deep"]);
  });
});

describe("factRows", () => {
  it("is empty for a key the facts map has no entry for", () => {
    expect(factRows(undefined)).toEqual([]);
  });

  it("sorts by name so declaring a fact cannot reorder the table", () => {
    // resolve_facts emits derived first, then declared: unsorted, the table
    // would rearrange itself the moment a human declared anything.
    const entry = { quant_method: fact("modelopt"), label: fact("x", "declared") };
    expect(factRows(entry).map(([name]) => name)).toEqual(["label", "quant_method"]);
  });
});

describe("partitionDrift", () => {
  it("has nothing to report when the key has no drift", () => {
    const { byName, unmatched } = partitionDrift(undefined, ["quant_method"]);
    expect(byName.size).toBe(0);
    expect(unmatched).toEqual([]);
  });

  it("keeps today's three rules visible even though none names a fact row", () => {
    // The live case: DRIFT_RULES (app/facts.py:31-36) reports runtime field
    // names — quantization / max_model_len / max_input_tokens — while the
    // facts are quant_method / max_position_embeddings / max_model_len_live.
    // Matching on the name alone would render drift nowhere.
    const items = [drift("quantization", "crash"), drift("max_model_len")];
    const { byName, unmatched } = partitionDrift(items, [
      "quant_method",
      "max_position_embeddings",
    ]);
    expect(byName.size).toBe(0);
    expect(unmatched.map((i) => i.field)).toEqual(["quantization", "max_model_len"]);
  });

  it("attaches an item to the row it names when the names do coincide", () => {
    const { byName, unmatched } = partitionDrift([drift("max_model_len")], ["max_model_len"]);
    expect(byName.get("max_model_len")?.severity).toBe("mismatch");
    expect(unmatched).toEqual([]);
  });

  it("lists a second item for the same row rather than dropping it", () => {
    const items = [drift("max_model_len", "mismatch"), drift("max_model_len", "crash")];
    const { byName, unmatched } = partitionDrift(items, ["max_model_len"]);
    expect(byName.get("max_model_len")?.severity).toBe("mismatch");
    expect(unmatched.map((i) => i.severity)).toEqual(["crash"]);
  });
});

describe("tagsWith", () => {
  it("appends a trimmed tag to the list that will be PUT", () => {
    expect(tagsWith(["fast"], "  deep ")).toEqual(["fast", "deep"]);
  });

  it("refuses a duplicate — so the call site can keep the typed text", () => {
    // Clearing the box on a duplicate left an operator watching their word
    // vanish with no chip to show for it.
    expect(tagsWith(["fast"], "fast")).toBeNull();
    expect(tagsWith(["fast"], " fast ")).toBeNull();
  });

  it("refuses a blank entry", () => {
    expect(tagsWith(["fast"], "")).toBeNull();
    expect(tagsWith(["fast"], "   ")).toBeNull();
  });

  it("never mutates the list it was given", () => {
    const tags = ["fast"];
    tagsWith(tags, "deep");
    expect(tags).toEqual(["fast"]);
  });
});

describe("tagsSettled", () => {
  it("is false while nothing has been written", () => {
    expect(tagsSettled(null, ["fast"])).toBe(false);
  });

  it("is false while the server still carries the PRE-write list", () => {
    // The loss window: a second edit composed against this would ship the
    // pre-write array and undo the first (remove A then B resurrects A).
    expect(tagsSettled(["fast"], ["fast", "deep"])).toBe(false);
  });

  it("is true once the server's own list matches what was written", () => {
    expect(tagsSettled(["fast", "deep"], ["fast", "deep"])).toBe(true);
  });

  it("treats an empty write as settled only against an empty server list", () => {
    // Removing the last tag is a real write, and it has to be able to settle.
    expect(tagsSettled([], [])).toBe(true);
    expect(tagsSettled([], undefined)).toBe(true);
    expect(tagsSettled([], ["fast"])).toBe(false);
  });

  it("is order-sensitive, because the whole array is what gets replaced", () => {
    expect(tagsSettled(["fast", "deep"], ["deep", "fast"])).toBe(false);
  });

  it("holds rather than settles against a malformed server value", () => {
    expect(tagsSettled(["fast"], "fast")).toBe(false);
  });
});
