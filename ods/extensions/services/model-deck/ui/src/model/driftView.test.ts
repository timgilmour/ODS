import { describe, expect, it } from "vitest";
import { displayKeyFor, driftRows, driftValueText, partitionDrift } from "./driftView";

describe("displayKeyFor", () => {
  it("strips the args: namespace prefix for display", () => {
    expect(displayKeyFor("args:max-model-len")).toBe("max-model-len");
  });

  it("keeps the env: prefix visible", () => {
    expect(displayKeyFor("env:VLLM_USE_V1")).toBe("env:VLLM_USE_V1");
  });

  it("keeps the container: prefix visible", () => {
    expect(displayKeyFor("container:ulimits")).toBe("container:ulimits");
  });

  it("leaves an unqualified name (defensive — every real key is qualified) untouched", () => {
    expect(displayKeyFor("max-model-len")).toBe("max-model-len");
  });
});

describe("driftValueText", () => {
  it("renders a string verbatim, unquoted", () => {
    expect(driftValueText("131072")).toBe("131072");
  });

  it("renders an empty string as itself, not an empty JSON string", () => {
    expect(driftValueText("")).toBe("");
  });

  it("JSON.stringifies a boolean compactly", () => {
    expect(driftValueText(true)).toBe("true");
  });

  it("JSON.stringifies a number compactly", () => {
    expect(driftValueText(4)).toBe("4");
  });

  it("JSON.stringifies a string array compactly", () => {
    expect(driftValueText(["a", "b"])).toBe('["a","b"]');
  });

  it("JSON.stringifies an arbitrary container mapping compactly", () => {
    expect(driftValueText({ nofile: 65536 })).toBe('{"nofile":65536}');
  });
});

describe("driftRows", () => {
  it("shapes an ordinary old->new entry", () => {
    const rows = driftRows([
      { key: "args:max-model-len", old: "262144", new: "131072", ts: "2026-08-07T00:00:00Z" },
    ]);
    expect(rows).toEqual([
      { key: "args:max-model-len", displayKey: "max-model-len", oldText: "262144", newText: "131072" },
    ]);
  });

  it("keeps the env: namespace prefix visible on the display key", () => {
    const rows = driftRows([
      { key: "env:VLLM_USE_V1", old: "0", new: "1", ts: "2026-08-07T00:00:00Z" },
    ]);
    expect(rows[0].displayKey).toBe("env:VLLM_USE_V1");
  });

  it("marks a null old value as added rather than a literal 'null'", () => {
    const rows = driftRows([
      { key: "args:enable-auto-tool-choice", old: null, new: true, ts: "2026-08-07T00:00:00Z" },
    ]);
    expect(rows[0].oldText).toBeNull();
    expect(rows[0].newText).toBe("true");
  });

  it("marks a null new value as removed rather than a literal 'null'", () => {
    const rows = driftRows([
      { key: "args:enable-auto-tool-choice", old: "true", new: null, ts: "2026-08-07T00:00:00Z" },
    ]);
    expect(rows[0].oldText).toBe("true");
    expect(rows[0].newText).toBeNull();
  });

  it("never fabricates a value for a non-string/non-JSON-primitive shape it does not recognize — JSON.stringify is the only rule", () => {
    const rows = driftRows([
      { key: "container:ulimits", old: { nofile: 1024 }, new: { nofile: 65536 }, ts: "2026-08-07T00:00:00Z" },
    ]);
    expect(rows[0].oldText).toBe('{"nofile":1024}');
    expect(rows[0].newText).toBe('{"nofile":65536}');
  });

  it("maps an empty entries list to an empty rows list", () => {
    expect(driftRows([])).toEqual([]);
  });
});

describe("partitionDrift", () => {
  const entry = (key: string) => ({ key, old: "a", new: "b", ts: "2026-08-07T00:00:00Z" });

  it("splits a MIXED payload: rows for entried keys, names for the rest", () => {
    // F2, final branch review 2026-08-07. app/routers/__init__.py's
    // `_settings_drift` chooses per (scope, namespace): a journal-bearing
    // namespace yields exact entries, a pre-journal one yields names in
    // `changed` only — so one report holds both. Live ds4 hits this on its
    // first post-deploy settings edit (engine scope journaled, model scope
    // still carrying pre-journal stamps).
    const { rows, legacy } = partitionDrift(
      ["args:max-model-len", "args:gpu-memory-utilization", "env:VLLM_USE_V1"],
      [entry("args:max-model-len")],
    );

    expect(rows.map((r) => r.key)).toEqual(["args:max-model-len"]);
    // Every remaining key the header's "3 keys changed" counts is still
    // shown — the defect was these vanishing behind the rows.
    expect(legacy).toEqual([
      { key: "args:gpu-memory-utilization", displayKey: "gpu-memory-utilization" },
      { key: "env:VLLM_USE_V1", displayKey: "env:VLLM_USE_V1" },
    ]);
    expect(rows.length + legacy.length).toBe(3);
  });

  it("a fully journaled report has no legacy names", () => {
    const { rows, legacy } = partitionDrift(
      ["args:max-model-len", "env:VLLM_USE_V1"],
      [entry("args:max-model-len"), entry("env:VLLM_USE_V1")],
    );
    expect(rows).toHaveLength(2);
    expect(legacy).toEqual([]);
  });

  it("a fully legacy report has no rows and keeps every name in order", () => {
    const { rows, legacy } = partitionDrift(["args:max-model-len", "args:seed"], []);
    expect(rows).toEqual([]);
    expect(legacy.map((k) => k.displayKey)).toEqual(["max-model-len", "seed"]);
  });

  it("pairs on the display key, so an unqualified name is not printed twice", () => {
    // Defensive: both sides are namespace-qualified by the same backend
    // fold, so this only differs from a raw comparison for a fixture or an
    // older report that lost the prefix on one side.
    const { legacy } = partitionDrift(["max-model-len"], [entry("args:max-model-len")]);
    expect(legacy).toEqual([]);
  });

  it("same-named keys in different namespaces stay distinct", () => {
    const { rows, legacy } = partitionDrift(
      ["args:seed", "env:seed"],
      [entry("args:seed")],
    );
    expect(rows.map((r) => r.key)).toEqual(["args:seed"]);
    expect(legacy.map((k) => k.key)).toEqual(["env:seed"]);
  });
});
