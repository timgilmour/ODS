import { describe, expect, it } from "vitest";
import { displayKeyFor, driftRows, driftValueText } from "./driftView";

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
