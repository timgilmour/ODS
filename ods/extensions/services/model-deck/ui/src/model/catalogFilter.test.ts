import { describe, expect, it } from "vitest";
import type { Catalog, CatalogOption } from "../api";
import { filterCatalog, type CatalogRow } from "./catalogFilter";

function option(overrides: Partial<CatalogOption> = {}): CatalogOption {
  return {
    aliases: [],
    type: "string",
    choices: null,
    default: "None",
    nargs: undefined,
    repeatable: false,
    help: "An option",
    widget: "text",
    ...overrides,
  };
}

function catalog(options: Record<string, CatalogOption>): Catalog {
  return {
    engine_version: "1.0.0",
    harvested_ts: null,
    options,
  };
}

describe("filterCatalog", () => {
  it("returns rows sorted by name", () => {
    const c = catalog({
      "zebra-mode": option(),
      "alpha-mode": option(),
      "beta-mode": option(),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["alpha-mode", "beta-mode", "zebra-mode"]);
  });

  it("computes flag from name as --${name}", () => {
    const c = catalog({ "my-opt": option() });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows[0].flag).toBe("--my-opt");
  });

  it("matches query against option name (case-insensitive substring)", () => {
    const c = catalog({
      "context-length": option(),
      "max-tokens": option(),
      "temperature": option(),
    });

    const rows = filterCatalog(c, {
      query: "context",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["context-length"]);
  });

  it("matches query against aliases (dash-insensitive)", () => {
    const c = catalog({
      "max-model-len": option({ aliases: ["-q", "--ctx-len"] }),
      "other-opt": option(),
    });

    const rows = filterCatalog(c, {
      query: "q",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["max-model-len"]);
  });

  it("matches dash-prefixed query against dash-stripped aliases", () => {
    const c = catalog({
      "max-model-len": option({ aliases: ["-q", "--ctx-len"] }),
      "other-opt": option(),
    });

    const rows = filterCatalog(c, {
      query: "--q",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["max-model-len"]);
  });

  it("matches short alias -q via query 'q'", () => {
    const c = catalog({
      "max-model-len": option({ aliases: ["-q"] }),
      "other-opt": option(),
    });

    const rows = filterCatalog(c, {
      query: "q",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["max-model-len"]);
  });

  it("matches query against help text (case-insensitive substring)", () => {
    const c = catalog({
      "max-len": option({ help: "Maximum context length for inference" }),
      "min-len": option({ help: "Minimum length requirement" }),
    });

    const rows = filterCatalog(c, {
      query: "context",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["max-len"]);
  });

  it("matches case-insensitively", () => {
    const c = catalog({
      "MyOption": option({ help: "HELP TEXT" }),
      "other": option(),
    });

    const rows = filterCatalog(c, {
      query: "myopt",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["MyOption"]);

    const rows2 = filterCatalog(c, {
      query: "HELP",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows2.map((r) => r.name)).toEqual(["MyOption"]);
  });

  it("matches empty query against all options", () => {
    const c = catalog({
      "opt-a": option(),
      "opt-b": option(),
      "opt-c": option(),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.length).toBe(3);
  });

  it("matches whitespace query against all options", () => {
    const c = catalog({
      "opt-a": option(),
      "opt-b": option(),
    });

    const rows = filterCatalog(c, {
      query: "   ",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.length).toBe(2);
  });

  it("filters by widget type when widgets array is non-empty", () => {
    const c = catalog({
      "toggle-opt": option({ widget: "toggle" }),
      "text-opt": option({ widget: "text" }),
      "list-opt": option({ widget: "list" }),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: ["toggle", "text"],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["text-opt", "toggle-opt"]);
  });

  it("does not filter by widget when widgets array is empty", () => {
    const c = catalog({
      "toggle-opt": option({ widget: "toggle" }),
      "text-opt": option({ widget: "text" }),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.length).toBe(2);
  });

  it("filters to setNames when setOnly is true", () => {
    const c = catalog({
      "opt-a": option(),
      "opt-b": option(),
      "opt-c": option(),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: true,
      setNames: new Set(["opt-a", "opt-c"]),
    });

    expect(rows.map((r) => r.name)).toEqual(["opt-a", "opt-c"]);
  });

  it("returns empty array when setOnly is true but setNames is empty", () => {
    const c = catalog({
      "opt-a": option(),
      "opt-b": option(),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: true,
      setNames: new Set(),
    });

    expect(rows).toEqual([]);
  });

  it("badges rows in setNames with isSet=true even when setOnly is false", () => {
    const c = catalog({
      "opt-a": option(),
      "opt-b": option(),
      "opt-c": option(),
    });

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(["opt-b"]),
    });

    expect(rows.map((r) => ({ name: r.name, isSet: r.isSet }))).toEqual([
      { name: "opt-a", isSet: false },
      { name: "opt-b", isSet: true },
      { name: "opt-c", isSet: false },
    ]);
  });

  it("combines query and widget filters", () => {
    const c = catalog({
      "context-len": option({ widget: "text" }),
      "temperature": option({ widget: "number" }),
      "context-scale": option({ widget: "toggle" }),
    });

    const rows = filterCatalog(c, {
      query: "context",
      widgets: ["text", "toggle"],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows.map((r) => r.name)).toEqual(["context-len", "context-scale"]);
  });

  it("includes the full option in the row", () => {
    const opts = {
      "my-opt": option({
        aliases: ["-m"],
        type: "integer",
        help: "My help text",
        widget: "number",
      }),
    };
    const c = catalog(opts);

    const rows = filterCatalog(c, {
      query: "",
      widgets: [],
      setOnly: false,
      setNames: new Set(),
    });

    expect(rows[0].option).toEqual(opts["my-opt"]);
  });
});
