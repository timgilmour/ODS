import { describe, expect, test } from "vitest";
import type { DeclaredEngine, EngineKindsResponse } from "../api";
import { labels } from "./messages";
import {
  canSave,
  emptyForm,
  formErrors,
  formForEntry,
  setField,
  sortedEngines,
  toPayload,
  withKind,
  type EngineFormState,
} from "./engineForm";

// Fixture builder mirroring GET /api/engine-kinds's shape
// ({"kinds": [{kind, connection, human_verbs}, ...]}, app/routers/
// nodes.py:324-339) — never the live three (lemonade/comfyui/hipfire)
// except where the brief's own pinned tests name "comfyui" verbatim; every
// other test below uses invented kinds ("widget"/"gadget") per the fixture
// rule, since this module treats `kind` as opaque data, not a closed enum.
function kindsPayload(
  kinds: {
    kind: string;
    connection: Record<string, { required: boolean }>;
    human_verbs: string[];
  }[],
): EngineKindsResponse {
  return { kinds };
}

// The brief's Step 1 (pinned verbatim).
test("fields derive from the kinds payload, not literals", () => {
  const form = emptyForm(
    kindsPayload([{ kind: "comfyui", connection: { url: { required: true } }, human_verbs: ["free"] }]),
    "comfyui",
  );
  expect(Object.keys(form.connection)).toEqual(["url"]);
});

test("save is blocked until required fields are filled", () => {
  const form = emptyForm(
    kindsPayload([{ kind: "comfyui", connection: { url: { required: true } }, human_verbs: ["free"] }]),
    "comfyui",
  );
  expect(canSave(form)).toBe(false);
  expect(canSave(setField(form, "url", "http://img:8188"))).toBe(true);
});

// Invented two-kind fixture for the rest of this file: "widget" has one
// required and one optional connection field; "gadget" has a differently
// named single required field, so a kind switch has something real to
// prove it drops (never carries a same-named-by-coincidence value across).
const WIDGET_GADGET = kindsPayload([
  { kind: "widget", connection: { host: { required: true }, note: { required: false } }, human_verbs: ["free"] },
  { kind: "gadget", connection: { path: { required: true } }, human_verbs: ["load", "unload"] },
]);

describe("emptyForm", () => {
  test("seeds an optional connection field too, alongside the required one", () => {
    const form = emptyForm(WIDGET_GADGET, "widget");
    expect(form.connection).toEqual({ host: "", note: "" });
    expect(form.requiredConnectionFields).toEqual(["host"]);
  });

  test("defaults resource blank, no GPU picked, and a conservative policy", () => {
    const form = emptyForm(WIDGET_GADGET, "widget");
    expect(form.resource).toBe("");
    expect(form.gpuIndex).toBeNull();
    expect(form.priority).toBe(0);
    expect(form.pinned).toBe(false);
    // idle_ttl 0 = never idle-release (app.engine_kinds's per-kind
    // idle_action all gate on `policy["idle_ttl"] > 0`) — the conservative
    // default for something just declared.
    expect(form.idleTtl).toBe(0);
  });

  test("an unknown kind (not in the payload) yields an empty, harmless connection buffer", () => {
    const form = emptyForm(WIDGET_GADGET, "no-such-kind");
    expect(form.connection).toEqual({});
    expect(form.requiredConnectionFields).toEqual([]);
  });
});

describe("formForEntry", () => {
  const entry: DeclaredEngine = {
    resource: "widget-a",
    kind: "widget",
    connection: { host: "http://widget-a:9000", note: "spare" },
    gpu_index: 2,
    policy_defaults: { priority: 10, pinned: true, idle_ttl: 300 },
  };

  test("pre-fills every field, including the connection values and gpu/policy", () => {
    const form = formForEntry(entry, WIDGET_GADGET);
    expect(form.resource).toBe("widget-a");
    expect(form.kind).toBe("widget");
    expect(form.connection).toEqual({ host: "http://widget-a:9000", note: "spare" });
    expect(form.gpuIndex).toBe(2);
    expect(form.priority).toBe(10);
    expect(form.pinned).toBe(true);
    expect(form.idleTtl).toBe(300);
  });

  test("derives requiredConnectionFields from the entry's OWN kind in the payload", () => {
    const form = formForEntry(entry, WIDGET_GADGET);
    expect(form.requiredConnectionFields).toEqual(["host"]);
  });

  test("an entry whose kind the current kinds payload no longer knows heals to no required fields, not a crash", () => {
    const stale: DeclaredEngine = { ...entry, kind: "vanished-kind" };
    const form = formForEntry(stale, WIDGET_GADGET);
    expect(form.connection).toEqual({ host: "http://widget-a:9000", note: "spare" });
    expect(form.requiredConnectionFields).toEqual([]);
  });
});

describe("withKind", () => {
  test("rebuilds connection for the new kind's own schema, dropping the old kind's values", () => {
    const started = setField(emptyForm(WIDGET_GADGET, "widget"), "host", "http://widget:9000");
    const switched = withKind(started, WIDGET_GADGET, "gadget");
    expect(switched.connection).toEqual({ path: "" });
    expect(switched.requiredConnectionFields).toEqual(["path"]);
  });

  test("keeps resource, gpu, and policy across the switch — only connection identity changes", () => {
    const started: EngineFormState = {
      ...emptyForm(WIDGET_GADGET, "widget"),
      resource: "my-engine",
      gpuIndex: 3,
      priority: 5,
      pinned: true,
      idleTtl: 120,
    };
    const switched = withKind(started, WIDGET_GADGET, "gadget");
    expect(switched.resource).toBe("my-engine");
    expect(switched.gpuIndex).toBe(3);
    expect(switched.priority).toBe(5);
    expect(switched.pinned).toBe(true);
    expect(switched.idleTtl).toBe(120);
  });
});

describe("setField", () => {
  test("touches only the named connection field, nothing else on the form", () => {
    const form = emptyForm(WIDGET_GADGET, "widget");
    const next = setField(form, "note", "spare unit");
    expect(next.connection).toEqual({ host: "", note: "spare unit" });
    expect(next.resource).toBe(form.resource);
    expect(next.kind).toBe(form.kind);
    expect(next.requiredConnectionFields).toEqual(form.requiredConnectionFields);
  });

  test("does not mutate the original form (pure)", () => {
    const form = emptyForm(WIDGET_GADGET, "widget");
    setField(form, "host", "http://x:1");
    expect(form.connection.host).toBe("");
  });
});

describe("canSave", () => {
  // Documents the narrow, pinned scope: canSave is a connection-only check
  // (see the brief's own two tests above) — it does NOT gate on resource or
  // gpuIndex being set. formErrors below is the full save gate.
  test("is true with every required connection field filled, even with no resource or GPU chosen", () => {
    const form = setField(emptyForm(WIDGET_GADGET, "widget"), "host", "http://widget:9000");
    expect(form.resource).toBe("");
    expect(form.gpuIndex).toBeNull();
    expect(canSave(form)).toBe(true);
  });

  test("a kind with no required connection fields at all is save-able from emptyForm alone", () => {
    const noRequired = kindsPayload([
      { kind: "gizmo", connection: { note: { required: false } }, human_verbs: [] },
    ]);
    expect(canSave(emptyForm(noRequired, "gizmo"))).toBe(true);
  });
});

describe("formErrors", () => {
  test("names every missing structural + connection-required field on a blank form", () => {
    const errors = formErrors(emptyForm(WIDGET_GADGET, "widget"));
    expect(errors).toContain(labels.engineResourceRequired);
    expect(errors).toContain(labels.engineGpuRequired);
    expect(errors).toContain(labels.engineConnectionFieldRequired("host"));
    // "note" is optional — must not be named as missing.
    expect(errors).not.toContain(labels.engineConnectionFieldRequired("note"));
  });

  test("is empty once resource, gpu, and every required connection field are filled", () => {
    let form = emptyForm(WIDGET_GADGET, "widget");
    form = { ...form, resource: "widget-a", gpuIndex: 1 };
    form = setField(form, "host", "http://widget-a:9000");
    expect(formErrors(form)).toEqual([]);
  });
});

describe("toPayload", () => {
  test("produces exactly validate_engines' accepted shape — no extra keys, gpu_index numeric, policy_defaults nested", () => {
    let form = emptyForm(WIDGET_GADGET, "widget");
    form = { ...form, resource: "widget-a", gpuIndex: 2, priority: 7, pinned: true, idleTtl: 90 };
    form = setField(form, "host", "http://widget-a:9000");
    expect(toPayload(form)).toEqual({
      resource: "widget-a",
      kind: "widget",
      connection: { host: "http://widget-a:9000", note: "" },
      gpu_index: 2,
      policy_defaults: { priority: 7, pinned: true, idle_ttl: 90 },
    });
  });

  test("has exactly five top-level keys — matching engine_kinds.py:112-113's extra-field refusal", () => {
    let form = emptyForm(WIDGET_GADGET, "gadget");
    form = { ...form, resource: "g", gpuIndex: 0 };
    form = setField(form, "path", "/dev/g0");
    expect(Object.keys(toPayload(form)).sort()).toEqual(
      ["connection", "gpu_index", "kind", "policy_defaults", "resource"],
    );
  });
});

describe("sortedEngines", () => {
  const a: DeclaredEngine = {
    resource: "zeta", kind: "widget", connection: {}, gpu_index: 1,
    policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  };
  const b: DeclaredEngine = {
    resource: "alpha", kind: "widget", connection: {}, gpu_index: 1,
    policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  };
  const c: DeclaredEngine = {
    resource: "middle", kind: "gadget", connection: {}, gpu_index: 0,
    policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  };

  test("orders by gpu_index first, then resource name breaks a tie on the same GPU", () => {
    expect(sortedEngines([a, b, c]).map((e) => e.resource)).toEqual(["middle", "alpha", "zeta"]);
  });

  test("does not mutate its input array", () => {
    const input = [a, b, c];
    const copy = [...input];
    sortedEngines(input);
    expect(input).toEqual(copy);
  });
});
