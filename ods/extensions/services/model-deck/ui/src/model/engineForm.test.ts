import { describe, expect, test } from "vitest";
import type { DeclaredEngine, EngineKindsResponse, RemoteTenant, ResourceTenant } from "../api";
import { labels } from "./messages";
import {
  canSave,
  demandFor,
  emptyForm,
  formErrors,
  formForEntry,
  idleReleaseFor,
  kindsFor,
  resourceKindMap,
  setField,
  sortedEngines,
  toPayload,
  withKind,
  type EngineFormState,
} from "./engineForm";

// Fixture builder mirroring GET /api/engine-kinds's shape
// ({"kinds": [{kind, connection, remote_capable, local_capable, demand,
// human_verbs, idle_release}, ...]}, app/routers/nodes.py:493-519) — never
// the live four (lemonade/comfyui/hipfire/sglang-omni) except where a
// test's own pin names them verbatim (this file's Step 1, and kindsFor's
// own pinned real-payload test below); every other test uses invented
// kinds ("widget"/"gadget") per the fixture rule, since this module treats
// `kind` as opaque data, not a closed enum. `remote_capable`/`local_capable`
// are spelled out on every fixture kind (never defaulted) so a reader never
// has to guess which capability a given test kind carries — same posture
// the WIDGET_GADGET comment below already takes for connection fields.
// `idle_release` on a PINNED REAL kind must match the verified backend
// truth (app/engine_kinds.py's arbiter_verbs(): lemonade/comfyui/
// sglang-omni non-empty, hipfire empty) — a fabricated value there would
// let a fixture-honesty defect hide the same way FINDING 5's human_verbs
// slip did.
function kindsPayload(
  kinds: {
    kind: string;
    connection: Record<string, { required: boolean }>;
    remote_capable: boolean;
    local_capable: boolean;
    demand: boolean;
    human_verbs: string[];
    idle_release: boolean;
  }[],
): EngineKindsResponse {
  return { kinds };
}

// The brief's Step 1 (pinned verbatim); local_capable/remote_capable mirror
// comfyui's real KNOWN_KINDS entry (app/engine_kinds.py:180-181) — this test
// isn't about capability, but a pinned real-kind fixture should still carry
// its real flags rather than an arbitrary placeholder.
test("fields derive from the kinds payload, not literals", () => {
  const form = emptyForm(
    kindsPayload([{
      kind: "comfyui", connection: { url: { required: true } },
      remote_capable: false, local_capable: true, demand: false, human_verbs: ["free"], idle_release: true,
    }]),
    "comfyui",
  );
  expect(Object.keys(form.connection)).toEqual(["url"]);
});

test("save is blocked until required fields are filled", () => {
  const form = emptyForm(
    kindsPayload([{
      kind: "comfyui", connection: { url: { required: true } },
      remote_capable: false, local_capable: true, demand: false, human_verbs: ["free"], idle_release: true,
    }]),
    "comfyui",
  );
  expect(canSave(form)).toBe(false);
  expect(canSave(setField(form, "url", "http://img:8188"))).toBe(true);
});

// Invented two-kind fixture for the rest of this file: "widget" has one
// required and one optional connection field; "gadget" has a differently
// named single required field, so a kind switch has something real to
// prove it drops (never carries a same-named-by-coincidence value across).
// Capability flags are deliberately asymmetric (widget local-only, gadget
// remote-only) rather than both permissive — a stray capability check
// accidentally added to emptyForm/formForEntry/withKind (none of which take
// an `isRemote` argument today) would show up as a failure in THIS file's
// unrelated tests, not just kindsFor's own.
const WIDGET_GADGET = kindsPayload([
  {
    kind: "widget", connection: { host: { required: true }, note: { required: false } },
    remote_capable: false, local_capable: true, demand: true, human_verbs: ["free"], idle_release: true,
  },
  {
    kind: "gadget", connection: { path: { required: true } },
    remote_capable: true, local_capable: false, demand: false, human_verbs: ["load", "unload"], idle_release: false,
  },
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
      {
        kind: "gizmo", connection: { note: { required: false } },
        remote_capable: false, local_capable: true, demand: true, human_verbs: [], idle_release: true,
      },
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
    form = { ...form, resource: "widget-a", gpuIndex: 5 };
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

  test("has exactly five top-level keys — matching engine_kinds.py:220-223's extra-field refusal", () => {
    let form = emptyForm(WIDGET_GADGET, "gadget");
    form = { ...form, resource: "g", gpuIndex: 3 };
    form = setField(form, "path", "/dev/g0");
    expect(Object.keys(toPayload(form)).sort()).toEqual(
      ["connection", "gpu_index", "kind", "policy_defaults", "resource"],
    );
  });
});

describe("sortedEngines", () => {
  const a: DeclaredEngine = {
    resource: "zeta", kind: "widget", connection: {}, gpu_index: 5,
    policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  };
  const b: DeclaredEngine = {
    resource: "alpha", kind: "widget", connection: {}, gpu_index: 5,
    policy_defaults: { priority: 0, pinned: false, idle_ttl: 0 },
  };
  const c: DeclaredEngine = {
    resource: "middle", kind: "gadget", connection: {}, gpu_index: 3,
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

describe("kindsFor", () => {
  // The real payload GET /api/engine-kinds serves today (app/engine_kinds.py
  // KNOWN_KINDS, :177-192) — pinned verbatim (sglang-omni Task 10's own
  // obligation: prove the picker offers sglang-omni on a node-agent target
  // and withholds it on a local one, against a fixture mirroring the real
  // serialization, not an invented stand-in).
  const REAL_KINDS = kindsPayload([
    {
      kind: "lemonade",
      connection: { url: { required: true }, metrics_url: { required: true }, container: { required: true } },
      remote_capable: false, local_capable: true, demand: true, human_verbs: ["load", "unload"], idle_release: true,
    },
    {
      kind: "comfyui", connection: { url: { required: true } },
      remote_capable: false, local_capable: true, demand: false, human_verbs: ["free"], idle_release: true,
    },
    {
      kind: "hipfire", connection: { container: { required: true } },
      remote_capable: false, local_capable: true, demand: false, human_verbs: ["park", "resume"], idle_release: false,
    },
    {
      kind: "sglang-omni", connection: { url: { required: true } },
      remote_capable: true, local_capable: false, demand: false, human_verbs: ["load", "unload"], idle_release: true,
    },
  ]);

  test("a node-agent target is offered sglang-omni and none of the three local-only kinds", () => {
    expect(kindsFor(REAL_KINDS, true).map((k) => k.kind)).toEqual(["sglang-omni"]);
  });

  test("a local target is offered the three local-only kinds and NOT sglang-omni", () => {
    expect(kindsFor(REAL_KINDS, false).map((k) => k.kind)).toEqual(["lemonade", "comfyui", "hipfire"]);
  });

  // Generic coverage below uses invented kinds (fixture rule, this file's
  // header comment) so the property is proven independent of any real name.
  const MIXED = kindsPayload([
    {
      kind: "local-only", connection: {},
      remote_capable: false, local_capable: true, demand: true, human_verbs: [], idle_release: true,
    },
    {
      kind: "remote-only", connection: {},
      remote_capable: true, local_capable: false, demand: false, human_verbs: [], idle_release: false,
    },
    // A kind capable nowhere (the shape KNOWN_KINDS' own comment,
    // app/engine_kinds.py:175, says a real kind can never be — pinned here
    // as defensive coverage: this function must not accidentally default a
    // missing/false capability to "included").
    {
      kind: "capable-nowhere", connection: {},
      remote_capable: false, local_capable: false, demand: false, human_verbs: [], idle_release: false,
    },
  ]);

  test("isRemote true keeps only remote_capable kinds", () => {
    expect(kindsFor(MIXED, true).map((k) => k.kind)).toEqual(["remote-only"]);
  });

  test("isRemote false keeps only local_capable kinds", () => {
    expect(kindsFor(MIXED, false).map((k) => k.kind)).toEqual(["local-only"]);
  });

  test("a kind capable nowhere is excluded from both directions", () => {
    expect(kindsFor(MIXED, true)).not.toContainEqual(expect.objectContaining({ kind: "capable-nowhere" }));
    expect(kindsFor(MIXED, false)).not.toContainEqual(expect.objectContaining({ kind: "capable-nowhere" }));
  });

  test("an empty kinds payload yields an empty list in either direction, not a crash", () => {
    expect(kindsFor(kindsPayload([]), true)).toEqual([]);
    expect(kindsFor(kindsPayload([]), false)).toEqual([]);
  });

  test("does not mutate its input payload", () => {
    const before = JSON.parse(JSON.stringify(MIXED));
    kindsFor(MIXED, true);
    expect(MIXED).toEqual(before);
  });
});

describe("demandFor", () => {
  test("finds the demand flag for the kind being edited", () => {
    // The form must not look up a kind by name; it asks the catalog it
    // already holds for the picker.
    const catalog = [
      { kind: "lemonade", connection: {}, remote_capable: false, local_capable: true,
        human_verbs: ["load", "unload"], demand: true, idle_release: true },
      // sglang-omni's REAL human_verbs (app/engine_kinds.py's
      // _SglangOmniAdapter.human_verbs, ~line 1012-1016) is ["load",
      // "unload"] — never "free" (that vocabulary belongs to comfyui).
      // idle_release: true — its arbiter_verbs() is frozenset({"unload"})
      // and idle_action has a real rule (:1025-1048).
      { kind: "sglang-omni", connection: {}, remote_capable: true, local_capable: false,
        human_verbs: ["load", "unload"], demand: false, idle_release: true },
    ];
    expect(demandFor(catalog, "lemonade")).toBe(true);
    expect(demandFor(catalog, "sglang-omni")).toBe(false);
    // A kind the catalog does not carry is UNKNOWN, never a guessed false —
    // messages.ttlConsequence renders null as "unknown" on purpose.
    expect(demandFor(catalog, "nope")).toBe(null);
    expect(demandFor(null, "lemonade")).toBe(null);
  });
});

describe("resourceKindMap", () => {
  test("maps a local policy row to its declared kind", () => {
    // PolicyModal is keyed by RESOURCE; `demand`/`idle_release` are per
    // KIND. World.tenants stamps `engine` on every entry regardless of
    // kind (app/state.py's World.snapshot), which is the join.
    const tenants = {
      lemonade: { engine: "lemonade" },
    } as unknown as Record<string, ResourceTenant>;

    expect(resourceKindMap(tenants)).toEqual({ lemonade: "lemonade" });
    expect(resourceKindMap(undefined)).toEqual({});
  });

  test("also folds in REMOTE tenants — a remote-declared engine must not fall to the unknown path", () => {
    // FINDING 1: app/state.py's World builds `world.tenants` from the
    // LOCAL node alone, but PolicyModal's rows are seeded from the WHOLE
    // registry — a remote-declared engine (the live sglang-omni `omni` on
    // sparky, local_capable false: KNOWN_KINDS' own comment says it can
    // NEVER be a local tenant) was joining only against `tenants` and
    // always missed, falling to the `?? ""` path and rendering the false
    // "kind catalog unavailable" text instead of the exact warning this
    // whole feature exists to show.
    //
    // `omni` here is modeled as a REMOTE tenant (world.remote_tenants,
    // keyed `<node>/<resource>` per app/observe.py's node_key) — the
    // shape it can actually appear in, not the impossible local one the
    // old fixture used.
    const tenants = {
      lemonade: { engine: "lemonade" },
    } as unknown as Record<string, ResourceTenant>;
    const remoteTenants = {
      "sparky/omni": {
        engine: "sglang-omni",
        gpu_index: 0,
        state: "idle",
        node_id: "sparky",
        resource: "omni",
      },
    } as unknown as Record<string, RemoteTenant>;

    expect(resourceKindMap(tenants, remoteTenants)).toEqual({
      lemonade: "lemonade",
      omni: "sglang-omni",
    });
  });

  test("a remote tenant with no local tenants at all still joins", () => {
    const remoteTenants = {
      "sparky/omni": { engine: "sglang-omni", gpu_index: 0, state: "idle",
        node_id: "sparky", resource: "omni" },
    } as unknown as Record<string, RemoteTenant>;

    expect(resourceKindMap(undefined, remoteTenants)).toEqual({ omni: "sglang-omni" });
  });

  test("no remote_tenants argument at all still works — the join is additive, not required", () => {
    const tenants = { lemonade: { engine: "lemonade" } } as unknown as Record<string, ResourceTenant>;
    expect(resourceKindMap(tenants, undefined)).toEqual({ lemonade: "lemonade" });
  });
});

describe("idleReleaseFor", () => {
  test("finds the idle_release flag for the kind being edited — mirrors demandFor", () => {
    const catalog = [
      { kind: "lemonade", connection: {}, remote_capable: false, local_capable: true,
        human_verbs: ["load", "unload"], demand: true, idle_release: true },
      // hipfire: real idle_release is FALSE — arbiter_verbs() is empty,
      // idle_action is unconditionally None (app/engine_kinds.py's
      // _HipfireAdapter, ~line 830-844).
      { kind: "hipfire", connection: {}, remote_capable: false, local_capable: true,
        human_verbs: ["park", "resume"], demand: false, idle_release: false },
    ];
    expect(idleReleaseFor(catalog, "lemonade")).toBe(true);
    expect(idleReleaseFor(catalog, "hipfire")).toBe(false);
    // Unknown kind / no catalog: null, not a guessed false — same honest-
    // unknown posture as demandFor.
    expect(idleReleaseFor(catalog, "nope")).toBe(null);
    expect(idleReleaseFor(null, "lemonade")).toBe(null);
  });
});
