import { describe, expect, it } from "vitest";
import type { EngineKindsResponse } from "../api";
import type { RemoteEngineControl } from "./nodes";
import { loadVerbFor, remoteEngineVerbs } from "./engineVerbs";

// Only the fields loadVerbFor reads — nodeId/resource/kind/state are along
// for the ride on the real type, never consulted here.
function control(verbs: RemoteEngineControl["verbs"]): RemoteEngineControl {
  return { nodeId: "zeta", resource: "song-lab", kind: "sglang-omni", state: "down", verbs };
}

// GET /api/engine-kinds's shape (app/routers/nodes.py:493-504). Invented
// kinds ("widget"/"gadget") wherever the test is about the RULE rather than
// about a real kind's vocabulary — this module treats `kind` and `verb` as
// opaque payload data, so a fixture named after a live kind would let a
// hardcoded-name defect pass ([[defaults-that-hide-bugs]]).
function kindsPayload(
  kinds: {
    kind: string;
    connection: Record<string, { required: boolean }>;
    remote_capable: boolean;
    local_capable: boolean;
    demand: boolean;
    human_verbs: string[];
    idle_release: boolean;
    max_gpus: number | null;
    instance: boolean;
    instance_env: Record<string, { required: boolean }>;
  }[],
): EngineKindsResponse {
  return { kinds };
}

const WIDGET = kindsPayload([
  {
    kind: "widget",
    connection: { url: { required: true } },
    remote_capable: true,
    local_capable: false,
    demand: true,
    // Deliberately NOT alphabetical-by-accident-only: the backend serves
    // `sorted(human_verbs())` (app/routers/nodes.py:502), and this module
    // must render the payload's order rather than re-sorting it.
    human_verbs: ["load", "unload"],
    idle_release: true,
    max_gpus: null,
    instance: true,
    instance_env: {},
  },
  {
    kind: "gadget",
    connection: {},
    remote_capable: true,
    local_capable: false,
    demand: false,
    human_verbs: ["polish"],
    idle_release: false,
    max_gpus: null,
    instance: true,
    instance_env: {},
  },
]);

describe("remoteEngineVerbs", () => {
  it("takes the vocabulary from the kind's own human_verbs, never a literal", () => {
    expect(remoteEngineVerbs(WIDGET, "gadget", "idle", false).map((v) => v.verb)).toEqual([
      "polish",
    ]);
    expect(remoteEngineVerbs(WIDGET, "widget", "down", false).map((v) => v.verb)).toEqual([
      "load",
      "unload",
    ]);
  });

  it("offers nothing at all while the catalog has not landed", () => {
    expect(remoteEngineVerbs(null, "widget", "idle", false)).toEqual([]);
  });

  it("offers nothing for a kind the catalog does not list", () => {
    expect(remoteEngineVerbs(WIDGET, "sprocket", "idle", false)).toEqual([]);
  });

  it.each([
    // A RESIDENT engine (app/engine_kinds.py:949-955's `active`: busy|idle
    // are the two resident states) — load would be a no-op.
    ["busy", true, false],
    ["idle", true, false],
    // Not resident: unload is the no-op instead.
    ["down", false, true],
    // "we failed to look" (app/engine_kinds.py:931-947's `unknown()`) is
    // proof of NEITHER, so neither verb is withheld.
    ["unknown", false, false],
  ])("state %s disables load=%s unload=%s", (state, loadOff, unloadOff) => {
    const verbs = remoteEngineVerbs(WIDGET, "widget", state, false);
    expect(verbs.find((v) => v.verb === "load")!.disabled).toBe(loadOff);
    expect(verbs.find((v) => v.verb === "unload")!.disabled).toBe(unloadOff);
  });

  it("disables every verb when the owning node is unreachable", () => {
    // Even the verb that would otherwise be live — nothing can act on a
    // memory (ResourcePanel's own `stale` rule for local controls).
    expect(remoteEngineVerbs(WIDGET, "widget", "down", true)).toEqual([
      { verb: "load", disabled: true },
      { verb: "unload", disabled: true },
    ]);
  });

  it("a verb this module has no no-op rule for is offered as the kind declares it", () => {
    // "polish" is neither of the two the node-agent engine channel carries
    // (app/routers/serving.py:228's _REMOTE_VERBS). The BACKEND refuses it
    // (405/501, engine_verb); this module never silently withholds a verb a
    // kind declares.
    expect(remoteEngineVerbs(WIDGET, "gadget", "busy", false)).toEqual([
      { verb: "polish", disabled: false },
    ]);
  });

  it("reads the live sglang-omni vocabulary verbatim", () => {
    // The one pinned real-kind payload: app/engine_kinds.py:966-970's
    // human_verbs() is exactly {"load", "unload"} — never park/resume.
    const live = kindsPayload([
      {
        kind: "sglang-omni",
        connection: { url: { required: true } },
        remote_capable: true,
        local_capable: false,
        demand: false,
        human_verbs: ["load", "unload"],
        idle_release: true,
        max_gpus: 1,
        instance: false,
        instance_env: {},
      },
    ]);
    expect(remoteEngineVerbs(live, "sglang-omni", "busy", false)).toEqual([
      { verb: "load", disabled: true },
      { verb: "unload", disabled: false },
    ]);
  });
});

describe("loadVerbFor", () => {
  it("returns the load entry when the control's verbs carry one", () => {
    expect(
      loadVerbFor(control([{ verb: "load", disabled: false }, { verb: "unload", disabled: true }])),
    ).toEqual({ verb: "load", disabled: false });
  });

  it("returns the (disabled) load entry unchanged — this fn re-derives nothing", () => {
    expect(loadVerbFor(control([{ verb: "load", disabled: true }]))).toEqual({
      verb: "load",
      disabled: true,
    });
  });

  it("returns null for a kind whose verbs don't include load", () => {
    // e.g. "gadget" in the fixture above, whose only human_verb is "polish".
    expect(loadVerbFor(control([{ verb: "polish", disabled: false }]))).toBeNull();
  });

  it("returns null when verbs is empty — the catalog hasn't landed", () => {
    expect(loadVerbFor(control([]))).toBeNull();
  });
});
