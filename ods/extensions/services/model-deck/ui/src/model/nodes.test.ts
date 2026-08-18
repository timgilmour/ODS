import { describe, expect, it } from "vitest";
import type {
  DeckNodeEntry,
  EngineKindsResponse,
  Gpu,
  LifecycleEntry,
  PolicyMap,
  RemoteTenant,
  ResourceTenant,
  SettingsDrift,
  SparkStatus,
  StateResponse,
} from "../api";
import { remoteEngineVerbs } from "./engineVerbs";
import {
  buildNodes,
  controlHasPlacement,
  findPlacement,
  isSwapSlotId,
  nodeIdOfPlacement,
  swapNodes,
  type DeckNode,
  type DeckResource,
  type Placement,
} from "./nodes";

// Fixture rule (design §7, binding): generalization fixtures set AWAY from
// today's live topology — resource names gguf-a/img/agent (never
// lemonade/comfyui/hipfire), GPUs 2/3 (never 0/1). A same-shape fixture
// (three resources named after their kinds, on GPUs 0/1) cannot catch a
// resource-vs-kind-name confusion, because the two coincide by construction
// ([[defaults-that-hide-bugs]]).

const DEFAULT_GPUS: Gpu[] = [
  { index: 2, total: 34_000_000_000, used: 22_100_000_000, free: 11_900_000_000 },
  { index: 3, total: 34_000_000_000, used: 200_000_000, free: 33_800_000_000 },
];

// One of each kind: "agent" (hipfire-kind, GPU 2), "gguf-a" (lemonade-kind,
// GPU 3), "img" (comfyui-kind, GPU 3, sharing gguf-a's GPU on purpose — a
// co-residency case today's live box doesn't have).
const DEFAULT_TENANTS: Record<string, ResourceTenant> = {
  agent: {
    engine: "hipfire", gpu_index: 2, state: "running",
    model: "Qwen3.6-35B-A3B-heretic-NVFP4", footprint: 21_400_000_000, queue_depth: 0,
  },
  "gguf-a": {
    engine: "lemonade", gpu_index: 3, state: "unloaded",
    model: null, footprint: null, idle_s: null,
  },
  img: { engine: "comfyui", gpu_index: 3, state: "idle", queue: 0, idle_s: 12 },
};

const DEFAULT_POLICY: PolicyMap = {
  "gguf-a": { priority: 1, pinned: false, idle_ttl: 900 },
  img: { priority: 2, pinned: false, idle_ttl: 300 },
  agent: { priority: 3, pinned: true, idle_ttl: 0 },
};

const GENERIC_POLICY = { priority: 0, pinned: false, idle_ttl: 0 };

function stateWith(
  overrides: Partial<{
    tenants: Record<string, ResourceTenant>;
    gpus: Gpu[];
    externals: StateResponse["world"]["externals"];
    policy: PolicyMap;
    lifecycle: StateResponse["lifecycle"];
    nodes: DeckNodeEntry[];
    /** `world.remote_tenants` — the REMOTE half of the world snapshot
     * (app/state.py's World.snapshot_remote, merged in by
     * app/routers/__init__.py's build_world_snapshot). OMITTED entirely
     * unless a test asks for it, so every other test in this file exercises
     * the absent-key path a deck with no remote observer serves. */
    remoteTenants: Record<string, RemoteTenant>;
  }> = {},
): StateResponse {
  // Shallow copy: tests reassign a whole per-resource entry (never mutate
  // one in place), but without this every call sharing the default would
  // alias the SAME object — an earlier test's `s.world.tenants.x = {...}`
  // would silently leak into a later test's default fixture.
  const tenants = overrides.tenants ?? { ...DEFAULT_TENANTS };
  return {
    node: { id: "local", label: "autarch" },
    world: {
      gpus: overrides.gpus ?? DEFAULT_GPUS,
      tenants,
      externals: overrides.externals ?? [],
      default_route: null,
      // Redundant with each tenant's own gpu_index (see api.ts's World.placement
      // doc) — derived here so a fixture never has to state the same fact twice.
      placement: Object.fromEntries(Object.entries(tenants).map(([r, t]) => [r, t.gpu_index])),
      ...(overrides.remoteTenants ? { remote_tenants: overrides.remoteTenants } : {}),
    },
    // Every declared resource needs a policy row or tenantPlacement's
    // `policy[resource]` throws — known resources get their named default
    // above, an unlisted one (a test's own bespoke resource name) gets a
    // harmless generic row, since most tests exercising a custom name don't
    // care about its policy.
    policy: overrides.policy ?? Object.fromEntries(
      Object.keys(tenants).map((r) => [r, DEFAULT_POLICY[r] ?? GENERIC_POLICY]),
    ),
    models: [],
    lifecycle: overrides.lifecycle ?? {},
    nodes: overrides.nodes,
  };
}

/** The board is per-GPU: one card per GPU index, id `gpu<index>`. */
function gpuCard(node: DeckNode, index: number): DeckResource {
  return node.resources.find((r) => r.id === `gpu${index}`)!;
}

/** A local resource's own chip, wherever its GPU card is. Placement ids are
 * UNCHANGED by the per-GPU regroup (`local/<resource>` — app/observe.py's
 * node_key), which is exactly what makes this lookup possible: the lifecycle
 * join, the drawer and drift targeting all key on the same string. */
function localChip(node: DeckNode, resource: string): Placement | undefined {
  return node.resources.flatMap((r) => r.placements).find((p) => p.id === `local/${resource}`);
}

describe("buildNodes", () => {
  it("returns nothing before the first successful poll", () => {
    expect(buildNodes(null, {})).toEqual([]);
  });

  it("names the local node from the backend, not a hardcoded string", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(local.id).toBe("local");
    expect(local.label).toBe("autarch");
    expect(local.status).toBe("reachable");
  });

  it("local card renders one card per GPU, carrying every declared resource's controls", () => {
    // Reworked from "renders every declared resource, none hardcoded" (E1's
    // per-resource cards): the board is per-GPU now, so the "none hardcoded"
    // guarantee moved onto `controls` — every declared resource still
    // appears, on the card for the GPU it is declared on.
    const nodes = buildNodes(
      stateWith({
        tenants: {
          "gguf-a": {
            engine: "lemonade", gpu_index: 2, state: "loaded",
            model: "a.gguf", footprint: 10, idle_s: 0,
          },
          img: { engine: "comfyui", gpu_index: 3, state: "idle", queue: 0, idle_s: 5 },
        },
        policy: {
          "gguf-a": { priority: 0, pinned: false, idle_ttl: 0 },
          img: { priority: 0, pinned: false, idle_ttl: 0 },
        },
      }),
      {},
    );
    const local = nodes.find((n) => n.id === "local")!;
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(gpuCard(local, 2).controls).toEqual(["gguf-a"]);
    expect(gpuCard(local, 3).controls).toEqual(["img"]);
  });

  it("declaring nothing still renders the box's GPUs — they are the telemetry surface", () => {
    // Reworked: E1's per-resource board rendered NO cards for an empty
    // declaration. A GPU card always shows now (design ruling 2: "GPU cards
    // ALWAYS show, even empty"), carrying its meter and no controls.
    const nodes = buildNodes(stateWith({ tenants: {}, policy: {} }), {});
    const local = nodes.find((n) => n.id === "local")!;
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(local.resources.every((r) => r.controls.length === 0)).toBe(true);
    expect(local.resources.every((r) => r.placements.length === 0)).toBe(true);
  });

  it("renders nothing at all when the box reports no GPUs and declares nothing", () => {
    const nodes = buildNodes(stateWith({ tenants: {}, policy: {}, gpus: [] }), {});
    expect(nodes.find((n) => n.id === "local")!.resources).toEqual([]);
  });

  it("makes one card per GPU, ordered by index, with each GPU's declared resources in board order", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(local.resources[0].label).toBe("GPU 2");
    expect(local.resources[0].capacity).toEqual({ used: 22_100_000_000, total: 34_000_000_000 });
    expect(gpuCard(local, 2).controls).toEqual(["agent"]);
    // Two resources share GPU 3 — sortedResourceEntries' order (gpu_index
    // then resource NAME) is what orders the controls within a card.
    expect(gpuCard(local, 3).controls).toEqual(["gguf-a", "img"]);
  });

  it("still cards a declared resource's GPU when the live GPU list has no such index, capacity unknown", () => {
    // Controls must never vanish because of a world/declaration
    // inconsistency: the card appears, its meter reads unknown (not a
    // fabricated 0/0), and "agent" keeps its verbs.
    const s = stateWith({ gpus: [DEFAULT_GPUS[1]] }); // GPU 2 (agent's) dropped
    const [local] = buildNodes(s, {});
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(gpuCard(local, 2).capacity).toBeNull();
    expect(gpuCard(local, 2).controls).toEqual(["agent"]);
  });

  it("carries as many declared resources as exist onto their GPUs' cards — §5's five, on two GPUs", () => {
    const tenants: Record<string, ResourceTenant> = {
      "gguf-a": { engine: "lemonade", gpu_index: 2, state: "unloaded", model: null, footprint: null, idle_s: null },
      "gguf-b": { engine: "lemonade", gpu_index: 3, state: "unloaded", model: null, footprint: null, idle_s: null },
      img: { engine: "comfyui", gpu_index: 2, state: "idle", queue: 0, idle_s: 0 },
      agent: { engine: "hipfire", gpu_index: 3, state: "parked", model: null, footprint: 0, queue_depth: null },
      agent2: { engine: "hipfire", gpu_index: 3, state: "parked", model: null, footprint: 0, queue_depth: null },
    };
    const [local] = buildNodes(
      stateWith({
        tenants,
        policy: Object.fromEntries(Object.keys(tenants).map((r) => [r, GENERIC_POLICY])),
      }),
      {},
    );
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(gpuCard(local, 2).controls).toEqual(["gguf-a", "img"]);
    expect(gpuCard(local, 3).controls).toEqual(["agent", "agent2", "gguf-b"]);
  });

  it("shows the resource's placement when it is occupying its slot", () => {
    const [local] = buildNodes(stateWith(), {});
    const agent = gpuCard(local, 2);
    expect(agent.placements.map((p) => p.name)).toEqual(["Qwen3.6-35B-A3B-heretic-NVFP4"]);
    expect(agent.placements[0].engine).toBe("hipfire");
    expect(agent.placements[0].bytes).toBe(21_400_000_000);
  });

  it("keeps a model identity verbatim", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(localChip(local, "agent")!.name).toBe("Qwen3.6-35B-A3B-heretic-NVFP4");
  });

  it("omits an unloaded tenant rather than showing an empty chip", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(localChip(local, "gguf-a")).toBeUndefined();
  });

  it("omits a load in flight the same as unloaded (no chip mid-load)", () => {
    // tenantPlacement's `state !== "loaded"` guard also covers "loading" —
    // deliberate: the PlacementActions pill (STATE_TONE) is where a load in
    // flight shows up today; a dedicated chip treatment is deferred.
    const s = stateWith();
    s.world.tenants["gguf-a"] = {
      engine: "lemonade", gpu_index: 3, state: "loading", model: null, footprint: null, idle_s: null,
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "gguf-a")).toBeUndefined();
  });

  it("keeps an unloaded tenant's control on its GPU's card", () => {
    // The empty-slot case: no chip, but a load-verb kind's Load dropdown
    // still has to render somewhere, so the GPU card carries the control.
    const [local] = buildNodes(stateWith(), {});
    expect(gpuCard(local, 3).controls).toContain("gguf-a");
    expect(gpuCard(local, 2).controls).toEqual(["agent"]);
  });

  it("shows a loaded tenant", () => {
    const s = stateWith();
    s.world.tenants["gguf-a"] = {
      engine: "lemonade", gpu_index: 3, state: "loaded",
      model: "qwen2.5-14b-instruct-4k-q4_k_m.gguf", footprint: 9_500_000_000, idle_s: 4,
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "gguf-a")!.name).toBe("qwen2.5-14b-instruct-4k-q4_k_m.gguf");
  });

  it("omits a parked hipfire-kind resource", () => {
    const s = stateWith();
    s.world.tenants.agent = {
      engine: "hipfire", gpu_index: 2, state: "parked", model: null, footprint: 0, queue_depth: null,
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "agent")).toBeUndefined();
    // The card and its control survive the empty chip — that empty space is
    // where the serving-slot dropzone shows.
    expect(gpuCard(local, 2).controls).toEqual(["agent"]);
  });

  it("surfaces an external process as its own placement", () => {
    const s = stateWith();
    s.world.externals = [{ pid: 4242, gpu: 2, bytes: 1_200_000_000 }];
    const [local] = buildNodes(s, {});
    const external = gpuCard(local, 2).placements.find((p) => p.kind === "external");
    expect(external?.bytes).toBe(1_200_000_000);
    expect(external?.status).toBe("unmanaged");
  });

  it("attaches a shared GPU's external to that GPU's card exactly once", () => {
    // Reworked from "attributes a shared GPU's external to only the FIRST
    // resource card on it": gguf-a and img share GPU 3, and with one card
    // per GPU there is no longer a set of co-located cards to double-count
    // across — the first-card-claim bookkeeping is gone, and the external
    // simply belongs to its GPU.
    const s = stateWith();
    s.world.externals = [{ pid: 99, gpu: 3, bytes: 2_000_000_000 }];
    const [local] = buildNodes(s, {});
    const externals = local.resources.flatMap((r) =>
      r.placements.filter((p) => p.kind === "external").map((p) => [r.id, p.id]));
    expect(externals).toEqual([["gpu3", "external/99"]]);
  });

  it("carries each tenant's policy onto its placement", () => {
    // The board's 📌 and P{n}. They come from state.policy, which buildNodes
    // already has in hand — the alternative is drilling a `policy` prop back
    // down to every card, which is the coupling this adapter exists to end.
    const s = stateWith();
    s.world.tenants["gguf-a"] = {
      engine: "lemonade", gpu_index: 3, state: "loaded", model: "qwen.gguf",
      footprint: 9_000_000_000, idle_s: 4,
    };
    const [local] = buildNodes(s, {});
    const agent = localChip(local, "agent")!;
    const gguf = localChip(local, "gguf-a");

    expect(agent.pinned).toBe(true);
    expect(agent.priority).toBe(3);
    expect(gguf?.pinned).toBe(false);
    expect(gguf?.priority).toBe(1);
  });

  it("marks hipfire busy when a turn is in flight, and not otherwise", () => {
    // Predicts the park refusal BEFORE the click: hipfire's single admission
    // slot is what makes park/apply 409 without force.
    const idle = localChip(buildNodes(stateWith(), {})[0], "agent")!;
    expect(idle.busy).toBe(false);

    const s = stateWith();
    s.world.tenants.agent = { ...s.world.tenants.agent, queue_depth: 2 };
    const busy = localChip(buildNodes(s, {})[0], "agent")!;
    expect(busy.busy).toBe(true);
  });

  it("treats a missing hipfire queue reading as not busy", () => {
    const s = stateWith();
    s.world.tenants.agent = { ...s.world.tenants.agent, queue_depth: null };
    const placement = localChip(buildNodes(s, {})[0], "agent")!;
    expect(placement.busy).toBe(false);
  });

  it("carries ComfyUI's queue and idle time, and gives hipfire neither", () => {
    const s = stateWith();
    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: 3, idle_s: 0 };
    const [local] = buildNodes(s, {});
    const img = localChip(local, "img");
    const agent = localChip(local, "agent")!;

    expect(img?.queue).toBe(3);
    expect(img?.idleSeconds).toBe(0);
    // Absent, not zero: hipfire has no queue and reports no idle time, and
    // "0" on screen would be a claim neither engine ever made.
    expect(agent.queue).toBeUndefined();
    expect(agent.idleSeconds).toBeUndefined();
  });

  it("keeps a drained ComfyUI queue distinct from an unreadable one", () => {
    // Chip visibility (Task 2 ruling) is forced via `state: "busy"` in both
    // cases, so what is under test stays the QUEUE field's own distinction
    // (0 vs null) rather than whether the chip shows at all — that is
    // chipVisibility.test.ts's job.
    const s = stateWith();
    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: 0, idle_s: 12 };
    const zero = localChip(buildNodes(s, {})[0], "img");
    expect(zero?.queue).toBe(0);

    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: null, idle_s: null };
    const unknown = localChip(buildNodes(s, {})[0], "img");
    expect(unknown?.queue).toBeNull();
  });

  it("names a non-model engine placement after its RESOURCE, not a hardcoded kind literal", () => {
    // Two comfyui-kind resources would otherwise both render "comfyui" —
    // the resource name is what tells them apart on screen. `state: "busy"`
    // (rather than the default fixture's idle/queue-0) keeps the chip
    // visible under the Task 2 ruling — this test is about the NAME, not
    // visibility.
    const s = stateWith();
    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: 3, idle_s: 12 };
    const [local] = buildNodes(s, {});
    const img = localChip(local, "img");
    expect(img?.name).toBe("img");
  });

  it("badges every local model chip with its own engine name, unconditionally", () => {
    // Superseded ruling (2026-08-18): the badge used to answer "what is this
    // node running that it usually isn't", so a local resource — always its
    // own declared kind — never carried one. It is unconditional now: every
    // model chip names its engine, local or remote alike, so nothing on the
    // board depends on tribal knowledge of "the usual engine" to read what a
    // chip is running.
    const s = stateWith();
    s.world.tenants["gguf-a"] = {
      engine: "lemonade", gpu_index: 3, state: "loaded", model: "qwen.gguf",
      footprint: 9_000_000_000, idle_s: 4,
    };
    const [local] = buildNodes(s, {});
    const agent = localChip(local, "agent")!;
    const gguf = localChip(local, "gguf-a")!;
    expect(agent.engineBadge).toBe("hipfire");
    expect(gguf.engineBadge).toBe("lemonade");
  });

  it("takes a placement's status from the lifecycle view", () => {
    const s = stateWith();
    s.lifecycle = {
      "local/agent": {
        status: "drifted",
        reason: "settings changed",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: null,
      },
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "agent")!.status).toBe("drifted");
  });

  it("carries settings drift onto a placement from the lifecycle view", () => {
    // app/routers/__init__.py:116 writes `settings_drift` onto every
    // lifecycle entry; this is the one place that value crosses into the
    // board's own Placement shape.
    const drift: SettingsDrift = {
      changed: ["args:max-model-len"],
      entries: [
        { key: "args:max-model-len", old: "262144", new: "131072", ts: "2026-08-07T00:00:00Z" },
      ],
      since: "2026-08-07T00:00:00Z",
    };
    const s = stateWith();
    s.lifecycle = {
      "local/agent": {
        status: "drifted",
        reason: "settings changed",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: drift,
      },
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "agent")!.settingsDrift).toEqual(drift);
  });

  it("leaves settingsDrift undefined when the lifecycle entry carries none", () => {
    const s = stateWith();
    s.lifecycle = {
      "local/agent": {
        status: "serving",
        reason: "",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: null,
      },
    };
    const [local] = buildNodes(s, {});
    expect(localChip(local, "agent")!.settingsDrift).toBeUndefined();
  });
});

function sparkStatus(overrides: Partial<SparkStatus> = {}): SparkStatus {
  return {
    profiles: [{ name: "heretic", engine: "vllm", health_url: null, container: "spark-heretic" }],
    swap_status: null,
    serving: { model: "heretic", endpoint_ok: true, container_status: "running" },
    ...overrides,
  };
}

function lifecycleEntry(overrides: Partial<LifecycleEntry> = {}): LifecycleEntry {
  return {
    status: "serving",
    reason: "",
    intent: null,
    settings_drift: null,
    observed: { reachable: true, loaded: true, model: "heretic", transitioning: false },
    last_healthy_ts: "2026-08-04T14:23:11Z",
    ...overrides,
  };
}

// The registry's own entry for the local box — always present in
// state.nodes (app/routers/status.py's _nodes_block), always agent_kind
// "local", always skipped by buildNodes' registry loop (which only turns
// "node-agent" entries into cards). Included in these fixtures because the
// real payload always carries it alongside the remote entries. `control` is
// carried on the wire regardless of agent_kind (app/node_store.py's
// `_heal_control`), even though the local loop never reads it.
const localEntry: DeckNodeEntry = {
  id: "local", label: "autarch", agent_kind: "local",
  address: null, serving_address: null, credential_set: false, control: "none",
  status: "online", last_seen: null, gpus: null, serving: null, error: null,
};

// Labels deliberately ≠ ids: a fixture whose label equals its id cannot
// catch a label used as a key (the node_label class of defect). control
// "none" — the observe-only case, whether or not it happens to carry
// serving data (see the "watcher-shaped" test below).
const heraEntry: DeckNodeEntry = {
  id: "hera", label: "Hera Box", agent_kind: "node-agent",
  address: "http://hera:7720", serving_address: null, credential_set: true,
  control: "none",
  status: "online", last_seen: "2026-08-10T00:00:00+00:00",
  gpus: [{ index: 0, name: "RTX", memory_used_mb: 1024, memory_total_mb: 24576,
           utilization_percent: 5 }],
  serving: { model: "big-model", endpoint_ok: true }, error: null,
};

// Two swap-control entries — deliberately different ids/labels so a test can
// tell them apart on screen, the same reasoning as heraEntry's label choice.
// `gpus` carries a reading (unlike heraEntry's) because a swap node is a
// node-agent too — the node-observer probes it the same way — so its
// observe-only fallback (buildSwapNode returning null) has a real resource
// to render rather than an empty shell.
const boxaEntry: DeckNodeEntry = {
  id: "boxa", label: "Box Alpha", agent_kind: "node-agent",
  address: "http://boxa:7720", serving_address: "http://boxa:8000",
  credential_set: true, control: "swap",
  status: "online", last_seen: "2026-08-10T00:00:00+00:00",
  gpus: [{ index: 0, name: "RTX 6000", memory_used_mb: 4096, memory_total_mb: 49152,
           utilization_percent: 10 }],
  serving: null, error: null,
};

const boxbEntry: DeckNodeEntry = {
  id: "boxb", label: "Box Beta", agent_kind: "node-agent",
  address: "http://boxb:7720", serving_address: "http://boxb:8000",
  credential_set: true, control: "swap",
  status: "online", last_seen: "2026-08-10T00:00:00+00:00",
  gpus: [{ index: 0, name: "RTX 6000", memory_used_mb: 2048, memory_total_mb: 49152,
           utilization_percent: 5 }],
  serving: null, error: null,
};

describe("buildNodes — swap nodes", () => {
  it("makes one card per swap node, each with its own placement id and label from the registry", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry, boxbEntry] });
    const nodes = buildNodes(s, {
      boxa: sparkStatus(),
      boxb: sparkStatus({
        profiles: [{ name: "ds4", engine: "ds4", health_url: null, container: "spark-ds4" }],
        serving: { model: "ds4", endpoint_ok: true, container_status: "running" },
      }),
    });

    const a = nodes.find((n) => n.id === "boxa")!;
    const b = nodes.find((n) => n.id === "boxb")!;
    expect(a.label).toBe("Box Alpha");
    expect(b.label).toBe("Box Beta");
    // The happy-path status derivation (endpoint_ok -> "reachable") — lost
    // when this test replaced the old single-spark "adds a single
    // serving-slot resource when configured" test; a broken
    // `else if (endpointOk) status = "reachable"` must fail this suite.
    expect(a.status).toBe("reachable");
    // Reworked: the slot is no longer a card of its own. A swap node renders
    // one card per GPU it reports (boxa reports one), and the serving slot's
    // chip and profile picker ride the FIRST of them — the sparky collapse
    // this wave exists for. The placement ids below are unchanged by that.
    expect(a.resources.map((r) => r.id)).toEqual(["gpu0"]);
    expect(a.resources.map((r) => r.label)).toEqual(["GPU 0"]);
    expect(a.resources[0].controls).toEqual(["spark"]);
    expect(a.resources[0].placements[0].id).toBe("boxa/slot0");
    expect(b.resources[0].placements[0].id).toBe("boxb/slot0");
    expect(a.resources[0].placements[0].name).toBe("heretic");
    expect(b.resources[0].placements[0].name).toBe("ds4");
    // Neither node's serving status leaks into the other's card.
    expect(a.resources[0].placements).toHaveLength(1);
    expect(b.resources[0].placements).toHaveLength(1);
  });

  it("meters the swap node's own reported GPU, and reports unknown rather than zero when it reports none", () => {
    // Reworked: the old serving-slot card had no meter at all ("Spark reports
    // no VRAM figures to the deck"), because it was not a GPU card. It IS one
    // now — the node-agent's own reading (MB -> bytes) — and the unknown case
    // is the node that reported no GPUs at all.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].capacity).toEqual({
      used: 4096 * 1024 * 1024, total: 49152 * 1024 * 1024 });

    const blind = stateWith({ nodes: [localEntry, { ...boxaEntry, gpus: null }] });
    const dark = buildNodes(blind, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(dark.resources).toHaveLength(1);
    expect(dark.resources[0].capacity).toBeNull();
  });

  it("control decides, data never does: a control:'none' entry never gets swap controls, even carrying serving data and a servingByNode entry", () => {
    // The watcher-shaped fixture nodes.ts's swapNodes docstring points at:
    // heraEntry already reports `serving` on the wire and is handed a real
    // SparkStatus here too, and still must render observe-only — presence of
    // serving data must never promote a node, only the DECLARED control does.
    const s = stateWith({ nodes: [localEntry, heraEntry] });
    const hera = buildNodes(s, { hera: sparkStatus() }).find((n) => n.id === "hera")!;
    expect(hera.resources[0].controls).toEqual([]);
    expect(hera.resources[0].placements).toEqual([]);
  });

  it("falls back to the observe-only card when no status has landed yet, never a phantom control surface", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: null }).find((n) => n.id === "boxa")!;
    expect(boxa.label).toBe("Box Alpha"); // the registry label, same as the swap card would show
    expect(boxa.resources[0].controls).toEqual([]);
    expect(boxa.resources[0].placements).toEqual([]);
  });

  it("also falls back to observe-only when the id is simply absent from servingByNode (fetch not landed)", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {}).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].controls).toEqual([]);
  });

  it("retains last-known placements when unreachable, and marks them stale", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "unreachable",
        observed: { reachable: false, loaded: false, model: null, transitioning: false },
      }),
    };
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;

    expect(boxa.status).toBe("unreachable");
    expect(boxa.lastSeen).toBe("2026-08-04T14:23:11Z");
    // The whole point: an offline node must never blank out.
    expect(boxa.resources[0].placements[0].name).toBe("heretic");
    expect(boxa.resources[0].placements[0].stale).toBe(true);
  });

  it("carries the PROFILE for settings identity and the served name for display", () => {
    // AWAY FROM THE DEFAULT FIXTURE, deliberately: sparkStatus() serves
    // "heretic" from a profile also named "heretic", so served-name and
    // profile coincide and nothing can tell a profile-keyed lookup from a
    // name-keyed one. The real mm27b/aeon case does not coincide — the
    // node-agent reports models[0].id "aeon" while the profile is "mm27b".
    //
    // The identity vocabulary is the PROFILE: app/routers/settings.py:293
    // writes identities[meta["name"]] (a profiles[] entry's name), and
    // app/observe.py:180-184 says so outright — "Identity is the PROFILE the
    // node last swapped to, not the served model name ... comparing served
    // names would report permanent drift for a correct placement".
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({
        // engine "ds4" on a profile named mm27b is a deliberately
        // NON-default engine, chosen so the join is observable — the real
        // mm27b is a vllm profile, so read this as "a profile whose engine
        // differs from SPARK_DEFAULT_ENGINE", not as live topology.
        profiles: [{ name: "mm27b", engine: "ds4", health_url: null, container: "spark-mm27b" }],
        serving: { model: "aeon", endpoint_ok: true, container_status: "running" },
        swap_status: {
          state: "done", profile: "mm27b", id: "1",
          message: "", ts: "2026-08-05T00:00:00Z",
        },
      }),
    }).find((n) => n.id === "boxa")!;
    const placement = boxa.resources[0].placements[0];

    expect(placement.name).toBe("aeon");       // the chip shows what is served
    expect(placement.profile).toBe("mm27b");   // settings/facts key vocabulary
    // Joined by PROFILE, not served name: keyed on "aeon" this finds no
    // profiles[] entry and silently falls back to the default engine.
    expect(placement.engine).toBe("ds4");
  });

  it("leaves profile undefined when the node has never reported a swap", () => {
    // swap_status null is a real state (a node up since before the deck
    // watched it). Callers fall back to placement.name, so this must be
    // absent rather than an empty string.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements[0].profile).toBeUndefined();
  });

  it("reads as warming while a swap is in flight and the endpoint is down", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({
        serving: { model: "heretic", endpoint_ok: false, container_status: "running" },
        swap_status: {
          state: "swapping", profile: "heretic", id: "1",
          message: "", ts: "2026-08-05T00:00:00Z",
        },
      }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("warming");
  });

  it("reads as warming through the long tail of a boot the helper already calls done", () => {
    // The case a real boot spends nearly all its time in, and the one the
    // swap_status-only rule got wrong. The helper reports "done" as soon as
    // swap.sh launches (app/engines/spark.py, _BOOTING_STATES), so the 5-15
    // minutes of weight load and autotune look like this: state "done",
    // endpoint still down. app/observe.py's boot_in_flight() is what knows
    // better, and it arrives here as observed.transitioning.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "warming",
        reason: "a load or boot is in flight",
        observed: { reachable: true, loaded: false, model: null, transitioning: true },
      }),
    };
    const boxa = buildNodes(s, {
      boxa: sparkStatus({
        serving: { model: "heretic", endpoint_ok: false, container_status: "running" },
        swap_status: {
          state: "done", profile: "heretic", id: "1",
          message: "", ts: "2026-08-05T00:00:00Z",
        },
      }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("warming");
  });

  it("is down, not warming, when the endpoint is dead with no swap running", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: "heretic", endpoint_ok: false, container_status: "exited" } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("down");
  });

  it("is unreachable when the lifecycle view says so, even with a healthy-looking serving payload", () => {
    // Status derivation order: `!reachable` wins over everything else.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "unreachable",
        observed: { reachable: false, loaded: false, model: null, transitioning: false },
      }),
    };
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: "heretic", endpoint_ok: true, container_status: "running" } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("unreachable");
  });

  it("badges the spark placement with its own engine name, unconditionally, default engine included", () => {
    // Superseded ruling (2026-08-18): the badge used to stay silent for
    // SPARK_DEFAULT_ENGINE ("vllm" is unremarkable) and only appear for a
    // non-default engine like "ds4". It is unconditional now — see the
    // local-tenant equivalent above.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const vllm = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(vllm.resources[0].placements[0].engine).toBe("vllm");
    expect(vllm.resources[0].placements[0].engineBadge).toBe("vllm");

    const ds4 = buildNodes(s, {
      boxa: sparkStatus({
        profiles: [{ name: "ds4", engine: "ds4", health_url: null, container: "spark-ds4" }],
        serving: { model: "ds4", endpoint_ok: true, container_status: "running" },
      }),
    }).find((n) => n.id === "boxa")!;
    expect(ds4.resources[0].placements[0].engineBadge).toBe("ds4");
  });

  it("carries the swap helper's error text as the node's detail", () => {
    // The failed-swap case: without this the board shows a red pill and
    // nothing anywhere says the helper died.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({
        serving: { model: null, endpoint_ok: false, container_status: "exited" },
        swap_status: {
          state: "error", profile: "ornith", id: "7",
          message: "swap-helper: container spark-ornith exited (1)",
          ts: "2026-08-05T00:00:00Z",
        },
      }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("down");
    expect(boxa.detail).toBe("swap-helper: container spark-ornith exited (1)");
  });

  it("falls back to lifecycle's own reason when no swap failed", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "down",
        reason: "intended 'heretic' is not loaded",
        observed: { reachable: true, loaded: false, model: null, transitioning: false },
      }),
    };
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: null, endpoint_ok: false, container_status: "exited" } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("down");
    expect(boxa.detail).toBe("intended 'heretic' is not loaded");
  });

  it("prefers a real reason over an error swap with an empty message", () => {
    // A blank explanation is worse than the generic one: it renders as a
    // banner trailing an em-dash into nothing.
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "down",
        reason: "intended 'heretic' is not loaded",
        observed: { reachable: true, loaded: false, model: null, transitioning: false },
      }),
    };
    const boxa = buildNodes(s, {
      boxa: sparkStatus({
        serving: { model: null, endpoint_ok: false, container_status: "exited" },
        swap_status: {
          state: "error", profile: "ornith", id: "7", message: "",
          ts: "2026-08-05T00:00:00Z",
        },
      }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.detail).toBe("intended 'heretic' is not loaded");
  });

  it("leaves detail absent when the backend offered no reason at all", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: null, endpoint_ok: false, container_status: "exited" } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.detail).toBeUndefined();
  });

  it("carries settings drift onto the swap slot too", () => {
    // The reload verb is gated on isSwapSlotId(placement.id), so drift has
    // to reach exactly this placement, not just local tenants.
    const drift: SettingsDrift = {
      changed: ["args:served-model-name"],
      entries: [],
      since: "2026-08-07T00:00:00Z",
    };
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({ settings_drift: drift }),
    };
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements[0].settingsDrift).toEqual(drift);
  });

  it("has an empty slot when nothing is serving", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: null, endpoint_ok: false, container_status: null } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements).toEqual([]);
  });
});

describe("isSwapSlotId", () => {
  it("recognizes a swap node's synthesized slot key, regardless of node id", () => {
    expect(isSwapSlotId("boxa/slot0")).toBe(true);
    expect(isSwapSlotId("boxb/slot0")).toBe(true);
  });

  it("rejects a local tenant placement id", () => {
    expect(isSwapSlotId("local/agent")).toBe(false);
  });

  it("rejects an external placement id", () => {
    expect(isSwapSlotId("external/4242")).toBe(false);
  });
});

describe("nodeIdOfPlacement", () => {
  it("takes the node id off a swap slot key", () => {
    expect(nodeIdOfPlacement("boxa/slot0")).toBe("boxa");
  });

  it("takes the node id off a local tenant placement id the same way", () => {
    expect(nodeIdOfPlacement("local/agent")).toBe("local");
  });
});

describe("swapNodes", () => {
  it("returns only the control:swap entries, declared not inferred", () => {
    const s = stateWith({ nodes: [localEntry, heraEntry, boxaEntry, boxbEntry] });
    expect(swapNodes(s).map((e) => e.id)).toEqual(["boxa", "boxb"]);
  });

  it("returns nothing before state has loaded", () => {
    expect(swapNodes(null)).toEqual([]);
  });

  it("returns nothing when the registry carries no swap nodes", () => {
    expect(swapNodes(stateWith({ nodes: [localEntry, heraEntry] }))).toEqual([]);
  });
});

describe("findPlacement", () => {
  const nodes = buildNodes(stateWith({ nodes: [localEntry, boxaEntry] }), { boxa: sparkStatus() });

  it("finds a local tenant placement and hands back the resource that carries it", () => {
    // The drawer needs the RESOURCE too: its `controls` are what decide
    // which verbs (if any) that placement gets.
    const spot = findPlacement(nodes, "local/agent");
    expect(spot?.node.id).toBe("local");
    // The card is the GPU's now — the placement id it is found by is not.
    expect(spot?.resource.id).toBe("gpu2");
    expect(spot?.placement.name).toBe("Qwen3.6-35B-A3B-heretic-NVFP4");
    expect(spot?.resource.controls).toContain("agent");
  });

  it("finds a swap node's slot on the remote node", () => {
    const spot = findPlacement(nodes, "boxa/slot0");
    expect(spot?.node.id).toBe("boxa");
    expect(spot?.resource.id).toBe("gpu0");
    expect(spot?.placement.name).toBe("heretic");
  });

  it("returns null once the placement leaves the board", () => {
    // Parked/unloaded/swapped away: a real answer, not a lookup failure —
    // the drawer keeps its last-known data and says the placement is gone.
    const parked = stateWith();
    parked.world.tenants.agent = {
      engine: "hipfire", gpu_index: 2, state: "parked", model: null, footprint: 0, queue_depth: null,
    };
    expect(findPlacement(buildNodes(parked, {}), "local/agent")).toBeNull();
  });

  it("returns null for an id no node ever carried", () => {
    expect(findPlacement(nodes, "local/nope")).toBeNull();
    expect(findPlacement([], "local/agent")).toBeNull();
  });
});

describe("buildNodes — registry nodes", () => {
  it("a registry node-agent entry with control:'none' becomes an observe-only card", () => {
    const s = stateWith({ nodes: [localEntry, heraEntry] });
    const nodes = buildNodes(s, {});
    const hera = nodes.find((n) => n.id === "hera")!;
    expect(hera.label).toBe("Hera Box");
    expect(hera.status).toBe("reachable");
    expect(hera.servingLine).toBe("big-model");
    expect(hera.resources).toHaveLength(1);
    expect(hera.resources[0].capacity).toEqual({
      used: 1024 * 1024 * 1024, total: 24576 * 1024 * 1024 });
    expect(hera.resources[0].controls).toEqual([]); // observe-only: no verbs
    expect(hera.resources[0].placements).toEqual([]); // and no placements
  });

  it.each([
    ["offline", "unreachable"],
    ["error", "down"],
    ["unconfigured", "unreachable"],
    [null, "unreachable"],
  ] as const)("observer status %s renders as %s", (status, expected) => {
    const s = stateWith({ nodes: [localEntry, { ...heraEntry, status,
      error: "backend sentence" }] });
    const hera = buildNodes(s, {}).find((n) => n.id === "hera")!;
    expect(hera.status).toBe(expected);
    expect(hera.detail).toBe("backend sentence");
  });

  it("a swap node's label always comes from the registry, never a hardcoded id", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const nodes = buildNodes(s, { boxa: sparkStatus() });
    // Exactly one card for the entry — never a second, separately-built one.
    expect(nodes.filter((n) => n.id === "boxa")).toHaveLength(1);
    expect(nodes.find((n) => n.id === "boxa")?.label).toBe("Box Alpha");
  });
});

// ---------------------------------------------------------------------------
// Declared REMOTE engines (Task 10b) — a node-agent entry's own engines[],
// seen through `world.remote_tenants` (app/state.py's World.snapshot_remote
// stamps node_id/resource/engine/gpu_index on every record, :300-310) and
// keyed `<node>/<resource>` (app/observe.py's node_key).
// ---------------------------------------------------------------------------

// Fixture rule again: node "zeta" (never sparky), resource "song-lab" (never
// omni), GPUs 4/5 (never 0/1, never the local fixture's 2/3). GPU 5 carries
// NO declared engine on purpose — the bare-GPU card must survive beside the
// engine cards.
const zetaEntry: DeckNodeEntry = {
  id: "zeta", label: "Zeta Box", agent_kind: "node-agent",
  address: "http://zeta:7720", serving_address: null, credential_set: true,
  control: "none",
  status: "online", last_seen: "2026-08-16T00:00:00+00:00",
  gpus: [
    { index: 4, name: "GB10", memory_used_mb: 62_000, memory_total_mb: 122_880,
      utilization_percent: 41 },
    { index: 5, name: "GB10", memory_used_mb: 1_024, memory_total_mb: 122_880,
      utilization_percent: 0 },
  ],
  serving: null, error: null,
};

function remoteTenant(overrides: Partial<RemoteTenant> = {}): RemoteTenant {
  return {
    // The kind's own observe() shape (app/engine_kinds.py:917-929) plus the
    // four fields World.snapshot_remote stamps on every record.
    engine: "sglang-omni", gpu_index: 4, state: "idle",
    busy_requests: 0, model: null, idle_s: 42,
    node_id: "zeta", resource: "song-lab",
    ...overrides,
  };
}

// GET /api/engine-kinds, pinned to the live sglang-omni row
// (app/engine_kinds.py:190-191 + :966-970's human_verbs).
const OMNI_KINDS: EngineKindsResponse = {
  kinds: [{
    kind: "sglang-omni", connection: { url: { required: true } },
    remote_capable: true, local_capable: false, human_verbs: ["load", "unload"],
  }],
};

function zetaState(
  overrides: Parameters<typeof stateWith>[0] = {},
): StateResponse {
  return stateWith({
    nodes: [localEntry, zetaEntry],
    remoteTenants: { "zeta/song-lab": remoteTenant() },
    ...overrides,
  });
}

describe("buildNodes — declared remote engines", () => {
  it("puts a node-agent entry's declared engine on its GPU's card, chip and verbs together", () => {
    // Reworked from "gives a declared engine its own card": the standalone
    // engine card is what showed sparky a 110 GB meter belonging to the
    // serving model. The engine's chip and its verbs ride the GPU card now.
    const zeta = buildNodes(zetaState(), {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    const card = gpuCard(zeta, 4);
    expect(card.label).toBe("GPU 4");
    // Capacity comes from the node's OWN gpu list, matched on the declared
    // gpu_index and converted MB -> bytes.
    expect(card.capacity).toEqual({
      used: 62_000 * 1024 * 1024, total: 122_880 * 1024 * 1024 });
    // The local-world control dispatch must NEVER fire for a remote card
    // (nodes.ts's header: App.tsx drills the LOCAL box's world down to every
    // card, so a remote card reading `world.tenants[control]` would describe
    // the wrong machine).
    expect(card.controls).toEqual([]);
    expect(card.remoteEngines).toEqual([{
      nodeId: "zeta", resource: "song-lab", kind: "sglang-omni", state: "idle",
      verbs: [{ verb: "load", disabled: true }, { verb: "unload", disabled: false }],
    }]);
    expect(card.placements.map((p) => p.id)).toEqual(["zeta/song-lab"]);
  });

  it("every reported GPU keeps its card; the engine's GPU is not claimed away", () => {
    // Reworked from "the engine's card replaces its GPU's bare meter": GPU
    // cards always show (design ruling 2), so GPU 4 keeps its own card AND
    // carries the engine, instead of being suppressed in favour of one.
    const zeta = buildNodes(zetaState(), {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    expect(zeta.resources.map((r) => r.id)).toEqual(["gpu4", "gpu5"]);
    expect(gpuCard(zeta, 5).remoteEngines).toEqual([]);
    expect(gpuCard(zeta, 5).placements).toEqual([]);
  });

  it("reports unknown capacity rather than zero when the declared GPU is not in the node's list", () => {
    const s = zetaState({
      remoteTenants: { "zeta/song-lab": remoteTenant({ gpu_index: 9 }) },
    });
    const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    // The declared index still earns a card — an engine whose gpu_index the
    // node never reported must not vanish with it.
    expect(zeta.resources.map((r) => r.id)).toEqual(["gpu4", "gpu5", "gpu9"]);
    expect(gpuCard(zeta, 9).capacity).toBeNull();
    expect(gpuCard(zeta, 9).remoteEngines!.map((c) => c.resource)).toEqual(["song-lab"]);
  });

  it("renders the lifecycle's word, never a re-derived one", () => {
    for (const status of ["serving", "warming", "quarantined", "parked", "down"] as const) {
      const s = zetaState({
        lifecycle: { "zeta/song-lab": lifecycleEntry({ status }) },
      });
      const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
      expect(card.placements[0].status).toBe(status);
    }
  });

  it.each([
    // Only reachable on a deck with NO intent store, where the whole
    // lifecycle view is `{}` (app/routers/__init__.py:136-137) — so these
    // are derive_status's own no-intent arms, not a second derivation.
    // Both rows here are RESIDENT states (busy|idle), so the Task 2
    // visibility filter never enters into it — the down/unknown rows this
    // table used to carry moved to the test below, because those are
    // NOT resident and (with no intent store to force a failure status) the
    // card no longer renders at all.
    ["idle", "unmanaged"],
    ["busy", "unmanaged"],
  ])("with an empty lifecycle view, %s falls back to %s", (state, expected) => {
    const s = zetaState({
      remoteTenants: { "zeta/song-lab": remoteTenant({ state }) },
      lifecycle: {},
    });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements[0].status).toBe(expected);
    // The engine's own word is untouched by that fallback — it is a
    // different vocabulary, carried for the verbs beside the chip.
    expect(card.remoteEngines![0].state).toBe(state);
  });

  it.each(["down", "unknown"] as const)(
    "with an empty lifecycle view, a non-resident %s engine loses its CHIP, not its GPU card",
    (state) => {
      // Reworked twice: Task 2's interim filter dropped the whole card (a
      // GPU meter disappearing with an idle engine). Now the visibility rule
      // is applied exactly once, per CHIP: the GPU card and its meter stay,
      // the chip and its verbs go, and the engine moves to the node's
      // hiddenEngines — the header menu's source, so it can still be loaded.
      const s = zetaState({
        remoteTenants: { "zeta/song-lab": remoteTenant({ state }) },
        lifecycle: {},
      });
      const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
      expect(zeta.resources.map((r) => r.id)).toEqual(["gpu4", "gpu5"]);
      expect(gpuCard(zeta, 4).capacity).toEqual({
        used: 62_000 * 1024 * 1024, total: 122_880 * 1024 * 1024 });
      expect(gpuCard(zeta, 4).placements).toEqual([]);
      expect(gpuCard(zeta, 4).remoteEngines).toEqual([]);
      expect(zeta.hiddenEngines).toEqual([{
        nodeId: "zeta", resource: "song-lab", kind: "sglang-omni", state,
        verbs: remoteEngineVerbs(OMNI_KINDS, "sglang-omni", state, false),
      }]);
      // The point of listing it: load is offerable on a reachable node.
      expect(zeta.hiddenEngines![0].verbs.find((v) => v.verb === "load")!.disabled).toBe(false);
    },
  );

  it("keeps the chip via an ALWAYS_VISIBLE lifecycle status even while the engine is not resident", () => {
    // quarantined is a failure the operator must still be able to see, even
    // though a quarantined engine typically has nothing loaded — the
    // approved exception to the "no model, no chip" ruling.
    const s = zetaState({
      remoteTenants: {
        "zeta/song-lab": remoteTenant({ state: "down", busy_requests: null, idle_s: null }),
      },
      lifecycle: { "zeta/song-lab": lifecycleEntry({ status: "quarantined" }) },
    });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements).toHaveLength(1);
    expect(card.placements[0].status).toBe("quarantined");
    expect(card.placements[0].name).toBe("song-lab");
    expect(card.placements[0].kind).toBe("engine");
    // No idle reading at all on this record — absent, never a fabricated 0.
    expect(card.placements[0].idleSeconds).toBeNull();
    expect(card.remoteEngines![0].verbs).toEqual([
      { verb: "load", disabled: false }, { verb: "unload", disabled: true },
    ]);
  });

  it("hides a parked-but-not-resident remote engine — parked is not in the ALWAYS_VISIBLE failure set", () => {
    // Superseded ruling: this used to be grouped with down/quarantined as
    // "the chip must survive". Parked is the resource's intentional empty
    // state (an operator unloaded it), not a failure, so under the Task 2
    // ruling it is exactly the case the whole redesign exists to hide.
    const s = zetaState({
      remoteTenants: {
        "zeta/song-lab": remoteTenant({ state: "down", busy_requests: null, idle_s: null }),
      },
      lifecycle: { "zeta/song-lab": lifecycleEntry({ status: "parked" }) },
    });
    const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).placements).toEqual([]);
    expect(gpuCard(zeta, 4).remoteEngines).toEqual([]);
    expect(zeta.hiddenEngines!.map((c) => c.resource)).toEqual(["song-lab"]);
  });

  it("an unavailable busy indicator reads BUSY, exactly as the backend already read it", () => {
    // app/engine_kinds.py:902-908: `busy is None or busy > 0` -> state
    // "busy". The count itself is NOT re-read here — a UI that branched on
    // busy_requests === null would be a second, drifting copy of design
    // §4's fails-toward-alive rule, and there is no distinct "unknown busy"
    // presentation because the backend vocabulary serves none.
    const s = zetaState({
      remoteTenants: {
        "zeta/song-lab": remoteTenant({ state: "busy", busy_requests: null }),
      },
    });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.remoteEngines![0].state).toBe("busy");
    // Neither chip fact is borrowed from a kind that means something else by
    // it: `busy` carries hipfire's park-refusal copy (labels.inUseTitle) and
    // this kind has no queue concept at all. Absent renders nothing.
    expect(card.placements[0].busy).toBeUndefined();
    expect(card.placements[0].queue).toBeUndefined();
  });

  it("badges the kind: a remote card has no 'usual engine' for it to be unremarkable against", () => {
    const card = gpuCard(buildNodes(zetaState(), {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements[0].engine).toBe("sglang-omni");
    expect(card.placements[0].engineBadge).toBe("sglang-omni");
  });

  it("carries the deck-wide policy row for the resource", () => {
    // PolicyStore is keyed by BARE resource across the whole deck (ruling
    // R10, app/policy.py's declared_defaults) — the same map the local cards
    // read, not a node-scoped one.
    const s = zetaState({
      policy: { ...DEFAULT_POLICY, "song-lab": { priority: 7, pinned: true, idle_ttl: 600 } },
    });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements[0].pinned).toBe(true);
    expect(card.placements[0].priority).toBe(7);
    expect(card.placements[0].idleSeconds).toBe(42);
  });

  it("survives a poll where the policy row has not been materialized yet", () => {
    // The local rows are all there; only the just-declared remote engine's
    // is missing — the poll that reads policy.json a moment before the
    // declaration lands. An absent row must not blank the board.
    const s = zetaState({ policy: { ...DEFAULT_POLICY } });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements[0].pinned).toBeUndefined();
    expect(card.placements[0].priority).toBeUndefined();
  });

  it("an unreachable node keeps its engine chip, marks it stale, and withholds every verb", () => {
    const s = zetaState({ nodes: [localEntry, { ...zetaEntry, status: "offline" }] });
    const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    const card = gpuCard(zeta, 4);
    expect(zeta.status).toBe("unreachable");
    expect(card.placements[0].stale).toBe(true);
    expect(card.remoteEngines![0].verbs.every((v) => v.disabled)).toBe(true);
  });

  it("renders the card with no verbs at all until the kinds catalog lands", () => {
    const card = gpuCard(buildNodes(zetaState(), {}, null).find((n) => n.id === "zeta")!, 4);
    expect(card.placements).toHaveLength(1);
    expect(card.remoteEngines![0].verbs).toEqual([]);
  });

  it("never carries settings drift on a remote engine placement", () => {
    // Deliberate omission: DriftCard's Settings target is the NODE's
    // configurable engine (a swap node's vllm), which is not this engine —
    // offering it would open a scope key nothing resolves (the D11 defect
    // ModelDetailDrawer's own comment names). See nodes.ts.
    const s = zetaState({
      lifecycle: {
        "zeta/song-lab": lifecycleEntry({
          settings_drift: { changed: ["args:x"], entries: [], since: null },
        }),
      },
    });
    const card = gpuCard(buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!, 4);
    expect(card.placements[0].settingsDrift).toBeUndefined();
  });

  it("another node's remote engines never land on this card", () => {
    const s = zetaState({
      nodes: [localEntry, zetaEntry, heraEntry],
      remoteTenants: {
        "zeta/song-lab": remoteTenant(),
        "hera/mixdown": remoteTenant({ node_id: "hera", resource: "mixdown", gpu_index: 0 }),
      },
    });
    const nodes = buildNodes(s, {}, OMNI_KINDS);
    const zeta = nodes.find((n) => n.id === "zeta")!;
    const hera = nodes.find((n) => n.id === "hera")!;
    expect(gpuCard(zeta, 4).remoteEngines!.map((c) => c.resource)).toEqual(["song-lab"]);
    expect(gpuCard(zeta, 5).remoteEngines).toEqual([]);
    // hera reports one GPU (index 0) and declares mixdown on it.
    expect(hera.resources.map((r) => r.id)).toEqual(["gpu0"]);
    expect(gpuCard(hera, 0).remoteEngines!.map((c) => c.resource)).toEqual(["mixdown"]);
  });

  it("a swap node's slot and its declared engines share ONE card — the sparky collapse", () => {
    // The one live topology, and the defect that started this wave: sparky
    // rendered the serving slot and a standalone omni card whose meter showed
    // the SERVING model's 110 GB. One GB10, one card: the slot's chip and
    // profile picker, the engine's chip and verbs, one meter.
    const s = stateWith({
      nodes: [localEntry, boxaEntry],
      remoteTenants: {
        "boxa/song-lab": remoteTenant({ node_id: "boxa", gpu_index: 0 }),
      },
    });
    const boxa = buildNodes(s, { boxa: sparkStatus() }, OMNI_KINDS)
      .find((n) => n.id === "boxa")!;
    expect(boxa.resources.map((r) => r.id)).toEqual(["gpu0"]);
    const card = boxa.resources[0];
    expect(card.controls).toEqual(["spark"]);
    // Slot chip first, then the engines declared on that GPU.
    expect(card.placements.map((p) => p.id)).toEqual(["boxa/slot0", "boxa/song-lab"]);
    expect(card.remoteEngines![0].nodeId).toBe("boxa");
    // The swap node's own GPU list is the meter source here too.
    expect(card.capacity).toEqual({ used: 4096 * 1024 * 1024, total: 49152 * 1024 * 1024 });
  });

  it("a node-agent entry with no remote tenants renders exactly as before", () => {
    // The observe-only regression pin: `world.remote_tenants` absent
    // entirely (every deck built without a remote observer) must change
    // nothing about the card this file already produced.
    const s = stateWith({ nodes: [localEntry, zetaEntry] });
    const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    expect(zeta.resources.map((r) => r.id)).toEqual(["gpu4", "gpu5"]);
    expect(zeta.resources.every((r) => r.controls.length === 0)).toBe(true);
    expect(zeta.resources.every((r) => r.placements.length === 0)).toBe(true);
    expect(zeta.resources.every((r) => r.remoteEngines!.length === 0)).toBe(true);
    expect(zeta.hiddenEngines).toEqual([]);
  });

  it("the placement id is the lifecycle key, so the detail drawer can re-derive it", () => {
    const nodes = buildNodes(zetaState(), {}, OMNI_KINDS);
    const spot = findPlacement(nodes, "zeta/song-lab")!;
    expect(spot.node.id).toBe("zeta");
    expect(spot.resource.id).toBe("gpu4");
    expect(spot.placement.name).toBe("song-lab");
  });
});

// ---------------------------------------------------------------------------
// Per-GPU telemetry (this wave) — `nodes[].gpus`, the ONE telemetry shape
// every node speaks: a node-agent entry's own probe (node-agent/models.py:23-37)
// for a remote box, app/telemetry.py's pass-through of dashboard-api's
// IndividualGPU rows (dashboard-api/models.py:129-143) for the local one,
// attached to the LOCAL entry by app/routers/status.py's _nodes_block.
// ---------------------------------------------------------------------------

// Deliberately NOT the same numbers as DEFAULT_GPUS' bytes: capacity stays
// world.gpus' fact and stats stay telemetry's, and a fixture where the two
// agreed could not tell one being read for the other ([[defaults-that-hide-bugs]]).
const localTelemetryEntry: DeckNodeEntry = {
  ...localEntry,
  gpus: [
    { index: 2, name: "AMD Radeon AI PRO R9700", memory_used_mb: 21_000,
      memory_total_mb: 32_624, memory_percent: 64.4, utilization_percent: 97,
      temperature_c: 71, power_w: 214.5, uuid: "gpu-uuid-2" },
    { index: 3, name: "AMD Radeon AI PRO R9700", memory_used_mb: 190,
      memory_total_mb: 32_624, memory_percent: 0.6, utilization_percent: 0,
      temperature_c: 29, power_w: null, uuid: "gpu-uuid-3" },
  ],
};

describe("buildNodes — GPU telemetry on local cards", () => {
  it("maps the LOCAL registry entry's telemetry onto each GPU card", () => {
    const [local] = buildNodes(stateWith({ nodes: [localTelemetryEntry, heraEntry] }), {});
    expect(gpuCard(local, 2).stats).toEqual({
      name: "AMD Radeon AI PRO R9700",
      utilizationPercent: 97,
      temperatureC: 71,
      powerW: 214.5,
    });
    // Absent, never zero: dashboard-api's own power_w is Optional
    // (dashboard-api/models.py:138), and "0.0W" would be a reading it never took.
    expect(gpuCard(local, 3).stats!.powerW).toBeNull();
    // Capacity is still world.gpus' bytes — telemetry is display garnish and
    // never the meter's source.
    expect(gpuCard(local, 2).capacity).toEqual({
      used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("leaves stats absent when the local entry reports no telemetry at all", () => {
    // The dashboard-api-unreachable case: `gpus: null` for local exactly as
    // for a dark remote. The meter still reads, the stats block renders none.
    const [local] = buildNodes(stateWith({ nodes: [localEntry] }), {});
    expect(gpuCard(local, 2).stats).toBeUndefined();
    expect(gpuCard(local, 2).capacity).toEqual({
      used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("leaves stats absent when the payload carries no nodes block at all", () => {
    // Every pre-registry fixture and every deck polled before the block
    // landed: absence is representable, and it is not an error.
    const [local] = buildNodes(stateWith(), {});
    expect(local.resources.every((r) => r.stats === undefined)).toBe(true);
  });

  it("leaves stats absent for a GPU a REMOTE node's telemetry does not cover", () => {
    // The remote arm keeps this case: `cardIndices` unions the reported GPUs
    // with the indices visible declared engines name, so a card can exist for
    // an index the probe never covered. (The LOCAL arm no longer has this
    // case — a short local list is a cross-sensor disagreement and drops
    // wholesale; see the count-guard tests below.)
    const partial: DeckNodeEntry = {
      ...zetaEntry,
      gpus: [{ index: 4, name: "GB10", memory_used_mb: 62_000, memory_total_mb: 122_880,
               utilization_percent: 41, temperature_c: 63, power_w: 88.25 }],
    };
    const s = stateWith({
      nodes: [localEntry, partial],
      remoteTenants: { "zeta/song-lab": remoteTenant({ gpu_index: 9 }) },
    });
    const zeta = buildNodes(s, {}, OMNI_KINDS).find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).stats!.utilizationPercent).toBe(41);
    expect(gpuCard(zeta, 9).stats).toBeUndefined();
  });

  it("reads a remote node's telemetry off its own entry, temperature and power included", () => {
    const hot: DeckNodeEntry = {
      ...zetaEntry,
      gpus: [
        { index: 4, name: "GB10", memory_used_mb: 62_000, memory_total_mb: 122_880,
          utilization_percent: 41, temperature_c: 63, power_w: 88.25 },
        // The REAL failed-sensor payload: `temperature_c` and
        // `utilization_percent` are REQUIRED fields in both producers
        // (node-agent/models.py:23-37, dashboard-api/models.py:129-143), so a
        // reading that could not be taken arrives as 0 WITH its flag false
        // (dashboard-api/gpu.py:172 `temperature_available=temp > 0`). Both
        // must render "—", never the 0 that is on the wire.
        { index: 5, name: "GB10", memory_used_mb: 1_024, memory_total_mb: 122_880,
          utilization_percent: 0, utilization_available: false,
          temperature_c: 0, temperature_available: false, power_w: null },
      ],
    };
    const zeta = buildNodes(stateWith({ nodes: [localEntry, hot] }), {}).find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).stats).toEqual({
      name: "GB10", utilizationPercent: 41, temperatureC: 63, powerW: 88.25 });
    expect(gpuCard(zeta, 5).stats).toEqual({
      name: "GB10", utilizationPercent: null, temperatureC: null, powerW: null });
  });

  it("reads an OLDER node-agent's omitted fields as no reading too", () => {
    // The one case an omitted `temperature_c` is honest: a node-agent
    // predating the availability triple (and the fields themselves). Absent
    // means "no verdict offered", read as available — so an omitted READING
    // is still null, and a present one is still a reading.
    const legacy: DeckNodeEntry = {
      ...zetaEntry,
      gpus: [{ index: 4, name: "GB10", memory_used_mb: 62_000,
               memory_total_mb: 122_880, utilization_percent: 41 }],
    };
    const zeta = buildNodes(stateWith({ nodes: [localEntry, legacy] }), {})
      .find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).stats).toEqual({
      name: "GB10", utilizationPercent: 41, temperatureC: null, powerW: null });
    expect(gpuCard(zeta, 4).capacity).toEqual({
      used: 62_000 * 1024 * 1024, total: 122_880 * 1024 * 1024 });
  });

  it("folds the availability flags on the LOCAL entry's telemetry too", () => {
    // Same producer convention, the other arm: dashboard-api sends the
    // required number plus its flag, and a false flag means the number is
    // filler. Its own gpu.py:172 is the live case (an AMD card whose hwmon
    // temp read came back 0).
    const dark: DeckNodeEntry = {
      ...localTelemetryEntry,
      gpus: localTelemetryEntry.gpus!.map((g) => ({
        ...g, utilization_percent: 0, utilization_available: false,
        temperature_c: 0, temperature_available: false,
      })),
    };
    const [local] = buildNodes(stateWith({ nodes: [dark] }), {});
    expect(gpuCard(local, 2).stats).toEqual({
      name: "AMD Radeon AI PRO R9700", utilizationPercent: null,
      temperatureC: null, powerW: 214.5 });
    // The meter is world.gpus' own observation and is untouched by any of it.
    expect(gpuCard(local, 2).capacity).toEqual({
      used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("meters a remote GPU as UNKNOWN when its memory read was unavailable", () => {
    // A remote card's capacity comes from the SAME row as its stats, so an
    // unread VRAM figure would otherwise draw a confident empty bar. Null =
    // hatched (unknown), never a fabricated 0-used.
    const unread: DeckNodeEntry = {
      ...zetaEntry,
      gpus: [{ index: 4, name: "GB10", memory_used_mb: 0, memory_total_mb: 122_880,
               utilization_percent: 41, memory_usage_available: false }],
    };
    const zeta = buildNodes(stateWith({ nodes: [localEntry, unread] }), {})
      .find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).capacity).toBeNull();
    // The other readings still render — one dead sensor is not all of them.
    expect(gpuCard(zeta, 4).stats!.utilizationPercent).toBe(41);
  });
});

// ---------------------------------------------------------------------------
// The cross-sensor count guard (F7). world.gpus is the deck's OWN sysfs read
// (app/gpu.py) and the local telemetry is dashboard-api's rocm/nvidia-smi
// enumeration, joined POSITIONALLY. app/telemetry.py's `_qualified` aligns the
// divergence we know about; nothing can prove the two enumerations match, so a
// length disagreement drops the telemetry rather than sliding every card's
// readings onto its neighbour.
// ---------------------------------------------------------------------------

describe("buildNodes — local telemetry cross-sensor guard", () => {
  it("keeps the stats when both sensors enumerate the same number of GPUs", () => {
    const [local] = buildNodes(stateWith({ nodes: [localTelemetryEntry] }), {});
    expect(gpuCard(local, 2).stats!.temperatureC).toBe(71);
    expect(gpuCard(local, 3).stats!.temperatureC).toBe(29);
  });

  it("drops local telemetry WHOLESALE when the two enumerations disagree", () => {
    // dashboard-api sees one card the deck sees two of (or vice versa): its
    // row for index 2 may describe either GPU, so neither card may show it.
    // Capacity — the deck's own observation — is unaffected.
    const short = { ...localTelemetryEntry, gpus: [localTelemetryEntry.gpus![0]] };
    const [local] = buildNodes(stateWith({ nodes: [short] }), {});
    expect(local.resources.every((r) => r.stats === undefined)).toBe(true);
    expect(gpuCard(local, 2).capacity).toEqual({
      used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("drops it when telemetry reports MORE GPUs than the deck sees", () => {
    // The live autarch shape if `_qualified` ever stopped filtering: an iGPU
    // dashboard-api enumerates and app.gpu.read_gpus excludes.
    const extra = {
      ...localTelemetryEntry,
      gpus: [...localTelemetryEntry.gpus!,
             { index: 4, name: "AMD iGPU", memory_used_mb: 100,
               memory_total_mb: 2_048, utilization_percent: 0 }],
    };
    const [local] = buildNodes(stateWith({ nodes: [extra] }), {});
    expect(local.resources.every((r) => r.stats === undefined)).toBe(true);
  });

  it("leaves a REMOTE node's stats alone — one list is both its sensors", () => {
    // A remote card's capacity and stats come from the same `entry.gpus`
    // row, so there is no second enumeration to disagree with and no guard
    // to apply: zeta reports one GPU while this box has two.
    const one: DeckNodeEntry = {
      ...zetaEntry,
      gpus: [{ index: 4, name: "GB10", memory_used_mb: 62_000,
               memory_total_mb: 122_880, utilization_percent: 41, temperature_c: 63 }],
    };
    const zeta = buildNodes(stateWith({ nodes: [localTelemetryEntry, one] }), {})
      .find((n) => n.id === "zeta")!;
    expect(gpuCard(zeta, 4).stats!.temperatureC).toBe(63);
  });

  it("carries a swap node's telemetry onto the card its serving slot rides", () => {
    const s = stateWith({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].stats).toEqual({
      name: "RTX 6000", utilizationPercent: 10, temperatureC: null, powerW: null });
  });
});

describe("buildNodes — local per-GPU card composition", () => {
  it("puts each declared resource's chip on the card for the GPU it runs on", () => {
    // The wave's load-bearing local case: 2 GPUs, 3 resources, one of them
    // co-resident. A loaded hipfire chip on its own GPU; on the shared one an
    // unloaded lemonade (control, no chip) beside an idle comfyui (also no
    // chip — Task 2's ruling), so the card is a meter with two control rows.
    const [local] = buildNodes(stateWith({ nodes: [localTelemetryEntry] }), {});
    expect(local.resources.map((r) => r.id)).toEqual(["gpu2", "gpu3"]);
    expect(gpuCard(local, 2).placements.map((p) => p.id)).toEqual(["local/agent"]);
    expect(gpuCard(local, 3).controls).toEqual(["gguf-a", "img"]);
    expect(gpuCard(local, 3).placements).toEqual([]);
  });

  it("puts two visible co-resident chips on the one card, in board order", () => {
    const s = stateWith();
    s.world.tenants["gguf-a"] = {
      engine: "lemonade", gpu_index: 3, state: "loaded", model: "qwen.gguf",
      footprint: 9_000_000_000, idle_s: 4,
    };
    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: 2, idle_s: 0 };
    const [local] = buildNodes(s, {});
    expect(gpuCard(local, 3).placements.map((p) => p.id))
      .toEqual(["local/gguf-a", "local/img"]);
  });

  it("never carries remoteEngines on a local card — nothing remote is declared there", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(local.resources.every((r) => r.remoteEngines === undefined)).toBe(true);
    expect(local.hiddenEngines).toBeUndefined();
  });
});

describe("controlHasPlacement", () => {
  it("answers per CONTROL where a card-level question cannot: one GPU, two resources", () => {
    // A shared GPU card carries both controls. A card-level "is anything on
    // this card" cannot serve a control row: PlacementActions needs "does
    // THIS control have a chip", or the unloaded neighbour's row stops
    // naming itself.
    const s = stateWith();
    s.world.tenants.img = { engine: "comfyui", gpu_index: 3, state: "busy", queue: 1, idle_s: 0 };
    const [local] = buildNodes(s, {});
    const shared = gpuCard(local, 3);
    expect(shared.controls).toEqual(["gguf-a", "img"]);
    expect(controlHasPlacement(shared, "gguf-a")).toBe(false); // unloaded: no chip
    expect(controlHasPlacement(shared, "img")).toBe(true);     // busy: a chip
  });

  it("ignores an external process sharing the same card", () => {
    // An external pid's placement is `external/<pid>`, never `local/<control>`,
    // so it can never be mistaken for a control's own chip.
    const s = stateWith();
    s.world.externals = [{ pid: 99, gpu: 3, bytes: 2_000_000_000 }];
    const [local] = buildNodes(s, {});
    expect(controlHasPlacement(gpuCard(local, 3), "gguf-a")).toBe(false);
  });

  it("is false for a control that is not on the card at all", () => {
    const [local] = buildNodes(stateWith(), {});
    expect(controlHasPlacement(gpuCard(local, 2), "gguf-a")).toBe(false);
  });
});

describe("buildNodes — swap node per-GPU composition", () => {
  const withOmni = (overrides: Partial<RemoteTenant> = {}) => stateWith({
    nodes: [localEntry, boxaEntry],
    remoteTenants: {
      "boxa/song-lab": remoteTenant({ node_id: "boxa", gpu_index: 0, ...overrides }),
    },
  });

  it("hides a non-resident engine's chip while keeping the slot's card intact", () => {
    // The live sparky case this wave fixes: omni declared, nothing loaded on
    // it, Nemotron serving. One card, one meter, the serving chip — and omni
    // in hiddenEngines so the header menu can still load it.
    const boxa = buildNodes(
      withOmni({ state: "down", busy_requests: null, idle_s: null }),
      { boxa: sparkStatus() }, OMNI_KINDS,
    ).find((n) => n.id === "boxa")!;
    const card = boxa.resources[0];
    expect(boxa.resources.map((r) => r.id)).toEqual(["gpu0"]);
    expect(card.placements.map((p) => p.id)).toEqual(["boxa/slot0"]);
    expect(card.remoteEngines).toEqual([]);
    expect(card.controls).toEqual(["spark"]);
    expect(boxa.hiddenEngines).toEqual([{
      nodeId: "boxa", resource: "song-lab", kind: "sglang-omni", state: "down",
      verbs: [{ verb: "load", disabled: false }, { verb: "unload", disabled: true }],
    }]);
  });

  it("withholds a hidden engine's verbs while the node is dark, and still lists it", () => {
    // Task 5's menu renders from this list on an unreachable node too: the
    // engine is still declared, and the disabled-ness travels with the verb
    // rather than being re-decided at the menu.
    const s = withOmni({ state: "unknown" });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({
        status: "unreachable",
        observed: { reachable: false, loaded: false, model: null, transitioning: false },
      }),
    };
    const boxa = buildNodes(s, { boxa: sparkStatus() }, OMNI_KINDS).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("unreachable");
    expect(boxa.hiddenEngines!.map((c) => c.resource)).toEqual(["song-lab"]);
    expect(boxa.hiddenEngines![0].verbs.every((v) => v.disabled)).toBe(true);
  });

  it("falls back to ONE card, capacity unknown, when the node reports no GPUs — slot and engines intact", () => {
    const s = withOmni();
    s.nodes = [localEntry, { ...boxaEntry, gpus: null }];
    const boxa = buildNodes(s, { boxa: sparkStatus() }, OMNI_KINDS).find((n) => n.id === "boxa")!;
    expect(boxa.resources).toHaveLength(1);
    expect(boxa.resources[0].capacity).toBeNull();
    expect(boxa.resources[0].stats).toBeUndefined();
    expect(boxa.resources[0].controls).toEqual(["spark"]);
    expect(boxa.resources[0].placements.map((p) => p.id))
      .toEqual(["boxa/slot0", "boxa/song-lab"]);
    expect(boxa.resources[0].remoteEngines!.map((c) => c.resource)).toEqual(["song-lab"]);
  });

  it("keeps the serving slot's card even when the node reports neither GPUs nor engines", () => {
    // Nothing to name a GPU card after, and the profile picker still has to
    // render: the slot keeps its own card, exactly as before this wave.
    const s = stateWith({ nodes: [localEntry, { ...boxaEntry, gpus: null }] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources.map((r) => r.id)).toEqual(["slot0"]);
    expect(boxa.resources[0].label).toBe("Serving slot");
    expect(boxa.resources[0].controls).toEqual(["spark"]);
    expect(boxa.resources[0].placements[0].id).toBe("boxa/slot0");
  });

  it("puts the slot on the node's FIRST GPU card and engines on their own", () => {
    // A two-GPU swap node: spark reports no gpu_index for the slot, so it
    // rides the first card by convention; the engine rides its declared one.
    const twoGpu: DeckNodeEntry = {
      ...boxaEntry,
      gpus: [
        { index: 0, name: "RTX 6000", memory_used_mb: 4096, memory_total_mb: 49152,
          utilization_percent: 10 },
        { index: 1, name: "RTX 6000", memory_used_mb: 8192, memory_total_mb: 49152,
          utilization_percent: 20 },
      ],
    };
    const s = stateWith({
      nodes: [localEntry, twoGpu],
      remoteTenants: {
        "boxa/song-lab": remoteTenant({ node_id: "boxa", gpu_index: 1 }),
      },
    });
    const boxa = buildNodes(s, { boxa: sparkStatus() }, OMNI_KINDS).find((n) => n.id === "boxa")!;
    expect(boxa.resources.map((r) => r.id)).toEqual(["gpu0", "gpu1"]);
    expect(gpuCard(boxa, 0).controls).toEqual(["spark"]);
    expect(gpuCard(boxa, 0).placements.map((p) => p.id)).toEqual(["boxa/slot0"]);
    expect(gpuCard(boxa, 1).controls).toEqual([]);
    expect(gpuCard(boxa, 1).placements.map((p) => p.id)).toEqual(["boxa/song-lab"]);
  });
});
