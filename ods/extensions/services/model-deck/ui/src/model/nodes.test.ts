import { describe, expect, it } from "vitest";
import type { DeckNodeEntry, LifecycleEntry, SettingsDrift, SparkStatus, StateResponse } from "../api";
import {
  buildNodes,
  findPlacement,
  isSwapSlotId,
  isTenantName,
  nodeIdOfPlacement,
  swapNodes,
  TENANT_ORDER,
} from "./nodes";

function state(overrides: Partial<StateResponse> = {}): StateResponse {
  return {
    node: { id: "local", label: "autarch" },
    world: {
      gpus: [
        { index: 0, total: 34_000_000_000, used: 22_100_000_000, free: 11_900_000_000 },
        { index: 1, total: 34_000_000_000, used: 200_000_000, free: 33_800_000_000 },
      ],
      tenants: {
        lemonade: { state: "unloaded", model: null, footprint: null, idle_s: null },
        comfyui: { state: "idle", queue: 0, idle_s: 12 },
        hipfire: { state: "running", model: "Qwen3.6-35B-A3B-heretic-NVFP4", footprint: 21_400_000_000, queue_depth: 0 },
      },
      externals: [],
      default_route: null,
      placement: { hipfire: 0, lemonade: 1, comfyui: 1 },
    },
    policy: {
      lemonade: { priority: 1, pinned: false, idle_ttl: 900 },
      comfyui: { priority: 2, pinned: false, idle_ttl: 300 },
      hipfire: { priority: 3, pinned: true, idle_ttl: 0 },
    },
    models: [],
    lifecycle: {},
    ...overrides,
  };
}

describe("buildNodes", () => {
  it("returns nothing before the first successful poll", () => {
    expect(buildNodes(null, {})).toEqual([]);
  });

  it("names the local node from the backend, not a hardcoded string", () => {
    const [local] = buildNodes(state(), {});
    expect(local.id).toBe("local");
    expect(local.label).toBe("autarch");
    expect(local.status).toBe("reachable");
  });

  it("makes one resource per GPU, in index order", () => {
    const [local] = buildNodes(state(), {});
    expect(local.resources.map((r) => r.id)).toEqual(["gpu0", "gpu1"]);
    expect(local.resources[0].label).toBe("GPU 0");
    expect(local.resources[0].capacity).toEqual({ used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("places a tenant on the GPU the backend assigned it", () => {
    const [local] = buildNodes(state(), {});
    const gpu0 = local.resources[0];
    expect(gpu0.placements.map((p) => p.name)).toEqual(["Qwen3.6-35B-A3B-heretic-NVFP4"]);
    expect(gpu0.placements[0].engine).toBe("hipfire");
    expect(gpu0.placements[0].bytes).toBe(21_400_000_000);
  });

  it("keeps a model identity verbatim", () => {
    const [local] = buildNodes(state(), {});
    expect(local.resources[0].placements[0].name).toBe("Qwen3.6-35B-A3B-heretic-NVFP4");
  });

  it("omits an unloaded tenant rather than showing an empty chip", () => {
    const [local] = buildNodes(state(), {});
    const gpu1 = local.resources[1];
    expect(gpu1.placements.some((p) => p.engine === "lemonade")).toBe(false);
  });

  it("omits a lemonade load in flight the same as unloaded (no chip mid-load)", () => {
    // tenantPlacement's `t.state !== "loaded"` guard also covers "loading" —
    // deliberate: the PlacementActions pill (STATE_TONE) is where a load in
    // flight shows up today; a dedicated chip treatment is deferred.
    const s = state();
    s.world.tenants.lemonade = { state: "loading", model: null, footprint: null, idle_s: null };
    const [local] = buildNodes(s, {});
    expect(local.resources[1].placements.some((p) => p.engine === "lemonade")).toBe(false);
  });

  it("keeps an unloaded tenant's controls on its resource", () => {
    // The empty-slot case: no chip, but lemonade's Load dropdown still has
    // to render somewhere, so the resource carries the control list.
    const [local] = buildNodes(state(), {});
    expect(local.resources[1].controls).toEqual(["lemonade", "comfyui"]);
    expect(local.resources[0].controls).toEqual(["hipfire"]);
  });

  it("shows a loaded tenant", () => {
    const s = state();
    s.world.tenants.lemonade = {
      state: "loaded",
      model: "qwen2.5-14b-instruct-4k-q4_k_m.gguf",
      footprint: 9_500_000_000,
      idle_s: 4,
    };
    const [local] = buildNodes(s, {});
    expect(local.resources[1].placements.map((p) => p.name)).toContain(
      "qwen2.5-14b-instruct-4k-q4_k_m.gguf",
    );
  });

  it("omits a parked hipfire", () => {
    const s = state();
    s.world.tenants.hipfire = { state: "parked", model: null, footprint: 0, queue_depth: null };
    const [local] = buildNodes(s, {});
    expect(local.resources[0].placements).toEqual([]);
  });

  it("surfaces an external process as its own placement", () => {
    const s = state();
    s.world.externals = [{ pid: 4242, gpu: 0, bytes: 1_200_000_000 }];
    const [local] = buildNodes(s, {});
    const external = local.resources[0].placements.find((p) => p.kind === "external");
    expect(external?.bytes).toBe(1_200_000_000);
    expect(external?.status).toBe("unmanaged");
  });

  it("carries each tenant's policy onto its placement", () => {
    // The board's 📌 and P{n}. They come from state.policy, which buildNodes
    // already has in hand — the alternative is drilling a `policy` prop back
    // down to every card, which is the coupling this adapter exists to end.
    const s = state();
    s.world.tenants.lemonade = {
      state: "loaded", model: "qwen.gguf", footprint: 9_000_000_000, idle_s: 4,
    };
    const [local] = buildNodes(s, {});
    const hipfire = local.resources[0].placements[0];
    const lemonade = local.resources[1].placements.find((p) => p.engine === "lemonade");

    expect(hipfire.pinned).toBe(true);
    expect(hipfire.priority).toBe(3);
    expect(lemonade?.pinned).toBe(false);
    expect(lemonade?.priority).toBe(1);
  });

  it("marks hipfire busy when a turn is in flight, and not otherwise", () => {
    // Predicts the park refusal BEFORE the click: hipfire's single admission
    // slot is what makes park/apply 409 without force.
    const idle = buildNodes(state(), {})[0].resources[0].placements[0];
    expect(idle.busy).toBe(false);

    const s = state();
    s.world.tenants.hipfire = { ...s.world.tenants.hipfire, queue_depth: 2 };
    expect(buildNodes(s, {})[0].resources[0].placements[0].busy).toBe(true);
  });

  it("treats a missing hipfire queue reading as not busy", () => {
    const s = state();
    s.world.tenants.hipfire = { ...s.world.tenants.hipfire, queue_depth: null };
    expect(buildNodes(s, {})[0].resources[0].placements[0].busy).toBe(false);
  });

  it("carries ComfyUI's queue and idle time, and gives hipfire neither", () => {
    const s = state();
    s.world.tenants.comfyui = { state: "busy", queue: 3, idle_s: 0 };
    const [local] = buildNodes(s, {});
    const comfy = local.resources[1].placements.find((p) => p.engine === "comfyui");
    const hipfire = local.resources[0].placements[0];

    expect(comfy?.queue).toBe(3);
    expect(comfy?.idleSeconds).toBe(0);
    // Absent, not zero: hipfire has no queue and reports no idle time, and
    // "0" on screen would be a claim neither engine ever made.
    expect(hipfire.queue).toBeUndefined();
    expect(hipfire.idleSeconds).toBeUndefined();
  });

  it("keeps a drained ComfyUI queue distinct from an unreadable one", () => {
    const zero = buildNodes(state(), {})[0].resources[1].placements
      .find((p) => p.engine === "comfyui");
    expect(zero?.queue).toBe(0);

    const s = state();
    s.world.tenants.comfyui = { state: "unknown", queue: null, idle_s: null };
    const unknown = buildNodes(s, {})[0].resources[1].placements
      .find((p) => p.engine === "comfyui");
    expect(unknown?.queue).toBeNull();
  });

  it("never badges a local tenant with its own engine name", () => {
    // The badge answers "what is this node running that it usually isn't".
    // On the local box each tenant IS its engine, so there is no such
    // question and no badge.
    const [local] = buildNodes(state(), {});
    for (const r of local.resources) {
      for (const p of r.placements) expect(p.engineBadge).toBeUndefined();
    }
  });

  it("takes a placement's status from the lifecycle view", () => {
    const s = state();
    s.lifecycle = {
      "local/hipfire": {
        status: "drifted",
        reason: "settings changed",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: null,
      },
    };
    const [local] = buildNodes(s, {});
    expect(local.resources[0].placements[0].status).toBe("drifted");
  });

  it("carries settings drift onto a placement from the lifecycle view", () => {
    // Task 11's adapter copy — app/routers/__init__.py:116 writes
    // `settings_drift` onto every lifecycle entry; this is the one place
    // that value crosses into the board's own Placement shape.
    const drift: SettingsDrift = {
      changed: ["args:max-model-len"],
      entries: [
        { key: "args:max-model-len", old: "262144", new: "131072", ts: "2026-08-07T00:00:00Z" },
      ],
      since: "2026-08-07T00:00:00Z",
    };
    const s = state();
    s.lifecycle = {
      "local/hipfire": {
        status: "drifted",
        reason: "settings changed",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: drift,
      },
    };
    const [local] = buildNodes(s, {});
    expect(local.resources[0].placements[0].settingsDrift).toEqual(drift);
  });

  it("leaves settingsDrift undefined when the lifecycle entry carries none", () => {
    const s = state();
    s.lifecycle = {
      "local/hipfire": {
        status: "serving",
        reason: "",
        intent: null,
        observed: { reachable: true, loaded: true, model: null, transitioning: false },
        last_healthy_ts: null,
        settings_drift: null,
      },
    };
    const [local] = buildNodes(s, {});
    expect(local.resources[0].placements[0].settingsDrift).toBeUndefined();
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
    const s = state({ nodes: [localEntry, boxaEntry, boxbEntry] });
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
    expect(a.resources[0].placements[0].id).toBe("boxa/slot0");
    expect(b.resources[0].placements[0].id).toBe("boxb/slot0");
    expect(a.resources[0].placements[0].name).toBe("heretic");
    expect(b.resources[0].placements[0].name).toBe("ds4");
    // Neither node's serving status leaks into the other's card.
    expect(a.resources[0].placements).toHaveLength(1);
    expect(b.resources[0].placements).toHaveLength(1);
  });

  it("reports unknown capacity rather than zero", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].capacity).toBeNull();
  });

  it("control decides, data never does: a control:'none' entry never gets swap controls, even carrying serving data and a servingByNode entry", () => {
    // The watcher-shaped fixture nodes.ts's swapNodes docstring points at:
    // heraEntry already reports `serving` on the wire and is handed a real
    // SparkStatus here too, and still must render observe-only — presence of
    // serving data must never promote a node, only the DECLARED control does.
    const s = state({ nodes: [localEntry, heraEntry] });
    const hera = buildNodes(s, { hera: sparkStatus() }).find((n) => n.id === "hera")!;
    expect(hera.resources[0].controls).toEqual([]);
    expect(hera.resources[0].placements).toEqual([]);
  });

  it("falls back to the observe-only card when no status has landed yet, never a phantom control surface", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: null }).find((n) => n.id === "boxa")!;
    expect(boxa.label).toBe("Box Alpha"); // the registry label, same as the swap card would show
    expect(boxa.resources[0].controls).toEqual([]);
    expect(boxa.resources[0].placements).toEqual([]);
  });

  it("also falls back to observe-only when the id is simply absent from servingByNode (fetch not landed)", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {}).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].controls).toEqual([]);
  });

  it("retains last-known placements when unreachable, and marks them stale", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements[0].profile).toBeUndefined();
  });

  it("reads as warming while a swap is in flight and the endpoint is down", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: "heretic", endpoint_ok: false, container_status: "exited" } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.status).toBe("down");
  });

  it("is unreachable when the lifecycle view says so, even with a healthy-looking serving payload", () => {
    // Status derivation order: `!reachable` wins over everything else.
    const s = state({ nodes: [localEntry, boxaEntry] });
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

  it("badges a non-default engine and stays silent about the default one", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const vllm = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(vllm.resources[0].placements[0].engine).toBe("vllm");
    expect(vllm.resources[0].placements[0].engineBadge).toBeUndefined();

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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
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
    const s = state({ nodes: [localEntry, boxaEntry] });
    s.lifecycle = {
      "boxa/slot0": lifecycleEntry({ settings_drift: drift }),
    };
    const boxa = buildNodes(s, { boxa: sparkStatus() }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements[0].settingsDrift).toEqual(drift);
  });

  it("has an empty slot when nothing is serving", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const boxa = buildNodes(s, {
      boxa: sparkStatus({ serving: { model: null, endpoint_ok: false, container_status: null } }),
    }).find((n) => n.id === "boxa")!;
    expect(boxa.resources[0].placements).toEqual([]);
  });
});

describe("isTenantName", () => {
  it("accepts every tenant the adapter can emit as a control", () => {
    for (const t of TENANT_ORDER) expect(isTenantName(t)).toBe(true);
  });

  it("rejects the spark control, which is a surface and not a tenant", () => {
    // The board dispatches on this: if it ever returned true for "spark",
    // PlacementActions would be handed a non-tenant and read
    // world.tenants.spark — the exact confusion the guard exists to stop.
    expect(isTenantName("spark")).toBe(false);
  });

  it("rejects an unknown control rather than guessing", () => {
    expect(isTenantName("vllm")).toBe(false);
    expect(isTenantName("")).toBe(false);
  });
});

describe("isSwapSlotId", () => {
  it("recognizes a swap node's synthesized slot key, regardless of node id", () => {
    expect(isSwapSlotId("boxa/slot0")).toBe(true);
    expect(isSwapSlotId("boxb/slot0")).toBe(true);
  });

  it("rejects a local tenant placement id", () => {
    expect(isSwapSlotId("local/hipfire")).toBe(false);
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
    expect(nodeIdOfPlacement("local/hipfire")).toBe("local");
  });
});

describe("swapNodes", () => {
  it("returns only the control:swap entries, declared not inferred", () => {
    const s = state({ nodes: [localEntry, heraEntry, boxaEntry, boxbEntry] });
    expect(swapNodes(s).map((e) => e.id)).toEqual(["boxa", "boxb"]);
  });

  it("returns nothing before state has loaded", () => {
    expect(swapNodes(null)).toEqual([]);
  });

  it("returns nothing when the registry carries no swap nodes", () => {
    expect(swapNodes(state({ nodes: [localEntry, heraEntry] }))).toEqual([]);
  });
});

describe("findPlacement", () => {
  const nodes = buildNodes(state({ nodes: [localEntry, boxaEntry] }), { boxa: sparkStatus() });

  it("finds a local tenant placement and hands back the resource that carries it", () => {
    // The drawer needs the RESOURCE too: its `controls` are what decide
    // which verbs (if any) that placement gets.
    const spot = findPlacement(nodes, "local/hipfire");
    expect(spot?.node.id).toBe("local");
    expect(spot?.resource.id).toBe("gpu0");
    expect(spot?.placement.name).toBe("Qwen3.6-35B-A3B-heretic-NVFP4");
    expect(spot?.resource.controls).toContain("hipfire");
  });

  it("finds a swap node's slot on the remote node", () => {
    const spot = findPlacement(nodes, "boxa/slot0");
    expect(spot?.node.id).toBe("boxa");
    expect(spot?.resource.id).toBe("slot0");
    expect(spot?.placement.name).toBe("heretic");
  });

  it("returns null once the placement leaves the board", () => {
    // Parked/unloaded/swapped away: a real answer, not a lookup failure —
    // the drawer keeps its last-known data and says the placement is gone.
    const parked = state();
    parked.world.tenants.hipfire = {
      state: "parked", model: null, footprint: 0, queue_depth: null,
    };
    expect(findPlacement(buildNodes(parked, {}), "local/hipfire")).toBeNull();
  });

  it("returns null for an id no node ever carried", () => {
    expect(findPlacement(nodes, "local/nope")).toBeNull();
    expect(findPlacement([], "local/hipfire")).toBeNull();
  });
});

describe("buildNodes — registry nodes", () => {
  it("a registry node-agent entry with control:'none' becomes an observe-only card", () => {
    const s = state({ nodes: [localEntry, heraEntry] });
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
    const s = state({ nodes: [localEntry, { ...heraEntry, status,
      error: "backend sentence" }] });
    const hera = buildNodes(s, {}).find((n) => n.id === "hera")!;
    expect(hera.status).toBe(expected);
    expect(hera.detail).toBe("backend sentence");
  });

  it("a swap node's label always comes from the registry, never a hardcoded id", () => {
    const s = state({ nodes: [localEntry, boxaEntry] });
    const nodes = buildNodes(s, { boxa: sparkStatus() });
    // Exactly one card for the entry — never a second, separately-built one.
    expect(nodes.filter((n) => n.id === "boxa")).toHaveLength(1);
    expect(nodes.find((n) => n.id === "boxa")?.label).toBe("Box Alpha");
  });
});
