import { describe, expect, it } from "vitest";
import type { LifecycleEntry, SparkStatus, StateResponse } from "../api";
import { buildNodes, isTenantName, TENANT_ORDER } from "./nodes";

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
    expect(buildNodes(null, null)).toEqual([]);
  });

  it("names the local node from the backend, not a hardcoded string", () => {
    const [local] = buildNodes(state(), null);
    expect(local.id).toBe("local");
    expect(local.label).toBe("autarch");
    expect(local.status).toBe("reachable");
  });

  it("makes one resource per GPU, in index order", () => {
    const [local] = buildNodes(state(), null);
    expect(local.resources.map((r) => r.id)).toEqual(["gpu0", "gpu1"]);
    expect(local.resources[0].label).toBe("GPU 0");
    expect(local.resources[0].capacity).toEqual({ used: 22_100_000_000, total: 34_000_000_000 });
  });

  it("places a tenant on the GPU the backend assigned it", () => {
    const [local] = buildNodes(state(), null);
    const gpu0 = local.resources[0];
    expect(gpu0.placements.map((p) => p.name)).toEqual(["Qwen3.6-35B-A3B-heretic-NVFP4"]);
    expect(gpu0.placements[0].engine).toBe("hipfire");
    expect(gpu0.placements[0].bytes).toBe(21_400_000_000);
  });

  it("keeps a model identity verbatim", () => {
    const [local] = buildNodes(state(), null);
    expect(local.resources[0].placements[0].name).toBe("Qwen3.6-35B-A3B-heretic-NVFP4");
  });

  it("omits an unloaded tenant rather than showing an empty chip", () => {
    const [local] = buildNodes(state(), null);
    const gpu1 = local.resources[1];
    expect(gpu1.placements.some((p) => p.engine === "lemonade")).toBe(false);
  });

  it("keeps an unloaded tenant's controls on its resource", () => {
    // The empty-slot case: no chip, but lemonade's Load dropdown still has
    // to render somewhere, so the resource carries the control list.
    const [local] = buildNodes(state(), null);
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
    const [local] = buildNodes(s, null);
    expect(local.resources[1].placements.map((p) => p.name)).toContain(
      "qwen2.5-14b-instruct-4k-q4_k_m.gguf",
    );
  });

  it("omits a parked hipfire", () => {
    const s = state();
    s.world.tenants.hipfire = { state: "parked", model: null, footprint: 0, queue_depth: null };
    const [local] = buildNodes(s, null);
    expect(local.resources[0].placements).toEqual([]);
  });

  it("surfaces an external process as its own placement", () => {
    const s = state();
    s.world.externals = [{ pid: 4242, gpu: 0, bytes: 1_200_000_000 }];
    const [local] = buildNodes(s, null);
    const external = local.resources[0].placements.find((p) => p.kind === "external");
    expect(external?.bytes).toBe(1_200_000_000);
    expect(external?.status).toBe("unmanaged");
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
      },
    };
    const [local] = buildNodes(s, null);
    expect(local.resources[0].placements[0].status).toBe("drifted");
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
    observed: { reachable: true, loaded: true, model: "heretic", transitioning: false },
    last_healthy_ts: "2026-08-04T14:23:11Z",
    ...overrides,
  };
}

describe("buildNodes — spark", () => {
  it("omits the node entirely when spark is not configured", () => {
    expect(buildNodes(state(), null).map((n) => n.id)).toEqual(["local"]);
  });

  it("adds a single serving-slot resource when configured", () => {
    const nodes = buildNodes(state(), sparkStatus());
    const spark = nodes[1];
    expect(spark.id).toBe("sparky");
    expect(spark.status).toBe("reachable");
    expect(spark.resources.map((r) => r.label)).toEqual(["Serving slot"]);
    expect(spark.resources[0].placements[0].name).toBe("heretic");
  });

  it("reports unknown capacity rather than zero", () => {
    const spark = buildNodes(state(), sparkStatus())[1];
    expect(spark.resources[0].capacity).toBeNull();
  });

  it("retains last-known placements when unreachable, and marks them stale", () => {
    const s = state();
    s.lifecycle = {
      "sparky/slot0": lifecycleEntry({
        status: "unreachable",
        observed: { reachable: false, loaded: false, model: null, transitioning: false },
      }),
    };
    const spark = buildNodes(s, sparkStatus())[1];

    expect(spark.status).toBe("unreachable");
    expect(spark.lastSeen).toBe("2026-08-04T14:23:11Z");
    // The whole point: an offline node must never blank out.
    expect(spark.resources[0].placements[0].name).toBe("heretic");
    expect(spark.resources[0].placements[0].stale).toBe(true);
  });

  it("reads as warming while a swap is in flight and the endpoint is down", () => {
    const spark = buildNodes(
      state(),
      sparkStatus({
        serving: { model: "heretic", endpoint_ok: false, container_status: "running" },
        swap_status: {
          state: "swapping", profile: "heretic", id: "1",
          message: "", ts: "2026-08-05T00:00:00Z",
        },
      }),
    )[1];
    expect(spark.status).toBe("warming");
  });

  it("reads as warming through the long tail of a boot the helper already calls done", () => {
    // The case a real spark boot spends nearly all its time in, and the one
    // the swap_status-only rule got wrong. The helper reports "done" as soon
    // as swap.sh launches (app/engines/spark.py, _BOOTING_STATES), so the
    // 5-15 minutes of weight load and autotune look like this: state "done",
    // endpoint still down. app/observe.py's boot_in_flight() is what knows
    // better, and it arrives here as observed.transitioning.
    const s = state();
    s.lifecycle = {
      "sparky/slot0": lifecycleEntry({
        status: "warming",
        reason: "a load or boot is in flight",
        observed: { reachable: true, loaded: false, model: null, transitioning: true },
      }),
    };
    const spark = buildNodes(
      s,
      sparkStatus({
        serving: { model: "heretic", endpoint_ok: false, container_status: "running" },
        swap_status: {
          state: "done", profile: "heretic", id: "1",
          message: "", ts: "2026-08-05T00:00:00Z",
        },
      }),
    )[1];
    expect(spark.status).toBe("warming");
  });

  it("is down, not warming, when the endpoint is dead with no swap running", () => {
    const spark = buildNodes(
      state(),
      sparkStatus({ serving: { model: "heretic", endpoint_ok: false, container_status: "exited" } }),
    )[1];
    expect(spark.status).toBe("down");
  });

  it("has an empty slot when nothing is serving", () => {
    const spark = buildNodes(
      state(),
      sparkStatus({ serving: { model: null, endpoint_ok: false, container_status: null } }),
    )[1];
    expect(spark.resources[0].placements).toEqual([]);
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
