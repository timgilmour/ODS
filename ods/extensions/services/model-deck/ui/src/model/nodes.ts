/**
 * The board's data model — and the seam this whole redesign hangs off.
 *
 * Everything the board renders is a `DeckNode[]`. Today that is synthesized
 * from a backend that has no node concept: one local box (GPUs + three named
 * tenants) plus a special-cased spark. When a real node registry lands, THIS
 * FILE should be the only one that changes. If a registry change forces edits
 * in components, the seam was drawn in the wrong place.
 *
 * Pure: no React, no fetch, no clock. Everything is a function of its inputs.
 */

import type {
  ExternalProc,
  LifecycleMap,
  LifecycleStatus,
  SparkStatus,
  StateResponse,
  TenantName,
  World,
} from "../api";

/** Node-level rollup. Deliberately NOT LifecycleStatus: that describes one
 * resource's relationship to its recorded intent and stays on the placement.
 * This answers only "can I see this box, and how stale is what I'm showing" —
 * which is what decides whether the whole card desaturates. */
export type NodeStatus = "reachable" | "unreachable" | "warming" | "down";

export interface Placement {
  /** Lifecycle key where one exists ("local/hipfire", "sparky/slot0"), else
   * a synthetic id. Stable across polls, so it is safe as a React key. */
  id: string;
  /** Model identity, verbatim. Never prettified — truncate at render time. */
  name: string;
  bytes: number | null;
  status: LifecycleStatus;
  engine: string;
  kind: "model" | "engine" | "external";
  /** True when the owning node is unreachable: this is last-known
   * information, not current fact. Drives the italic caption on the chip. */
  stale: boolean;
}

export interface DeckResource {
  id: string;
  label: string;
  /** null means "capacity is unknown", which renders as a hatched track —
   * distinct from a real zero. An unreachable node reports null, never 0. */
  capacity: { used: number; total: number } | null;
  placements: Placement[];
  /** Which control surfaces belong to this resource, by engine name. This
   * exists because an EMPTY resource still needs its verbs: lemonade's Load
   * dropdown has to render on a GPU with nothing on it, and there is no
   * placement to hang it off. Keeping it here rather than letting components
   * re-derive it from `world.placement` is what keeps node-specific
   * knowledge inside this file. */
  controls: string[];
}

export interface DeckNode {
  id: string;
  label: string;
  status: NodeStatus;
  lastSeen: string | null;
  resources: DeckResource[];
}

// Spark is a single-slot node and its lifecycle key is a fixed constant
// backend-side (app/observe.py SPARK_SLOT_KEY). Mirrored, not derived.
const SPARK_SLOT_KEY = "sparky/slot0";
const SPARK_NODE_ID = "sparky";

const TENANT_ORDER: TenantName[] = ["hipfire", "lemonade", "comfyui"];

function statusOf(lifecycle: LifecycleMap, key: string, fallback: LifecycleStatus): LifecycleStatus {
  return lifecycle[key]?.status ?? fallback;
}

/** Whether a tenant is currently occupying its GPU with something worth a
 * chip. An unloaded lemonade or a parked hipfire gets no chip — that empty
 * space is the point, it is where the "serving slot" dropzone shows. */
function tenantPlacement(
  name: TenantName,
  world: World,
  lifecycle: LifecycleMap,
): Placement | null {
  const key = `local/${name}`;

  if (name === "lemonade") {
    const t = world.tenants.lemonade;
    if (t.state !== "loaded" || !t.model) return null;
    return {
      id: key, name: t.model, bytes: t.footprint, engine: "lemonade",
      kind: "model", stale: false, status: statusOf(lifecycle, key, "serving"),
    };
  }

  if (name === "hipfire") {
    const t = world.tenants.hipfire;
    if (t.state === "parked" || !t.model) return null;
    return {
      id: key, name: t.model, bytes: t.footprint, engine: "hipfire",
      kind: "model", stale: false,
      status: statusOf(lifecycle, key, t.state === "loading" ? "warming" : "serving"),
    };
  }

  // ComfyUI holds VRAM whether or not it is mid-render, and has no model
  // identity of its own, so it is always present and always an "engine".
  return {
    id: key, name: "comfyui", bytes: null, engine: "comfyui",
    kind: "engine", stale: false, status: statusOf(lifecycle, key, "idle"),
  };
}

function externalPlacement(e: ExternalProc): Placement {
  return {
    id: `external/${e.pid}`,
    name: `pid ${e.pid}`,
    bytes: e.bytes,
    status: "unmanaged",
    engine: "external",
    kind: "external",
    stale: false,
  };
}

function localNode(state: StateResponse): DeckNode {
  const { world, lifecycle } = state;
  return {
    id: state.node.id,
    label: state.node.label,
    // The local box is serving this very page, so it is reachable by
    // definition. A failed poll is an App-level error, not a node state.
    status: "reachable",
    lastSeen: null,
    resources: world.gpus.map((gpu) => ({
      id: `gpu${gpu.index}`,
      label: `GPU ${gpu.index}`,
      capacity: { used: gpu.used, total: gpu.total },
      controls: TENANT_ORDER.filter((t) => world.placement[t] === gpu.index),
      placements: [
        ...TENANT_ORDER.filter((t) => world.placement[t] === gpu.index)
          .map((t) => tenantPlacement(t, world, lifecycle))
          .filter((p): p is Placement => p !== null),
        ...world.externals.filter((e) => e.gpu === gpu.index).map(externalPlacement),
      ],
    })),
  };
}

export function buildNodes(
  state: StateResponse | null,
  spark: SparkStatus | null,
): DeckNode[] {
  if (state === null) return [];
  const nodes = [localNode(state)];
  const sparkNode = buildSparkNode(state.lifecycle, spark);
  if (sparkNode) nodes.push(sparkNode);
  return nodes;
}

/** The spark half of the board.
 *
 * `spark === null` means the engine is not configured on this deployment
 * (the backend answers 503) — an ABSENT node, so no card at all. That is a
 * different thing from a configured node we cannot reach, which keeps its
 * card and its last-known placements.
 */
function buildSparkNode(lifecycle: LifecycleMap, spark: SparkStatus | null): DeckNode | null {
  if (spark === null) return null;

  const entry = lifecycle[SPARK_SLOT_KEY];
  const reachable = entry?.observed.reachable ?? true;
  const swapping = spark.swap_status?.state === "swapping";
  const endpointOk = spark.serving.endpoint_ok;

  let status: NodeStatus;
  if (!reachable) status = "unreachable";
  else if (endpointOk) status = "reachable";
  else if (swapping) status = "warming";
  else status = "down";

  const stale = status === "unreachable";
  const model = spark.serving.model;

  return {
    id: SPARK_NODE_ID,
    label: SPARK_NODE_ID,
    status,
    lastSeen: entry?.last_healthy_ts ?? null,
    resources: [
      {
        id: "slot0",
        label: "Serving slot",
        // Spark reports no VRAM figures to the deck — unknown, not zero.
        capacity: null,
        // The profile picker, which must render whether or not a model is
        // currently serving.
        controls: ["spark"],
        placements: model
          ? [
              {
                id: SPARK_SLOT_KEY,
                name: model,
                bytes: null,
                status: entry?.status ?? (endpointOk ? "serving" : "down"),
                engine: spark.profiles.find((p) => p.name === model)?.engine ?? "vllm",
                kind: "model",
                stale,
              },
            ]
          : [],
      },
    ],
  };
}

export { SPARK_NODE_ID, SPARK_SLOT_KEY };
