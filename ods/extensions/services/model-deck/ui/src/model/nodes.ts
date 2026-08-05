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
 *
 * ⚠ INVARIANT THIS FILE DEPENDS ON, stated because its failure would be
 * silent. App.tsx drills the LOCAL box's snapshot — `world`, `models`,
 * `coldGgufs` — down to every node card, remote ones included. That is safe
 * only while exactly one node is "local-like", i.e. while no other node
 * carries tenant controls. Sparky's only control is SPARK_CONTROL, which is
 * dispatched separately and never touches `world`, so nothing reads the
 * wrong host today. The moment a second box exposes lemonade/comfyui/hipfire
 * verbs, those props start describing the wrong machine and the board will
 * render plausible, wrong numbers with no error anywhere. The fix then is a
 * richer `controls` type carrying its own node's data, decided HERE — not a
 * per-component guard.
 */

import type {
  ExternalProc,
  LifecycleMap,
  LifecycleStatus,
  PolicyMap,
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

  // -- operational facts ---------------------------------------------------
  // Every one of these is OPTIONAL, and an absent field means "this
  // placement has no answer to that question" — never zero. `queue: 0` (a
  // drained ComfyUI) and "this engine has no queue at all" (hipfire) are
  // different facts, and only the first should put a number on screen.
  // They live here rather than in the components because the values come
  // from `state.policy` and per-tenant World fields, which are exactly the
  // node-specific knowledge this file exists to absorb.

  /** Exempt from idle-TTL eviction (policy). */
  pinned?: boolean;
  /** Eviction priority (policy). Lower runs first when VRAM is short. */
  priority?: number;
  /** Serving a request RIGHT NOW. hipfire only: its single admission slot
   * has in-flight requests, which is what makes park/apply refuse without
   * force — so the board can predict that refusal before the click. */
  busy?: boolean;
  /** Queue depth for engines that have one (ComfyUI). `null` means the
   * engine has a queue but the reading is unavailable; absent means the
   * engine has no queue concept. */
  queue?: number | null;
  /** Seconds since this tenant last did anything. Feeds idle-TTL eviction,
   * so it explains why something is about to be dropped. */
  idleSeconds?: number | null;
  /** The engine's name, set ONLY when it differs from what the node
   * normally runs — otherwise every spark chip would carry a redundant
   * "vllm". The comparison is made here, not in a component, so a node
   * registry can vary the default per node without touching the board. */
  engineBadge?: string;
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

/** The control surface the spark serving slot exposes — the value that ends
 * up in `DeckResource.controls`, which the board dispatches on to decide
 * whether to render the profile picker rather than a tenant's verbs.
 *
 * Deliberately NOT the same value as SPARK_NODE_ID ("sparky"). A node id and
 * a control name are different namespaces that happen to describe the same
 * box today; nothing currently ties them, and the node-registry work needs
 * one grep target for each rather than a single string doing both jobs. */
const SPARK_CONTROL = "spark";

/** What the spark normally serves. Any other engine gets a badge on the
 * chip; this one does not, because it is the unremarkable case. */
const SPARK_DEFAULT_ENGINE = "vllm";

/** Fixed display order — and the authoritative list of what a tenant
 * control name can be. Exported so components dispatch off THIS list
 * instead of keeping a second copy: a copy makes adding a tenant here
 * silently drop its verbs there. */
export const TENANT_ORDER: TenantName[] = ["hipfire", "lemonade", "comfyui"];

/** Narrows a `DeckResource.controls` entry to a tenant. `controls` is
 * deliberately `string[]` (the spark resource sets "spark", which is not a
 * tenant name), so the board needs a guard rather than a cast — a cast lets
 * an unrecognized control render as nothing at all, with no type error and
 * no failing test to catch it. */
export function isTenantName(control: string): control is TenantName {
  return (TENANT_ORDER as readonly string[]).includes(control);
}

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
  policy: PolicyMap,
): Placement | null {
  const key = `local/${name}`;
  // Policy is per-tenant and identical for every placement of that tenant,
  // so it is read once here rather than drilled to the board as a prop.
  const { pinned, priority } = policy[name];

  if (name === "lemonade") {
    const t = world.tenants.lemonade;
    if (t.state !== "loaded" || !t.model) return null;
    return {
      id: key, name: t.model, bytes: t.footprint, engine: "lemonade",
      kind: "model", stale: false, status: statusOf(lifecycle, key, "serving"),
      pinned, priority, idleSeconds: t.idle_s,
    };
  }

  if (name === "hipfire") {
    const t = world.tenants.hipfire;
    if (t.state === "parked" || !t.model) return null;
    return {
      id: key, name: t.model, bytes: t.footprint, engine: "hipfire",
      kind: "model", stale: false,
      status: statusOf(lifecycle, key, t.state === "loading" ? "warming" : "serving"),
      pinned, priority,
      // hipfire reports no idle_s at all — it has one admission slot, and
      // what matters about it is whether a turn is in flight right now.
      busy: (t.queue_depth ?? 0) > 0,
    };
  }

  // ComfyUI holds VRAM whether or not it is mid-render, and has no model
  // identity of its own, so it is always present and always an "engine".
  const comfy = world.tenants.comfyui;
  return {
    id: key, name: "comfyui", bytes: null, engine: "comfyui",
    kind: "engine", stale: false, status: statusOf(lifecycle, key, "idle"),
    pinned, priority, queue: comfy.queue, idleSeconds: comfy.idle_s,
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
  const { world, lifecycle, policy } = state;
  return {
    id: state.node.id,
    label: state.node.label,
    // The local box is serving this very page, so it is reachable by
    // definition. A failed poll is an App-level error, not a node state.
    status: "reachable",
    lastSeen: null,
    resources: world.gpus.map((gpu) => {
      // One pass, used for BOTH the control list and the placements: they
      // are the same set of tenants by definition, and computing it twice
      // is how those two answers eventually drift apart.
      const tenants = TENANT_ORDER.filter((t) => world.placement[t] === gpu.index);
      return {
        id: `gpu${gpu.index}`,
        label: `GPU ${gpu.index}`,
        capacity: { used: gpu.used, total: gpu.total },
        controls: tenants,
        placements: [
          ...tenants
            .map((t) => tenantPlacement(t, world, lifecycle, policy))
            .filter((p): p is Placement => p !== null),
          ...world.externals.filter((e) => e.gpu === gpu.index).map(externalPlacement),
        ],
      };
    }),
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
  // The backend's own boot verdict, and the ONLY reliable one. The node's
  // swap helper reports swap_status.state === "done" the moment swap.sh
  // *launches* — not when the model serves (live finding 2026-07-30,
  // recorded in app/engines/spark.py's _BOOTING_STATES comment). So for the
  // 5-15 minutes of weight load and FlashInfer autotune the state reads
  // "done" with the endpoint still down, and reading `swapping` alone called
  // a perfectly healthy boot "down": a red failure pill, the reassurance
  // banner suppressed, and a blue `warming` placement pill sitting under it.
  // app/observe.py sets observed.transitioning from spark.boot_in_flight(),
  // which weighs state AND endpoint AND recency together — the judgement
  // this line must not try to re-derive.
  const transitioning = entry?.observed.transitioning ?? false;
  // Kept as a fallback for the poll where /api/spark/status is fresher than
  // the lifecycle view (the observer caches for SPARK_OBSERVE_TTL_S), so a
  // swap the operator just fired reads as warming immediately.
  const swapping = spark.swap_status?.state === "swapping";
  const endpointOk = spark.serving.endpoint_ok;

  let status: NodeStatus;
  if (!reachable) status = "unreachable";
  else if (endpointOk) status = "reachable";
  else if (transitioning || swapping) status = "warming";
  else status = "down";

  const stale = status === "unreachable";
  const model = spark.serving.model;
  const engine = spark.profiles.find((p) => p.name === model)?.engine ?? SPARK_DEFAULT_ENGINE;

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
        controls: [SPARK_CONTROL],
        placements: model
          ? [
              {
                id: SPARK_SLOT_KEY,
                name: model,
                bytes: null,
                status: entry?.status ?? (endpointOk ? "serving" : "down"),
                engine,
                // Which engine sparky is actually serving, but only when it
                // is not the one it usually serves — a "vllm" badge on every
                // chip is noise, a "ds4" badge is the answer to a question
                // the operator would otherwise have to go and look up.
                engineBadge: engine === SPARK_DEFAULT_ENGINE ? undefined : engine,
                kind: "model",
                stale,
              },
            ]
          : [],
      },
    ],
  };
}

export { SPARK_CONTROL, SPARK_DEFAULT_ENGINE, SPARK_NODE_ID, SPARK_SLOT_KEY };
