/**
 * The board's data model — and the seam this whole redesign hangs off.
 *
 * Everything the board renders is a `DeckNode[]`. A node is its GPUs: ONE
 * card per GPU (2026-08-18 ruling — "GPU cards ALWAYS show, even empty; they
 * are the telemetry surface"), carrying that GPU's meter and stats plus
 * whatever is DECLARED on it — the local box's resources (E1: zero, three or
 * five, spec §5), a swap node's serving slot, a remote node's declared
 * engines. The card is the GPU; the declarations ride it. When a real node
 * registry lands, THIS FILE should be the only one that changes. If a
 * registry change forces edits in components, the seam was drawn in the
 * wrong place.
 *
 * Pure: no React, no fetch, no clock. Everything is a function of its inputs.
 *
 * ⚠ INVARIANT THIS FILE DEPENDS ON, stated because its failure would be
 * silent. App.tsx drills the LOCAL box's snapshot — `world`, `models`,
 * `coldGgufs` — down to every node card, remote ones included. That is safe
 * only while exactly one node is "local-like", i.e. while no other node
 * carries tenant controls. Every swap node's only control is SPARK_CONTROL,
 * which is dispatched separately and never touches `world`, so nothing reads
 * the wrong host today — a registry with N swap nodes keeps the landmine
 * dormant the same way one hardcoded sparky used to (design §7: swap nodes
 * still carry no local-world props). The moment a second box exposes local
 * engine verbs, those props start describing the wrong machine and the board
 * will render plausible, wrong numbers with no error anywhere. The fix then
 * is a richer `controls` type carrying its own node's data, decided HERE —
 * not a per-component guard.
 *
 * Task 10b is the first surface to need that fix, and takes it as
 * prescribed: a DECLARED REMOTE engine rides its GPU card as a
 * `RemoteEngineControl` (`DeckResource.remoteEngines` below) — its own node
 * id, its own resource name and its own verbs, decided here — and never as a
 * TENANT control, so the local-world dispatch in ResourcePanel/
 * ModelDetailDrawer can never fire for it. A remote card's `controls` hold
 * at most `SPARK_CONTROL`, which is dispatched separately. The landmine
 * stays dormant: no remote card reads `world`, `models` or `coldGgufs` for
 * anything.
 */

import type {
  DeckNodeEntry,
  EngineKindsResponse,
  ExternalProc,
  LifecycleMap,
  LifecycleStatus,
  NodeGpu,
  PolicyMap,
  RemoteTenant,
  ResourceTenant,
  SettingsDrift,
  SparkStatus,
  StateResponse,
} from "../api";
import { isResident, remoteEngineVerbs, type EngineVerb } from "./engineVerbs";
import { engineChipVisible } from "./chipVisibility";

/** Node-level rollup. Deliberately NOT LifecycleStatus: that describes one
 * resource's relationship to its recorded intent and stays on the placement.
 * This answers only "can I see this box, and how stale is what I'm showing" —
 * which is what decides whether the whole card desaturates. */
export type NodeStatus = "reachable" | "unreachable" | "warming" | "down";

export interface Placement {
  /** Lifecycle key where one exists ("local/hipfire", "boxa/slot0"), else
   * a synthetic id. Stable across polls, so it is safe as a React key. */
  id: string;
  /** Model identity, verbatim. Never prettified — truncate at render time. */
  name: string;
  /** Spark only: the swap PROFILE that names this placement in the settings
   * and facts vocabularies. Distinct from `name`, which is what the endpoint
   * currently SERVES (`--served-model-name`) and is display-only — mm27b
   * serves as "aeon", and the two coincide often enough that keying on
   * `name` looks correct until it silently isn't.
   *
   * Settings/facts are keyed by profile: app/routers/settings.py:293 writes
   * `identities[meta["name"]]` from a profiles[] entry, and
   * app/observe.py:180-184 states the rule outright ("Identity is the
   * PROFILE the node last swapped to, not the served model name").
   * Undefined for every non-spark placement, and for a spark node that has
   * not reported a swap — callers fall back to `name`. */
  profile?: string;
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
  /** The engine's name, always present on model chips (ruling 2026-08-18 —
   * previously set only when it differed from what the node normally runs,
   * so a spark chip's "vllm" stayed silent; that special-casing is gone). */
  engineBadge?: string;
  /** Declared settings written more recently than this placement's intent
   * last (re)launched it — copied verbatim from the lifecycle view's own
   * `settings_drift` (app/routers/__init__.py:116's `_settings_drift`,
   * mirrored as `SettingsDrift`). `undefined` (never `null`) means no
   * drift, so `placement.settingsDrift &&` is a complete check at every call
   * site. */
  settingsDrift?: SettingsDrift;
}

/** The control surface a DECLARED REMOTE engine's card exposes (Task 10b) —
 * the "richer controls type carrying its own node's data" this file's header
 * prescribes, and the reason a remote card's `controls` stays empty.
 *
 * Everything a verb click needs is HERE, so no component has to join a
 * remote card back against the local box's `world`: the node id and resource
 * that address `POST /api/nodes/{node_id}/engines/{resource}/{verb}`
 * (app/routers/serving.py:247-357's `engine_verb`), and the verb list itself.
 */
export interface RemoteEngineControl {
  nodeId: string;
  /** The DECLARED resource name, verbatim — the second path segment of the
   * verb route and of this engine's lifecycle key. */
  resource: string;
  /** The declared KIND (`app.engine_kinds.KNOWN_KINDS`), as the world record
   * stamps it (`obs["engine"] = kind`, app/state.py:303). What the verb
   * vocabulary was looked up by; also what the chip badges. */
  kind: string;
  /** The ENGINE's own observed state word, verbatim — for sglang-omni
   * "busy"|"idle"|"down"|"unknown" (app/engine_kinds.py:899-947). A
   * DIFFERENT fact from the placement's `status`, which is the lifecycle
   * vocabulary (intent x observation): "serving" covers both busy and idle,
   * and only this one says a render is in flight RIGHT NOW. Rendered
   * verbatim — no "unknown busy" presentation is invented here, because the
   * unavailable-indicator case has already been folded into this word by the
   * adapter (`busy is None or busy > 0` -> "busy", :902-908). */
  state: string;
  /** From the kind's own `human_verbs` (model/engineVerbs.ts). Empty until
   * `/api/engine-kinds` lands — the card renders regardless. */
  verbs: EngineVerb[];
}

/** A GPU's live readings, as the GPU Monitor format shows them (design §D):
 * hardware name, utilization, temperature, power. DISPLAY GARNISH — the
 * meter's `capacity` stays the authoritative memory fact (backend-observed
 * bytes), and nothing here is ever an actuation input.
 *
 * Every field is `number | null` rather than optional: the board renders
 * exactly one thing for "no reading" ("—", GPUCard.jsx's own unavailable
 * pattern), so folding absent and null together HERE means no component has
 * to check two of them. `null` is a reading the producer did not take; the
 * whole object being absent means the node reported no telemetry at all. */
export interface GpuStats {
  name: string | null;
  utilizationPercent: number | null;
  temperatureC: number | null;
  powerW: number | null;
}

/** One GPU's card — the board's unit since the 2026-08-18 per-GPU ruling.
 * Named `DeckResource` still because a card is what a placement lands ON;
 * what changed is that the thing is a GPU rather than a declared resource. */
export interface DeckResource {
  /** `gpu<index>` for a GPU card (`slot0` only for a swap node that reported
   * no GPUs at all — see `buildSwapNode`). React key, never a lifecycle key:
   * PLACEMENT ids are what the lifecycle view, the drawer and drift
   * targeting join on, and those are unchanged by the per-GPU regroup. */
  id: string;
  label: string;
  /** null means "capacity is unknown", which renders as a hatched track —
   * distinct from a real zero. An unreachable node reports null, never 0. */
  capacity: { used: number; total: number } | null;
  /** This GPU's telemetry, when its node reported any. Absent means none
   * arrived (dashboard-api down for the local box, an older node-agent, a
   * test fixture with no `nodes` block) — so `resource.stats &&` is a
   * complete check at every call site. */
  stats?: GpuStats;
  placements: Placement[];
  /** Which control surfaces belong to this card. For a local GPU card these
   * are the resources DECLARED on that GPU, in board order (`sortedResource
   * Entries`) — dispatched through Task 7's generic
   * `/api/tenants/{resource}/{verb}` route; ONE card can carry several now
   * (two resources sharing a GPU), which is what `controlHasPlacement`
   * exists to ask per-control. A swap node's first card carries
   * `[SPARK_CONTROL]`; observe-only cards carry none. This exists because an
   * EMPTY resource still needs its verbs: a load-verb kind's Load dropdown
   * has to render with nothing loaded, and there is no placement to hang it
   * off. Keeping it here rather than letting components re-derive it is what
   * keeps node-specific knowledge inside this file. */
  controls: string[];
  /** The DECLARED REMOTE engines whose chips ride this GPU card — see
   * `RemoteEngineControl`. Present (possibly empty) on every card built for
   * a registry node-agent entry, absent on the local box's cards, where
   * nothing remote is ever declared. Only engines whose chip is VISIBLE
   * appear; the rest are the node's `hiddenEngines`. */
  remoteEngines?: RemoteEngineControl[];
}

export interface DeckNode {
  id: string;
  label: string;
  status: NodeStatus;
  lastSeen: string | null;
  resources: DeckResource[];
  /** The backend's own sentence about why this node is in the state it is —
   * never a phrase invented here. It is what a `down` node's banner shows,
   * which is the difference between "a swap failed and here is the helper's
   * error" and a red pill with no explanation anywhere on screen. Absent
   * when the backend offered no reason. */
  detail?: string;
  /** The model the node reports serving, when the deck holds no placement
   * for it — observe-only cards; placement-bearing nodes leave it unset. */
  servingLine?: string;
  /** The agent's own sentence that its config blinds serving detection (probe URL unset) — amber wants-a-decision, not failure. */
  warning?: string;
  /** The engines DECLARED on this node whose chip the visibility rule hides
   * (nothing loaded, no failure to show) — everything the board would
   * otherwise render nowhere at all, which is how a declared-but-unloaded
   * engine became unloadable. Present (possibly empty) on every node built
   * from a registry node-agent entry; absent on the local node, which
   * declares nothing remote. Task 5's node-header "Load engine…" menu is
   * the surface: the verbs travel with the entry, already disabled when the
   * node is dark. */
  hiddenEngines?: RemoteEngineControl[];
}

/** The control surface every swap node's serving slot exposes — the value
 * that ends up in `DeckResource.controls`, which the board dispatches on to
 * decide whether to render the profile picker rather than a tenant's verbs.
 *
 * Deliberately NOT a node id. A node id is declared per registry entry
 * (e.g. "boxa") and a control name is a different namespace that happens to
 * name the same kind of surface; nothing ties them, which is what lets N
 * swap nodes share one control name without colliding. */
const SPARK_CONTROL = "spark";

/** The engine `buildSwapNode` assumes when a served profile matches none of
 * the node's reported `profiles[]` (its join-miss fallback). No longer
 * special-cased on the chip — `engineBadge` is unconditional now (ruling
 * 2026-08-18) — so this constant's only remaining job is that fallback. */
const SPARK_DEFAULT_ENGINE = "vllm";

function statusOf(lifecycle: LifecycleMap, key: string, fallback: LifecycleStatus): LifecycleStatus {
  return lifecycle[key]?.status ?? fallback;
}

/** `lifecycle[key].settings_drift`, `null` folded to `undefined` so
 * `Placement.settingsDrift` stays an absent-means-none optional rather than
 * every caller re-checking two different "nothing here" values. */
function driftOf(lifecycle: LifecycleMap, key: string): SettingsDrift | undefined {
  return lifecycle[key]?.settings_drift ?? undefined;
}

/** Whether a declared resource is currently occupying its GPU with
 * something worth a chip. An unloaded lemonade-kind or a parked
 * hipfire-kind resource gets no chip — that empty space is the point, it is
 * where the "serving slot" dropzone shows.
 *
 * Branches on `tenant.engine` (the resource's declared KIND), never on the
 * resource's NAME — `KNOWN_KINDS` (app/engine_kinds.py:177-192) is the closed
 * backend enum this mirrors; a resource can be named anything. Adding a
 * kind there needs a branch here too, same as `PlacementActions.tsx` and
 * `ui/src/model/setDraft.ts`'s `KIND_DRAFT_SPEC`. */
function tenantPlacement(
  resource: string,
  tenant: ResourceTenant,
  lifecycle: LifecycleMap,
  policy: PolicyMap,
): Placement | null {
  const key = `local/${resource}`;
  // Policy is per-resource and identical for every placement of that
  // resource, so it is read once here rather than drilled to the board as
  // a prop.
  const { pinned, priority } = policy[resource];

  if (tenant.engine === "lemonade") {
    if (tenant.state !== "loaded" || !tenant.model) return null;
    return {
      id: key, name: tenant.model, bytes: tenant.footprint ?? null, engine: "lemonade",
      engineBadge: "lemonade",
      kind: "model", stale: false, status: statusOf(lifecycle, key, "serving"),
      pinned, priority, idleSeconds: tenant.idle_s ?? null, settingsDrift: driftOf(lifecycle, key),
    };
  }

  if (tenant.engine === "hipfire") {
    if (tenant.state === "parked" || !tenant.model) return null;
    return {
      id: key, name: tenant.model, bytes: tenant.footprint ?? null, engine: "hipfire",
      engineBadge: "hipfire",
      kind: "model", stale: false,
      status: statusOf(lifecycle, key, tenant.state === "loading" ? "warming" : "serving"),
      pinned, priority,
      // hipfire reports no idle_s at all — it has one admission slot, and
      // what matters about it is whether a turn is in flight right now.
      busy: (tenant.queue_depth ?? 0) > 0,
      settingsDrift: driftOf(lifecycle, key),
    };
  }

  // Any other kind (comfyui today) holds VRAM whether or not it is
  // mid-render, and has no model identity of its own, so when it earns a
  // chip at all it is always an "engine" — named by the RESOURCE now (the
  // old code hardcoded the literal "comfyui" here, back when resource name
  // and kind name were the same fact by construction).
  //
  // Ruling 2026-08-18: an engine with nothing to show earns no chip at all
  // (chipVisibility.ts's `engineChipVisible`) — except a failure/transition
  // status, which must stay visible. comfyui's "busy"/queue>0 is its own
  // "something's happening" reading, the opposite of sglang-omni's
  // busy|idle-is-resident one (see chipVisibility.ts's comment).
  if (!engineChipVisible(tenant.engine, tenant.state, tenant.queue ?? null, lifecycle[key]?.status)) {
    return null;
  }
  return {
    id: key, name: resource, bytes: null, engine: tenant.engine,
    kind: "engine", stale: false, status: statusOf(lifecycle, key, "idle"),
    pinned, priority, queue: tenant.queue ?? null, idleSeconds: tenant.idle_s ?? null,
    settingsDrift: driftOf(lifecycle, key),
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

/** Canonical declared-resource ordering: gpu_index then resource name —
 * never a fixed triple, never GPU-index alone (two resources sharing a GPU
 * still need a deterministic tiebreak). This is THE board order (spec §5:
 * zero, three, or five cards, this sequence) — `setDraft.ts`'s
 * `pickLoadResource` and `SetBuilder.tsx`'s card list both import this
 * rather than keeping their own copy, so the draft always lists resources
 * in the same order the live board does and a future ordering change can't
 * land in one place and not the others (fix-loop finding: SetBuilder's
 * model-library target pick used to be plain unsorted `Object.entries`,
 * nondeterministic whenever two lemonade-kind resources were declared). */
export function sortedResourceEntries(
  tenants: Record<string, ResourceTenant>,
): [string, ResourceTenant][] {
  return Object.entries(tenants).sort(
    ([nameA, a], [nameB, b]) => a.gpu_index - b.gpu_index || nameA.localeCompare(nameB),
  );
}

/** The GPU indices a node shows a card for: every GPU it REPORTS, plus every
 * index something is DECLARED on that the report does not cover. The union
 * rather than the report alone because a declaration whose gpu_index matches
 * nothing live is a data inconsistency, and dropping the card would take the
 * resource's CONTROLS off the board with it — the operator would lose the
 * verbs for a resource the deck still manages. That card meters `null`
 * (unknown), never a fabricated 0/0. */
function cardIndices(reported: number[], declared: number[]): number[] {
  return [...new Set([...reported, ...declared])].sort((a, b) => a - b);
}

/** One GPU's telemetry, from whichever `nodes[]` entry owns it (`NodeGpu` in
 * api.ts documents the one-shape-for-every-node rule). `undefined` when the
 * node reported nothing for that index at all — distinct from a row whose
 * individual readings are null, and rendered differently (no stats block vs
 * "—" per field).
 *
 * Every field is `?? null`: the wire's absent and its explicit null are the
 * same fact here ("no reading"), and collapsing them once, here, is what
 * lets `GpuStats` be four plain nullable numbers.
 *
 * The AVAILABILITY FOLD happens here too, and it is the reason this cannot
 * be a plain field copy. Both producers declare `temperature_c` and
 * `utilization_percent` REQUIRED (dashboard-api/models.py:129-143,
 * node-agent/models.py:23-37), so a sensor they could not read still has to
 * put a number on the wire: they send 0 and say so in the matching flag
 * (dashboard-api/gpu.py:172 — `temperature_available=temp > 0`,
 * `utilization_available=bool(gpu_busy_str)`). Reading the number without
 * the flag renders a failed sensor as a real 0 °C / 0 % — a fabricated
 * reading, which is the one thing this whole file's null-vs-zero discipline
 * exists to prevent. `=== false` deliberately, not falsy: an older producer
 * predating the flags omits them, and absent means "no verdict offered",
 * which both models' own `= True` defaults read as available.
 *
 * `power_w` needs no flag — it is genuinely `Optional[float]` upstream, so
 * its absence already says it. */
function statsOf(gpus: NodeGpu[] | null | undefined, index: number): GpuStats | undefined {
  const gpu = (gpus ?? []).find((g) => g.index === index);
  if (!gpu) return undefined;
  return {
    name: gpu.name ?? null,
    utilizationPercent:
      gpu.utilization_available === false ? null : (gpu.utilization_percent ?? null),
    temperatureC: gpu.temperature_available === false ? null : (gpu.temperature_c ?? null),
    powerW: gpu.power_w ?? null,
  };
}

function localNode(state: StateResponse): DeckNode {
  const { world, lifecycle, policy } = state;
  // Declared resources in board order (spec §5: zero, three, or five),
  // gpu_index then resource name — the order they take on their GPU's card.
  const sorted = sortedResourceEntries(world.tenants);
  // The LOCAL registry entry's telemetry: app/telemetry.py's pass-through of
  // dashboard-api's `/api/gpu/detailed`, attached by app/routers/status.py's
  // `_nodes_block` local arm. `world.gpus` stays the capacity source (bytes,
  // the deck's own observation) — this is the display garnish beside it, and
  // reading it off the same block every remote node uses is what keeps ONE
  // telemetry shape on the wire.
  const reported = (state.nodes ?? []).find((e) => e.agent_kind === "local")?.gpus ?? null;
  // CROSS-SENSOR GUARD. The join below is POSITIONAL — a card's stats are
  // looked up by the index its `world.gpus` capacity came from — and the two
  // lists are taken by DIFFERENT sensors: `world.gpus` is the deck's own
  // sysfs read (app/gpu.py's read_gpus), the telemetry is dashboard-api's
  // rocm-smi/nvidia-smi enumeration. `app/telemetry.py`'s `_qualified`
  // aligns the one divergence we know about (it drops sub-MIN_VRAM_BYTES
  // cards and re-sequences, so an iGPU cannot shift the numbering), but
  // alignment is not proof: neither list carries anything the other can be
  // checked against, and a card the deck sees and dashboard-api does not
  // (driver reset, a card in a state one reader skips) would silently slide
  // every subsequent GPU's temperature and power onto its neighbour.
  //
  // Equal LENGTHS is the one falsifiable check available, so a mismatch
  // drops the local telemetry wholesale — no stats beats misattributed
  // stats, and a missing stats block is a state the board already renders
  // ("—" per field). It cannot catch a same-length permutation; `uuid` is
  // carried through the pass-through precisely so a future join can verify
  // identity rather than count. Local only: a remote node's capacity and
  // stats come from ONE list (`entry.gpus`), so there is no second
  // enumeration there to disagree with.
  const telemetry =
    reported && reported.length === world.gpus.length ? reported : null;
  return {
    id: state.node.id,
    label: state.node.label,
    // The local box is serving this very page, so it is reachable by
    // definition. A failed poll is an App-level error, not a node state.
    status: "reachable",
    lastSeen: null,
    resources: cardIndices(
      world.gpus.map((g) => g.index),
      sorted.map(([, tenant]) => tenant.gpu_index),
    ).map((index) => {
      const gpu = world.gpus.find((g) => g.index === index);
      // Every resource declared on THIS GPU. The GPU name rides `stats`;
      // composing it into a title is a rendering concern, not this file's.
      const declared = sorted.filter(([, tenant]) => tenant.gpu_index === index);
      return {
        id: `gpu${index}`,
        label: `GPU ${index}`,
        capacity: gpu ? { used: gpu.used, total: gpu.total } : null,
        stats: statsOf(telemetry, index),
        // Always present, even for an empty/unloaded resource: the Load
        // dropdown (or Free/Park button) has to render somewhere with no
        // placement to hang it off.
        controls: declared.map(([resource]) => resource),
        placements: [
          ...declared
            .map(([resource, tenant]) => tenantPlacement(resource, tenant, lifecycle, policy))
            .filter((p): p is Placement => p !== null),
          // An external process belongs to its GPU, full stop. The old
          // per-resource board had to pick ONE of the cards sharing a GPU to
          // avoid double-counting one pid; with one card per GPU that
          // bookkeeping has nothing left to decide.
          ...world.externals.filter((e) => e.gpu === index).map(externalPlacement),
        ],
      };
    }),
  };
}

// ---------------------------------------------------------------------------
// Declared REMOTE engines (Task 10b) — a node-agent entry's own engines[],
// seen through `world.remote_tenants` and given a card with verbs.
// ---------------------------------------------------------------------------

/** node-agent reports MB (its `IndividualGPU`, node-agent/models.py:23-37);
 * the board's meters speak bytes (World.gpus). ONE conversion, shared by the
 * bare-GPU cards and the engine cards — two copies of a units conversion is
 * how the next units bug gets in (app/node_clients.py:54 says the same about
 * its own). `null` capacity means unknown: a declared gpu_index matching
 * nothing the node reported renders hatched, never a fabricated 0/0. */
function gpuCapacity(gpus: NodeGpu[] | null, index: number): DeckResource["capacity"] {
  const gpu = (gpus ?? []).find((g) => g.index === index);
  // `memory_usage_available: false` is the producer saying it could NOT read
  // this card's VRAM, and the required `memory_used_mb` beside it is filler
  // (the same 0-plus-flag convention `statsOf` folds — dashboard-api/gpu.py's
  // GPUInfo). Metering that 0 would draw a confident empty bar for a GPU
  // nobody measured, on the one number this card treats as authoritative.
  // Unknown capacity renders hatched, which is the honest answer.
  if (!gpu || gpu.memory_usage_available === false) return null;
  return { used: gpu.memory_used_mb * 1024 * 1024, total: gpu.memory_total_mb * 1024 * 1024 };
}

/** app/observe.py:31's `node_key` — the one backend function that builds
 * these ids, mirrored here so a remote placement's id IS its lifecycle key.
 * Built from the tenant's OWN `node_id`/`resource` fields rather than from
 * the `world.remote_tenants` map key, exactly as `observe_remote`
 * (app/observe.py:145-146) builds the observation keys it must match: two
 * derivations of one id is how a board and a lifecycle map come apart. */
function remoteKey(tenant: RemoteTenant): string {
  return `${tenant.node_id}/${tenant.resource}`;
}

/** The status a remote engine shows when the lifecycle view carries no entry
 * for it at all — reachable ONLY on a deck with no intent store, where
 * `build_lifecycle_view` returns `{}` wholesale (app/routers/__init__.py:136-137,
 * mirrored in api.ts's `StateResponse.lifecycle`). No intent store means no
 * intent by construction, so these are `derive_status`'s own no-intent arms,
 * not a second status derivation: unreachable when we failed to look
 * (app/lifecycle.py:45-46 — `unknown()` observes `reachable: False`,
 * app/observe.py:320-321), `unmanaged` for something loaded nobody asked for
 * (:57-59), `idle` for nothing loaded and nothing intended (:60). */
function noIntentStatus(state: string): LifecycleStatus {
  if (state === "unknown") return "unreachable";
  return isResident(state) ? "unmanaged" : "idle";
}

/** One declared remote engine's chip.
 *
 * Unconditional in THIS function — same "always an engine, no model
 * identity of its own" shape the comfyui-kind arm above takes. Its caller
 * (`remoteNodeCards`) splits the node's tenants through `engineChipVisible`
 * FIRST and only builds a chip for the visible ones (ruling 2026-08-18: a
 * model-less engine earns no chip, `down`/`quarantined`/`warming` excepted).
 * That split is the single application of the visibility rule on the remote
 * half: hidden engines never reach here, and never reach a card at all.
 *
 * `settingsDrift` is deliberately NOT carried: DriftCard's Settings target
 * is the NODE's configurable engine (a swap node's vllm — App.tsx's catalog
 * probe), which is not this engine, so the card would offer a scope key
 * nothing resolves — the D11 defect ModelDetailDrawer's own comment names.
 * Drift for a remote declared engine needs a per-placement settings target
 * first; until then, showing nothing beats showing a dead end.
 */
function remoteEnginePlacement(
  tenant: RemoteTenant,
  lifecycle: LifecycleMap,
  policy: PolicyMap,
  stale: boolean,
): Placement {
  const key = remoteKey(tenant);
  // PolicyStore rows are keyed by BARE resource across the whole deck
  // (ruling R10 — app/policy.py's `declared_defaults` walks EVERY registry
  // entry and the declaration boundary refuses a name another node already
  // uses), so this is the same map the local cards read. Optional-chained
  // unlike the local arms': the local declaration and its policy rows are
  // materialized together, but a remote engine declared seconds ago can be
  // observed by a poll that read policy.json a moment earlier — an absent
  // row must not blank the whole board.
  const row = policy[tenant.resource];
  return {
    id: key,
    // The RESOURCE is the identity: this kind reports no model of its own
    // (GF2 — its `/model` is the container's mount path, so the adapter
    // returns None, app/engine_kinds.py:920-927).
    name: tenant.resource,
    // No footprint on the wire for a remote engine (the declared constant
    // lives backend-side, in `reclaimable`) — absent, never invented.
    bytes: null,
    engine: tenant.engine,
    // Unlike a spark chip's badge, this one is unconditional: a remote node
    // has no "engine it usually runs" for the kind to be unremarkable
    // against, and the kind is the one thing distinguishing this card from
    // the local ones.
    engineBadge: tenant.engine,
    kind: "engine",
    stale,
    status: statusOf(lifecycle, key, noIntentStatus(tenant.state)),
    pinned: row?.pinned,
    priority: row?.priority,
    // NO `busy`, and no `queue`. Not because the fact is unknown — the
    // engine's own state word carries it, and `RemoteEngineControl.state`
    // renders it beside the verbs — but because both of those chip fields
    // come with hipfire's/comfyui's copy attached: `labels.inUseTitle` says
    // "a conversation turn ... park/apply will refuse without force", which
    // is that kind's guard and not this one's (its verbs are load/unload,
    // app/engine_kinds.py:966-970), and this kind has no queue concept at
    // all (its observe() emits state/busy_requests/model/idle_s, nothing
    // else). An absent field renders nothing, which is the honest answer
    // here; the in-flight fact is shown where it can be labelled truthfully.
    idleSeconds: tenant.idle_s ?? null,
  };
}

/** Every engine DECLARED on `nodeId`, in the board's canonical resource
 * order (gpu_index then resource name — `sortedResourceEntries`' rule,
 * applied to the remote half).
 *
 * `world.remote_tenants` IS the declaration as far as this file is
 * concerned: `World.snapshot_remote` iterates the registry's declared
 * engines and emits a record for every one of them, including the
 * kind's own `unknown()` record for a node it could not reach at all
 * (app/state.py:248-311). `/api/state`'s `nodes` block carries no
 * `engines[]` — that is the registry route's shape, not this one's. */
function remoteTenantsOf(world: StateResponse["world"], nodeId: string): RemoteTenant[] {
  return Object.values(world.remote_tenants ?? {})
    .filter((t) => t.node_id === nodeId)
    .sort((a, b) => a.gpu_index - b.gpu_index || a.resource.localeCompare(b.resource));
}

/** One declared remote engine's control surface. The SAME construction for
 * a visible engine (on its GPU card) and a hidden one (in the node's
 * `hiddenEngines`) — one function, so the header menu can never offer a
 * different verb list than the card would have. */
function engineControl(
  nodeId: string,
  tenant: RemoteTenant,
  kinds: EngineKindsResponse | null,
  stale: boolean,
): RemoteEngineControl {
  return {
    nodeId,
    resource: tenant.resource,
    kind: tenant.engine,
    state: tenant.state,
    verbs: remoteEngineVerbs(kinds, tenant.engine, tenant.state, stale),
  };
}

/** THE visibility question for the remote half, asked exactly once per
 * tenant (in `remoteNodeCards`) — `engineChipVisible` is the single
 * authority, and nothing downstream re-filters. */
function remoteChipVisible(tenant: RemoteTenant, lifecycle: LifecycleMap): boolean {
  return engineChipVisible(
    tenant.engine, tenant.state, tenant.queue ?? null,
    lifecycle[remoteKey(tenant)]?.status,
  );
}

/** A registry node-agent entry's GPU cards — one per GPU it reports, plus
 * any index a VISIBLE declared engine names that the report does not cover
 * (`cardIndices`) — and, beside them, the engines whose chip is hidden.
 *
 * Shared by `observedNode` and `buildSwapNode`: both are the same box seen
 * through the same probe, and the only difference between them is the
 * serving slot the swap path lays on top.
 *
 * A HIDDEN engine's gpu_index earns no card of its own, unlike a local
 * resource's: a hidden engine contributes no chip and no controls, so a card
 * conjured for it would be an empty meter for a GPU the node never reported.
 * The engine is not lost — it is in `hiddenEngines`, with its verbs. */
function remoteNodeCards(
  entry: DeckNodeEntry,
  state: StateResponse,
  kinds: EngineKindsResponse | null,
  stale: boolean,
): { resources: DeckResource[]; hiddenEngines: RemoteEngineControl[] } {
  const tenants = remoteTenantsOf(state.world, entry.id);
  // ONE partition pass, because `remoteChipVisible`'s own docstring promises
  // the question is asked exactly once per tenant — a second `filter` for the
  // hidden half would evaluate it twice and make that claim false, which is
  // how "the single authority" quietly becomes two evaluations that a future
  // stateful/expensive rule could disagree between.
  const visible: RemoteTenant[] = [];
  const hidden: RemoteTenant[] = [];
  for (const tenant of tenants) {
    (remoteChipVisible(tenant, state.lifecycle) ? visible : hidden).push(tenant);
  }
  const resources = cardIndices(
    (entry.gpus ?? []).map((g) => g.index),
    visible.map((t) => t.gpu_index),
  ).map((index) => {
    const onGpu = visible.filter((t) => t.gpu_index === index);
    return {
      id: `gpu${index}`,
      label: `GPU ${index}`,
      // The node's OWN GPU list (the node-observer's) — never `world.gpus`,
      // which is this box's, and never a second source for one fact.
      capacity: gpuCapacity(entry.gpus, index),
      stats: statsOf(entry.gpus, index),
      // Empty: a tenant control dispatches against the LOCAL box's world
      // (this file's header). `buildSwapNode` adds SPARK_CONTROL, which does
      // not.
      controls: [],
      remoteEngines: onGpu.map((t) => engineControl(entry.id, t, kinds, stale)),
      placements: onGpu.map((t) =>
        remoteEnginePlacement(t, state.lifecycle, state.policy, stale)),
    };
  });
  return {
    resources,
    hiddenEngines: hidden.map((t) => engineControl(entry.id, t, kinds, stale)),
  };
}

/** app/node_observer.py's status vocabulary -> the board's. `unconfigured`
 * and null both render unreachable; the backend's own sentence (entry.error)
 * carries the difference — never a phrase invented here. */
const OBSERVED_STATUS: Record<string, NodeStatus> = {
  online: "reachable",
  offline: "unreachable",
  error: "down",
  unconfigured: "unreachable",
};

/** A registry `node-agent` entry, rendered as one card per GPU it reports,
 * each carrying whatever it DECLARES on that GPU (Task 10b's engines). A
 * node that declares nothing is observe-only exactly as before — bare GPU
 * cards, no controls, no placements: this file has no verbs to give a box it
 * only watches.
 *
 * A GPU carrying a declared engine keeps its card (it IS the card now): the
 * standalone engine card the old code built instead is what showed sparky's
 * omni a meter reading the serving model's VRAM.
 *
 * See OBSERVED_STATUS for the status mapping, and this file's header for why
 * an observed card's `controls` stays empty (the App.tsx prop-drilling
 * landmine this keeps dormant — a remote card carries `remoteEngines`
 * instead, with its own node's data). */
function observedNode(
  entry: DeckNodeEntry,
  state: StateResponse,
  kinds: EngineKindsResponse | null,
): DeckNode {
  const status = (entry.status && OBSERVED_STATUS[entry.status]) || "unreachable";
  const { resources, hiddenEngines } = remoteNodeCards(
    entry, state, kinds, status === "unreachable");
  return {
    id: entry.id,
    label: entry.label,
    status,
    lastSeen: entry.last_seen,
    detail: entry.error ?? undefined,
    servingLine: entry.serving?.model ?? undefined,
    warning: entry.serving?.warning ?? undefined,
    resources,
    hiddenEngines,
  };
}

/** The registry entries the board gives serving controls to. Control is
 * DECLARED (app/node_store.py) — presence of serving data never promotes a
 * node (the watcher-shaped fixture in nodes.test.ts is the proof). */
export function swapNodes(state: StateResponse | null): DeckNodeEntry[] {
  return (state?.nodes ?? []).filter((e) => e.control === "swap");
}

/** Slot-key convention mirrored from app/observe.py:slot_key — the one
 * backend function that builds these ids. */
export function isSwapSlotId(id: string): boolean {
  return id.endsWith("/slot0");
}

/** Reverse of `${entry.id}/slot0`: the node id a swap placement's id
 * belongs to (app/observe.py:slot_key builds the id the other way). */
export function nodeIdOfPlacement(id: string): string {
  return id.split("/")[0];
}

/** `kinds` is `GET /api/engine-kinds` (App.tsx fetches it once): the source
 * of every declared remote engine's verb vocabulary, never a literal here
 * (spec §5). `null` — not landed yet, or its fetch failed — renders every
 * remote card with no verbs rather than guessed ones. Defaulted so the many
 * callers/tests that have no remote engines need not pass it. */
export function buildNodes(
  state: StateResponse | null,
  servingByNode: Record<string, SparkStatus | null>,
  kinds: EngineKindsResponse | null = null,
): DeckNode[] {
  if (state === null) return [];
  const nodes = [localNode(state)];
  for (const entry of state.nodes ?? []) {
    if (entry.agent_kind !== "node-agent") continue;
    if (entry.control === "swap") {
      const swapNode = buildSwapNode(entry, state, servingByNode[entry.id] ?? null, kinds);
      nodes.push(swapNode ?? observedNode(entry, state, kinds));
      continue;
    }
    nodes.push(observedNode(entry, state, kinds));
  }
  return nodes;
}

/** Where a placement sits: the node and resource that carry it, alongside
 * the placement itself. Returned as one object because the two answers are
 * only ever wanted together (a surface acting on a placement needs the
 * resource's `controls` and the node's id), and re-deriving either from the
 * other is what makes them disagree. */
export interface PlacementSpot {
  node: DeckNode;
  resource: DeckResource;
  placement: Placement;
}

/** Re-derives a placement from a freshly built board, by id.
 *
 * The detail drawer holds a placement the operator clicked, but the board
 * underneath it keeps polling — so the drawer must read its subject out of
 * the LATEST `buildNodes` output on every tick, never out of the object it
 * was opened with. Without that, a drawer opened over a serving model keeps
 * showing "serving" long after the model was unloaded from under it, which
 * is the one thing a live status pill must never do.
 *
 * `null` means the placement is no longer on the board at all — unloaded,
 * parked, or swapped away. That is a real answer, not a lookup failure: the
 * caller keeps showing what it last knew and says so.
 */
export function findPlacement(nodes: DeckNode[], id: string): PlacementSpot | null {
  for (const node of nodes) {
    for (const resource of node.resources) {
      const placement = resource.placements.find((p) => p.id === id);
      if (placement) return { node, resource, placement };
    }
  }
  return null;
}

/** Whether ONE control on a card has its own chip there — the board's ONLY
 * "has a placement" question, and the one a control row needs: on a shared
 * GPU an unloaded resource's row is the only thing on screen naming it, and
 * a co-resident neighbour's chip must not answer for it.
 *
 * There is deliberately no card-level counterpart. `ModelDetailDrawer` used
 * to ask one about the card its open chip sits on, but that call site had
 * already proved the answer: it reaches `PlacementActions` only through a
 * control whose `local/<control>` key MATCHED the open placement's id, so any
 * such function could only answer true. It passes the literal now, and this
 * one function has a single call site with a real question behind it.
 *
 * Keyed on the PLACEMENT ID rather than on the placement's `name` or
 * `engine`: `local/<resource>` is the lifecycle key `tenantPlacement` builds
 * (app/observe.py's `node_key`), so it identifies the control exactly, while
 * a name is the MODEL's identity and matches nothing for a chipless
 * resource. External placements (`external/<pid>`) can never collide with
 * it. */
export function controlHasPlacement(resource: DeckResource, control: string): boolean {
  return resource.placements.some((p) => p.id === `local/${control}`);
}

/** One swap node's serving-slot resource — buildNodes' registry loop calls
 * this once per `control: "swap"` entry — plus a card for each engine that
 * node DECLARES (Task 10b). Both, because they are the same box: the one
 * node that swaps vLLM profiles today is also the one an sglang-omni engine
 * gets declared on, and this path is the only one a `control: "swap"` entry
 * ever takes.
 *
 * `serving === null` means no landed status yet (the fetch has not resolved,
 * or 503'd), or the backend says the node is not operable — either way there
 * is nothing to synthesize a control surface from, so the caller falls back
 * to the observe-only card rather than rendering a phantom one. That
 * fallback carries the declared engines too: they are not the slot's, and
 * their verbs do not go through the swap channel.
 */
function buildSwapNode(
  entry: DeckNodeEntry,
  state: StateResponse,
  serving: SparkStatus | null,
  kinds: EngineKindsResponse | null,
): DeckNode | null {
  if (serving === null) return null;

  const lifecycle = state.lifecycle;
  const slotKey = `${entry.id}/slot0`;
  const lc = lifecycle[slotKey];
  const reachable = lc?.observed.reachable ?? true;
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
  const transitioning = lc?.observed.transitioning ?? false;
  // Kept as a fallback for the poll where the serving fetch is fresher than
  // the lifecycle view (the observer caches for SPARK_OBSERVE_TTL_S), so a
  // swap the operator just fired reads as warming immediately.
  const swapping = serving.swap_status?.state === "swapping";
  const endpointOk = serving.serving.endpoint_ok;

  let status: NodeStatus;
  if (!reachable) status = "unreachable";
  else if (endpointOk) status = "reachable";
  else if (transitioning || swapping) status = "warming";
  else status = "down";

  const stale = status === "unreachable";
  const model = serving.serving.model;
  // The PROFILE is the identity vocabulary (see Placement.profile). It also
  // fixes the engine join: profiles[] is keyed by profile name, so matching
  // on the SERVED name found nothing whenever the two differ and silently
  // fell back to SPARK_DEFAULT_ENGINE — an "aeon" placement reporting vllm
  // while ds4 was actually serving it.
  const profile = serving.swap_status?.profile ?? null;
  const engine = serving.profiles.find((p) => p.name === (profile ?? model))?.engine
    ?? SPARK_DEFAULT_ENGINE;

  // Both candidates are sentences the BACKEND wrote, so the banner reports
  // rather than guesses. The swap helper's own message wins when the last
  // swap ended in "error" — that is the specific failure (the helper died,
  // the container never came up), and it is the thing an asynchronous swap
  // failure would otherwise leave nowhere on screen. Otherwise
  // app/lifecycle.py's `reason` for the derived status, which is already
  // phrased for a human. An "error" swap with an empty message falls
  // through to the reason rather than showing a blank explanation.
  const swapError = serving.swap_status?.state === "error" ? serving.swap_status.message : null;
  const detail = swapError || lc?.reason || undefined;

  const slotPlacement: Placement | null = model
    ? {
        id: slotKey,
        name: model,
        profile: profile ?? undefined,
        bytes: null,
        status: lc?.status ?? (endpointOk ? "serving" : "down"),
        engine,
        engineBadge: engine,
        kind: "model",
        stale,
        settingsDrift: driftOf(lifecycle, slotKey),
      }
    : null;

  // This node's GPU cards, carrying the engines it DECLARES.
  //
  // `stale` is the SLOT's reachability, which is the node-agent's own
  // (app/observe.py's observe_spark reads the same probe the declared
  // engines' observations come through), so an engine on a dark box
  // withholds its verbs exactly as the slot withholds the profile picker.
  const { resources, hiddenEngines } = remoteNodeCards(entry, state, kinds, stale);
  // A swap node that reported no GPUs (and declares no visible engine to
  // name one) still needs somewhere to put the profile picker. Deliberately
  // NOT a fabricated `gpu0`: the node named no index, and inventing one
  // would put a GPU on the board that nothing observed. It is the serving
  // slot's own card, exactly as before this wave — capacity unknown, not
  // zero.
  const cards: DeckResource[] = resources.length > 0
    ? resources
    : [{ id: "slot0", label: "Serving slot", capacity: null, controls: [], remoteEngines: [], placements: [] }];

  return {
    id: entry.id,
    label: entry.label,
    status,
    detail,
    warning: serving.serving.warning ?? undefined,
    lastSeen: lc?.last_healthy_ts ?? null,
    // The slot rides the node's FIRST card. Spark reports no gpu_index for
    // what it serves (app/engines/spark.py observes an endpoint, not a
    // device), so there is no declared index to place it by — and the one
    // live swap node is a single-GPU GB10, which is exactly the collapse
    // this wave is for. Its chip leads the card, ahead of any engine chips.
    resources: cards.map((card, i) => (i > 0 ? card : {
      ...card,
      // The profile picker, which must render whether or not a model is
      // currently serving.
      controls: [SPARK_CONTROL],
      placements: [...(slotPlacement ? [slotPlacement] : []), ...card.placements],
    })),
    hiddenEngines,
  };
}

export { SPARK_CONTROL, SPARK_DEFAULT_ENGINE };
