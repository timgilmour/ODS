/**
 * The board's data model — and the seam this whole redesign hangs off.
 *
 * Everything the board renders is a `DeckNode[]`. The local box is one GPU
 * set plus whatever engines are DECLARED on it (E1: zero, three, or five —
 * spec §5) — one card per declared resource, never a fixed lemonade/comfyui/
 * hipfire triple — plus a special-cased spark. When a real node registry
 * lands, THIS FILE should be the only one that changes. If a registry change
 * forces edits in components, the seam was drawn in the wrong place.
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
 * prescribed: a DECLARED REMOTE engine's card carries `remoteEngine`
 * (`RemoteEngineControl` below) — its own node id, its own resource name and
 * its own verbs, decided here — and leaves `controls` EMPTY, so the
 * local-world dispatch in ResourcePanel/ModelDetailDrawer can never fire for
 * it. The landmine stays dormant: no remote card reads `world`, `models` or
 * `coldGgufs` for anything.
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

export interface DeckResource {
  id: string;
  label: string;
  /** null means "capacity is unknown", which renders as a hatched track —
   * distinct from a real zero. An unreachable node reports null, never 0. */
  capacity: { used: number; total: number } | null;
  placements: Placement[];
  /** Which control surfaces belong to this resource. For a local resource
   * this is `[resource]` (its own declared name — dispatched through Task
   * 7's generic `/api/tenants/{resource}/{verb}` route); for a swap node's
   * slot it is `[SPARK_CONTROL]`; observe-only cards carry none. This
   * exists because an EMPTY resource still needs its verbs: a load-verb
   * kind's Load dropdown has to render on a resource with nothing loaded,
   * and there is no placement to hang it off. Keeping it here rather than
   * letting components re-derive it is what keeps node-specific knowledge
   * inside this file. */
  controls: string[];
  /** Set on (and only on) a DECLARED REMOTE engine's card — see
   * `RemoteEngineControl`. Absent everywhere else, so
   * `resource.remoteEngine &&` is a complete check at every call site, and
   * mutually exclusive with a non-empty `controls` by construction. */
  remoteEngine?: RemoteEngineControl;
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
 * resource's NAME — `KNOWN_KINDS` (app/engine_kinds.py:90-94) is the closed
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

function localNode(state: StateResponse): DeckNode {
  const { world, lifecycle, policy } = state;
  // One card per DECLARED resource (spec §5: zero, three, or five), ordered
  // by gpu_index then resource name — never a fixed triple, never one card
  // per physical GPU (two resources may share a GPU; each still gets its
  // own card, the design this task's brief test pins).
  const sorted = sortedResourceEntries(world.tenants);
  // An external process is attributed to the FIRST resource card on its
  // GPU (in the sorted order above) rather than every card sharing that
  // GPU — otherwise two co-located resources would each show the same
  // external pid, double-counting one fact. Best-effort either way (see
  // app/state.py's own externals docstring); this just avoids the
  // duplicate-rendering failure mode a naive per-resource filter would add.
  const externalsClaimed = new Set<number>();
  return {
    id: state.node.id,
    label: state.node.label,
    // The local box is serving this very page, so it is reachable by
    // definition. A failed poll is an App-level error, not a node state.
    status: "reachable",
    lastSeen: null,
    resources: sorted.map(([resource, tenant]) => {
      const gpu = world.gpus.find((g) => g.index === tenant.gpu_index);
      const placement = tenantPlacement(resource, tenant, lifecycle, policy);
      const claimExternals = !externalsClaimed.has(tenant.gpu_index);
      if (claimExternals) externalsClaimed.add(tenant.gpu_index);
      return {
        id: resource,
        label: resource,
        // Unknown, not zero: a resource whose declared gpu_index matches no
        // entry in the live world.gpus (a data inconsistency) reports
        // capacity unknown rather than a fabricated 0/0.
        capacity: gpu ? { used: gpu.used, total: gpu.total } : null,
        // A local resource's only control is itself (PlacementActions
        // dispatches its verbs by resource name, per Task 7's generic
        // `/api/tenants/{resource}/{verb}` route) — one element, always
        // present, even for an empty/unloaded resource: the Load dropdown
        // (or Free/Park button) has to render somewhere with no placement
        // to hang it off.
        controls: [resource],
        placements: [
          ...(placement ? [placement] : []),
          ...(claimExternals
            ? world.externals.filter((e) => e.gpu === tenant.gpu_index).map(externalPlacement)
            : []),
        ],
      };
    }),
  };
}

// ---------------------------------------------------------------------------
// Declared REMOTE engines (Task 10b) — a node-agent entry's own engines[],
// seen through `world.remote_tenants` and given a card with verbs.
// ---------------------------------------------------------------------------

/** node-agent reports MB (its `IndividualGPU`, node-agent/models.py:23-31);
 * the board's meters speak bytes (World.gpus). ONE conversion, shared by the
 * bare-GPU cards and the engine cards — two copies of a units conversion is
 * how the next units bug gets in (app/node_clients.py:54 says the same about
 * its own). `null` capacity means unknown: a declared gpu_index matching
 * nothing the node reported renders hatched, never a fabricated 0/0. */
function gpuCapacity(gpus: NodeGpu[] | null, index: number): DeckResource["capacity"] {
  const gpu = (gpus ?? []).find((g) => g.index === index);
  return gpu
    ? { used: gpu.memory_used_mb * 1024 * 1024, total: gpu.memory_total_mb * 1024 * 1024 }
    : null;
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
 * identity of its own" shape the comfyui-kind arm above takes — but its only
 * caller, `remoteEngineResources`, now gates the tenant through
 * `engineChipVisible` before ever reaching here (ruling 2026-08-18: a
 * model-less engine earns no card, `down`/`quarantined`/`warming` excepted).
 * Kept unconditional here rather than threading the filter through this
 * function too, because Task 3's per-GPU regroup is what moves the check to
 * where it belongs (a per-chip filter, not a per-tenant one) — this function
 * stays the same shape either way.
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

/** One card per declared remote engine — the remote counterpart of
 * `localNode`'s per-resource cards, and the ONLY thing on the board that
 * carries `remoteEngine`. `controls` stays empty: those dispatch against the
 * LOCAL box's world (see this file's header). */
function remoteEngineResources(
  entry: DeckNodeEntry,
  tenants: RemoteTenant[],
  state: StateResponse,
  kinds: EngineKindsResponse | null,
  stale: boolean,
): DeckResource[] {
  // Interim: ruling 2026-08-18 hides a model-less engine's whole CARD here
  // (GPU meter, verbs, everything) rather than just its chip — Task 3's
  // per-GPU regroup is what rewrites this call site to filter only the chip
  // and keep the meter. Same function as the comfyui arm above, applied to
  // the remote half of the world snapshot.
  return tenants
    .filter((tenant) =>
      engineChipVisible(
        tenant.engine, tenant.state, tenant.queue ?? null,
        state.lifecycle[remoteKey(tenant)]?.status,
      ))
    .map((tenant) => ({
      id: tenant.resource,
      label: tenant.resource,
      // The node's OWN GPU list (the node-observer's, which is what this
      // card already meters) — never `world.gpus`, which is this box's, and
      // never a second source for one fact.
      capacity: gpuCapacity(entry.gpus, tenant.gpu_index),
      controls: [],
      remoteEngine: {
        nodeId: entry.id,
        resource: tenant.resource,
        kind: tenant.engine,
        state: tenant.state,
        verbs: remoteEngineVerbs(kinds, tenant.engine, tenant.state, stale),
      },
      placements: [remoteEnginePlacement(tenant, state.lifecycle, state.policy, stale)],
    }));
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

/** A registry `node-agent` entry, rendered as its GPU meters plus one card
 * per engine it DECLARES (Task 10b). A node that declares nothing is
 * observe-only exactly as before — bare GPU cards, no controls, no
 * placements: this file has no verbs to give a box it only watches.
 *
 * A GPU carrying a declared engine gets no bare card of its own: the
 * engine's card already meters that GPU, and two meters for one GPU would
 * report the same fact twice.
 *
 * See OBSERVED_STATUS for the status mapping, and this file's header for why
 * an engine card's `controls` stays empty (the App.tsx prop-drilling
 * landmine this keeps dormant — a remote card carries `remoteEngine`
 * instead, with its own node's data). */
function observedNode(
  entry: DeckNodeEntry,
  state: StateResponse,
  kinds: EngineKindsResponse | null,
): DeckNode {
  const status = (entry.status && OBSERVED_STATUS[entry.status]) || "unreachable";
  const tenants = remoteTenantsOf(state.world, entry.id);
  const claimed = new Set(tenants.map((t) => t.gpu_index));
  return {
    id: entry.id,
    label: entry.label,
    status,
    lastSeen: entry.last_seen,
    detail: entry.error ?? undefined,
    servingLine: entry.serving?.model ?? undefined,
    resources: [
      ...(entry.gpus ?? []).filter((g) => !claimed.has(g.index)).map((g) => ({
        id: `gpu${g.index}`,
        label: `GPU ${g.index}`,
        capacity: gpuCapacity(entry.gpus, g.index),
        controls: [],     // observe-only: no verbs, no placements (spec §1) —
        placements: [],   // which is also what keeps the App.tsx prop-drilling
                          // landmine (this file's header) dormant.
      })),
      ...remoteEngineResources(entry, tenants, state, kinds, status === "unreachable"),
    ],
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

/** Whether a resource currently carries its own (non-external) placement —
 * one card has at most one, E1's per-resource-card redesign (a shared GPU's
 * external process is a SEPARATE placement on the same card, never the
 * resource's own). This is the ONE fact two independent call sites need to
 * agree on: `ResourcePanel` uses it to decide whether `PlacementActions`'
 * bare state pill has to name the resource (no chip means the control row
 * is the only thing saying what it is), and `ModelDetailDrawer` uses the
 * identical question for the same component's `hasPlacement` prop when
 * it's reopened over that resource's own model chip. Collapsed to one
 * function so the two can never independently drift on what "has a
 * placement" means — the fix-loop finding this replaced was exactly that: a
 * tautological re-derivation at one call site and a real one at the other. */
export function resourceHasOwnPlacement(resource: DeckResource): boolean {
  return resource.placements.some((p) => p.kind !== "external");
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

  return {
    id: entry.id,
    label: entry.label,
    status,
    detail,
    lastSeen: lc?.last_healthy_ts ?? null,
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
              },
            ]
          : [],
      },
      // The slot is this node's SWAP surface; these are the engines it
      // DECLARES. Separate cards, separate verbs, separate route. This card
      // renders no BARE GPU cards at all (only observedNode does), so there
      // is nothing here to suppress the way that function does — but the
      // engine card still meters its GPU from this node's own list.
      //
      // `stale` is the SLOT's reachability, which is the node-agent's own
      // (app/observe.py's observe_spark reads the same probe the declared
      // engines' observations come through), so an engine card on a dark box
      // withholds its verbs exactly as the slot withholds the profile picker.
      ...remoteEngineResources(entry, remoteTenantsOf(state.world, entry.id),
                               state, kinds, stale),
    ],
  };
}

export { SPARK_CONTROL, SPARK_DEFAULT_ENGINE };
