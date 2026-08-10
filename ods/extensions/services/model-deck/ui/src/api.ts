/**
 * Model Deck API client — typed wrappers over the FastAPI backend mounted
 * under /api (see app/routers/*.py). No auth: the admin-token gate was
 * deliberately removed 2026-07-22 (ops-first; the LAN path still sits
 * behind Authelia via ods-lan).
 */

// ---------------------------------------------------------------------------
// Types — mirror app/state.py, app/sets.py, app/policy.py, app/registry.py,
// app/events.py exactly (bytes/seconds throughout, never pre-converted).
// ---------------------------------------------------------------------------

export type TenantName = "lemonade" | "comfyui" | "hipfire";

export interface Gpu {
  index: number;
  total: number;
  used: number;
  free: number;
}

export interface ExternalProc {
  pid: number;
  gpu: number;
  bytes: number;
}

export interface LemonadeTenant {
  // "loading" = a load is in flight; health reports nothing loaded while
  // weights stream in (World._snapshot_lemonade, app/state.py:143-144).
  state: "loaded" | "unloaded" | "loading" | "unknown";
  model: string | null;
  footprint: number | null;
  idle_s: number | null;
}

export interface ComfyuiTenant {
  state: "busy" | "idle" | "unknown";
  queue: number | null;
  idle_s: number | null;
}

export interface HipfireTenant {
  state: "running" | "loading" | "parked" | "unknown";
  model: string | null;
  footprint: number;
  /** In-flight requests holding hipfire's single admission slot (from the
   * daemon's /stats); null when parked/unreachable. > 0 means a
   * conversation turn is being served RIGHT NOW — park/apply will refuse
   * without force. */
  queue_depth: number | null;
}

export interface World {
  gpus: Gpu[];
  tenants: {
    lemonade: LemonadeTenant;
    comfyui: ComfyuiTenant;
    hipfire: HipfireTenant;
  };
  externals: ExternalProc[];
  default_route: string | null;
  placement: Record<TenantName, number>;
}

export interface TenantPolicy {
  priority: number;
  pinned: boolean;
  idle_ttl: number;
}

export type PolicyMap = Record<TenantName, TenantPolicy>;

export interface ModelFile {
  file: string;
  size: number;
  footprint: number;
}

/** One resource's derived status — mirrors app/lifecycle.py's STATUSES. */
export type LifecycleStatus =
  | "serving"
  | "drifted"
  | "down"
  | "parked"
  | "unexpected"
  | "unmanaged"
  | "idle"
  | "unreachable"
  | "quarantined"
  | "warming";

/** A durable intent record — mirrors app/intent.py's IntentStore.record. */
export interface IntentRecord {
  state: "loaded" | "unloaded";
  /** null means "loaded, no opinion which model" (single-model engines). */
  model: string | null;
  engine: string;
  updated_ts: string;
  last_healthy_ts: string | null;
  failures: number;
  quarantined: boolean;
}

/** One observation in app/observe.py's single shape. */
export interface Observation {
  reachable: boolean;
  loaded: boolean;
  model: string | null;
  transitioning: boolean;
}

/** Mirrors app/routers/__init__.py's build_lifecycle_view, keyed
 * `<node>/<resource>` (e.g. "local/hipfire", "sparky/slot0"). `intent` is
 * null when nothing was ever recorded — which is exactly what makes the
 * status `unmanaged`/`idle` rather than `down`/`parked`. */
export interface LifecycleEntry {
  status: LifecycleStatus;
  reason: string;
  intent: IntentRecord | null;
  observed: Observation;
  last_healthy_ts: string | null;
  settings_drift: SettingsDrift | null;
}

export type LifecycleMap = Record<string, LifecycleEntry>;

/** Identity of the box serving this UI. Mirrors app/routers/status.py's
 * `node` block; `id` matches app.observe's local-node key prefix. */
export interface NodeIdentity {
  id: string;
  label: string;
}

/** app/node_observer.py's vocabulary, verbatim (online|offline|error|
 * unconfigured); null = not yet observed (routers/status.py _nodes_block). */
export type NodeAgentStatus = "online" | "offline" | "error" | "unconfigured" | null;

/** node-agent IndividualGPU (node-agent/models.py:23-31), the fields the
 * board renders. Extra fields arrive and are ignored. */
export interface NodeGpu {
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_percent: number;
}

/** app/routers/status.py's `_nodes_block` — one entry per registered node,
 * local box included (agent_kind "local", always status "online"). */
export interface DeckNodeEntry {
  id: string;
  label: string;
  agent_kind: "local" | "node-agent";
  address: string | null;
  serving_address: string | null;
  credential_set: boolean;
  // True when the deck's ACTUATION path is still bound to configuration the
  // registry has moved past — swaps/restores keep using the boot-time
  // address until a restart, while observation already follows the edit.
  // Produced by app/routers/status.py's `_nodes_block` (via
  // app/node_binding.py's `entry_actuation_stale`); always present, and
  // always false for non-spark nodes.
  actuation_stale: boolean;
  status: NodeAgentStatus;
  last_seen: string | null;
  gpus: NodeGpu[] | null;
  serving: { model: string | null; endpoint_ok: boolean } | null;
  error: string | null;
}

export interface StateResponse {
  node: NodeIdentity;
  world: World;
  policy: PolicyMap;
  models: ModelFile[];
  /** Empty object when no intent store is wired (app/routers/__init__.py
   * returns {} rather than omitting the key). */
  lifecycle: LifecycleMap;
  /** app/routers/status.py's `_nodes_block` — optional so every existing
   * StateResponse fixture (pre-registry) still compiles unchanged. */
  nodes?: DeckNodeEntry[];
}

export interface EventEntry {
  ts: string;
  kind: string;
  detail: Record<string, unknown>;
}

// Settings & Facts (Phase 3) -------------------------------------------------

export type SettingsKind = "engines" | "models" | "engine_models"; // app/settings_store.py:117
export type Layer = "engine_defaults" | "checkpoint_recommendations"
  | "engine" | "model" | "engine_model"; // app/ladder.py:48
export type Widget = "toggle" | "list" | "select" | "number" | "text"; // app/harvest.py:widget_for
export type ArgValue = string | string[] | boolean; // app/argline.py:8-13, bare flags normalize as true end-to-end

/** app/harvest.py:parse_probe_output options[...] */
export interface CatalogOption {
  aliases: string[];
  type: string | null;
  choices: string[] | null;
  /** default: the harvested default, ALREADY DECODED for the wire by
   * app/routers/settings.py:get_catalog's `_catalog_default` — never the
   * raw `repr(action.default)` app/harvest.py stores (a string-typed
   * option's repr is e.g. `"'auto'"`, quotes included). `null` means there
   * is no default worth prefilling: the harvested "None" repr, and every
   * `_decode_harvested_default` drop shape (False/None/[]/{} — an
   * off-by-default option's honest rendering is an absent flag, not a
   * value) both decode to `null` on this route, per its own docstring
   * ("Engine-default decoding"). Anything else is the real decoded value —
   * string, number, boolean, or a list — never re-repr'd here. */
  default: string | number | boolean | string[] | null;
  nargs: unknown;
  repeatable: boolean;
  help: string;
  widget: Widget;
}

/** app/routers/settings.py:get_catalog */
export interface Catalog {
  engine_version: string;
  harvested_ts: string | null;
  options: Record<string, CatalogOption>;
}

/** app/ladder.py:resolve_settings output */
export interface ResolvedEntry {
  value: ArgValue;
  origin: "derived" | "declared";
  layer: Layer;
}

/** app/validate_settings.py:issues.append shape */
export interface SettingsWarning {
  key: string;
  "class": string;
  severity: string;
  message: string;
}

/** app/routers/settings.py:get_effective */
export interface EffectiveResponse {
  resolved: Record<string, ResolvedEntry>;
  argline: string;
  warnings: SettingsWarning[];
}

/** POST /api/settings/preview response: {parsed, argline, warnings} */
export interface SettingsPreviewResponse {
  parsed: Record<string, ArgValue>;
  argline: string;
  warnings: SettingsWarning[];
}

/** app/routers/__init__.py:291 - a single settings change entry
 * Drift folds changes across args, env, and container namespaces (app/routers/__init__.py fold loop).
 * Args values are ArgValue-shaped (string | string[] | boolean), but container values (ulimits etc.)
 * may be arbitrary nested mappings. old/new are unknown to accommodate all three namespaces. */
export interface SettingsDriftEntry {
  key: string;
  old: unknown;
  new: unknown;
  ts: string;
}

/** app/routers/__init__.py:291 - drift report on lifecycle entry */
export interface SettingsDrift {
  changed: string[];
  entries: SettingsDriftEntry[];
  since: string | null;
}

/** app/facts.py:resolve_facts */
export interface FactEntry {
  value: unknown;
  origin: "derived" | "declared";
  source: string;
  derived_ts: string | null;
  shadowed_value?: unknown;
}

/** Record<namespace, Record<key, FactEntry>> */
export type FactsMap = Record<string, Record<string, FactEntry>>;

/** app/facts.py:detect_drift entries — exactly six fields, severity always present */
export interface FactsDriftItem {
  field: string;
  expected: unknown;
  actual: unknown;
  expected_source: string;
  actual_source: string;
  severity: string;
}

// Config sets ----------------------------------------------------------------

export interface Durable {
  default_route_model: string;
  activate_model_id: string | null;
}

export interface LemonadeEphemeral {
  state: "loaded" | "unloaded";
}

export interface ComfyuiEphemeral {
  state: "free" | "leave";
  reserve_gb: number;
}

export interface HipfireEphemeral {
  state: "running" | "parked";
}

export interface Ephemeral {
  lemonade: LemonadeEphemeral | null;
  comfyui: ComfyuiEphemeral | null;
  hipfire: HipfireEphemeral | null;
}

export interface ConfigSet {
  name: string;
  notes: string;
  durable: Durable | null;
  ephemeral: Ephemeral | null;
  policy_overrides: Record<string, TenantPolicy> | null;
}

export interface SetsResponse {
  sets: ConfigSet[];
  previous: ConfigSet | null;
}

// A plan_apply() step is a small tagged dict whose extra fields vary by
// "step" kind (see app/sets.py:_execute_step) — rendered verbatim, never
// destructured field-by-field, so a plain index signature is sufficient.
export interface Step {
  step: string;
  [key: string]: unknown;
}

export interface PreviewResponse {
  steps: Step[];
  estimate_s: number;
}

export interface ApplyReport {
  completed: Step[];
  failed: Step | null;
  error: string | null;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Request core
// ---------------------------------------------------------------------------

interface ErrorBody {
  detail?: string;
}

/** Thrown by request() instead of a plain Error, so callers that need to
 * branch on the HTTP status (e.g. SetBuilder's 409-means-"exists" overwrite
 * confirm) can do so without re-parsing status out of a message string.
 * Still an Error, so every existing `err instanceof Error ? err.message :
 * ...` call site keeps working unchanged. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);

  if (!res.ok) {
    const body: ErrorBody | null = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getState(): Promise<StateResponse> {
  return request<StateResponse>("/api/state");
}

export async function getEvents(n = 50): Promise<EventEntry[]> {
  const { events } = await request<{ events: EventEntry[] }>(`/api/events?n=${n}`);
  return events;
}

export function postAction(path: string, body: unknown = {}): Promise<unknown> {
  return request<unknown>(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getSets(): Promise<SetsResponse> {
  return request<SetsResponse>("/api/sets");
}

export function previewSet(slug: string): Promise<PreviewResponse> {
  return request<PreviewResponse>(`/api/sets/${encodeURIComponent(slug)}/preview`, {
    method: "POST",
  });
}

/** POST /api/sets/{slug}/apply. Throws ApiError(409) when the hipfire
 * conversation-guard vetoes the apply (a chat is in flight or recently
 * active) — callers surface that as a "Force apply" offer and retry with
 * `force: true` rather than treating it as a hard failure. */
export function applySet(slug: string, force = false): Promise<ApplyReport> {
  return request<ApplyReport>(
    `/api/sets/${encodeURIComponent(slug)}/apply?force=${force}`,
    { method: "POST" },
  );
}

/** POST /api/sets. Throws ApiError(409) when the slug already exists and
 * `overwrite` is false — callers surface that as an inline confirm and
 * retry with `overwrite: true` rather than treating it as a hard failure. */
export function saveSet(cfgset: ConfigSet, overwrite = false): Promise<{ slug: string }> {
  return request<{ slug: string }>(`/api/sets?overwrite=${overwrite}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfgset),
  });
}

export function deleteSet(slug: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sets/${encodeURIComponent(slug)}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Spark (remote single-slot node — /api/spark/*, see app/routers/spark.py)
// ---------------------------------------------------------------------------

export interface SparkServing {
  model: string | null;
  endpoint_ok: boolean;
  container_status: string | null;
}

export interface SparkSwapStatus {
  state: "swapping" | "done" | "error";
  profile: string;
  id: string;
  message: string;
  ts: string;
}

export interface SparkProfile {
  name: string;
  engine: string; // "vllm" | "comfyui" | future engines
  health_url: string | null;
  container: string | null;
}

export interface SparkStatus {
  profiles: SparkProfile[];
  swap_status: SparkSwapStatus | null;
  serving: SparkServing;
}

/** GET /api/spark/status — null when the spark engine isn't configured
 * (503), which is how the card feature-detects whether to render at all.
 * Every other failure propagates: a configured-but-unreachable spark is a
 * real error the operator should see, not an absent card. */
export async function getSparkStatus(): Promise<SparkStatus | null> {
  try {
    return await request<SparkStatus>("/api/spark/status");
  } catch (err) {
    if (err instanceof ApiError && err.status === 503) return null;
    throw err;
  }
}

/** POST /api/spark/swap. Throws ApiError(409) for the busy-serving guard,
 * for a previous swap still booting, and for the litellm default-route
 * guard. Force overrides the first two backend-side and is ignored by the
 * third; which of them the UI actually offers a Force button for is a
 * separate, deliberate decision — see SparkSwap.tsx's docstring. */
export function sparkSwap(profile: string, force = false): Promise<unknown> {
  return request<unknown>("/api/spark/swap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, force }),
  });
}

export function putPolicy(policies: PolicyMap): Promise<PolicyMap> {
  return request<PolicyMap>("/api/policy", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(policies),
  });
}

// ---------------------------------------------------------------------------
// Settings & Facts (Phase 3) — /api/settings/*, /api/facts/*
// ---------------------------------------------------------------------------

/** GET /api/settings/catalog/{node}/{engine} — returns 200 with JSON null or object;
 * null when no such engine/node. */
export async function getCatalog(node: string, engine: string): Promise<Catalog | null> {
  const res = await fetch(`/api/settings/catalog/${encodeURIComponent(node)}/${encodeURIComponent(engine)}`);
  if (!res.ok) {
    const body: ErrorBody | null = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data === null ? null : (data as Catalog);
}

/** GET /api/settings/effective/{node}/{engine}/{model} */
export function getEffective(node: string, engine: string, model: string): Promise<EffectiveResponse> {
  const encodedModel = encodeURIComponent(model).replace(/%2F/g, '/');
  return request<EffectiveResponse>(
    `/api/settings/effective/${encodeURIComponent(node)}/${encodeURIComponent(engine)}/${encodedModel}`
  );
}

/** PUT /api/settings/{kind}/{key} with {namespace: "args", values, remove?} */
export function putSettings(
  kind: SettingsKind,
  key: string,
  namespace: "args",
  values: Record<string, ArgValue>,
  remove?: string[]
): Promise<unknown> {
  const encodedKey = encodeURIComponent(key).replace(/%2F/g, '/');
  const body: {namespace: string; values: Record<string, ArgValue>; remove?: string[]} = { namespace, values };
  if (remove) body.remove = remove;
  return request<unknown>(
    `/api/settings/${encodeURIComponent(kind)}/${encodedKey}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

/** POST /api/settings/preview with {argline} body */
export function previewParse(
  argline: string,
  ctx?: { node?: string; engine?: string; model?: string }
): Promise<SettingsPreviewResponse> {
  const body: {argline: string; node?: string; engine?: string; model?: string} = { argline };
  if (ctx?.node) body.node = ctx.node;
  if (ctx?.engine) body.engine = ctx.engine;
  if (ctx?.model) body.model = ctx.model;
  return request<SettingsPreviewResponse>("/api/settings/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** POST /api/settings/preview with {args} body */
export function previewRender(
  args: Record<string, ArgValue>,
  ctx?: { node?: string; engine?: string; model?: string }
): Promise<SettingsPreviewResponse> {
  const body: {args: Record<string, ArgValue>; node?: string; engine?: string; model?: string} = { args };
  if (ctx?.node) body.node = ctx.node;
  if (ctx?.engine) body.engine = ctx.engine;
  if (ctx?.model) body.model = ctx.model;
  return request<SettingsPreviewResponse>("/api/settings/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** POST /api/settings/harvest/{node}/{engine} */
export function postHarvest(node: string, engine: string): Promise<{ outcome: string }> {
  return request<{ outcome: string }>(
    `/api/settings/harvest/${encodeURIComponent(node)}/${encodeURIComponent(engine)}`,
    { method: "POST" }
  );
}

/** GET /api/facts */
export function getFacts(): Promise<FactsMap> {
  return request<FactsMap>("/api/facts");
}

/** GET /api/facts/drift */
export function getFactsDrift(): Promise<Record<string, FactsDriftItem[]>> {
  return request<Record<string, FactsDriftItem[]>>("/api/facts/drift");
}

/** PUT /api/facts/declared/{key} with {namespace, values, ...} body */
export function putDeclared(key: string, fields: Record<string, unknown>): Promise<unknown> {
  const encodedKey = encodeURIComponent(key).replace(/%2F/g, '/');
  return request<unknown>(`/api/facts/declared/${encodedKey}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
}

/** POST /api/spark/reload — route requires a JSON body (no default); matches sparkSwap pattern. */
export function sparkReload(): Promise<unknown> {
  return request<unknown>("/api/spark/reload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

// ---------------------------------------------------------------------------
// Small display/formatting helpers shared by 3+ components (GpuColumn's
// meter + externals rows, TenantCard's footprint) — kept here rather than
// duplicated per KISS "wait for 3+ use cases" before extracting.
// ---------------------------------------------------------------------------

export function bytesToGB(bytes: number | null | undefined, decimals = 1): string {
  if (bytes == null) return "—";
  return (bytes / 1e9).toFixed(decimals);
}

/** Shared meter fill color thresholds — used by GpuColumn (live VRAM) and
 * SetBuilder (hypothetical post-apply footprint) so the two never drift. */
export function meterFillClass(pct: number): string {
  if (pct >= 95) return "meter-fill meter-red";
  if (pct >= 80) return "meter-fill meter-amber";
  return "meter-fill meter-neutral";
}

export function truncateMiddle(text: string, max = 28): string {
  if (text.length <= max) return text;
  const half = Math.floor((max - 1) / 2);
  return `${text.slice(0, half)}…${text.slice(text.length - half)}`;
}

/**
 * Mirrors app/sets.py:slugify() exactly — lowercase, alnum runs joined by
 * single dashes. The list-sets endpoint returns ConfigSet bodies with no
 * slug field, so the UI derives the same slug the backend used to store
 * each set rather than the API echoing it back on every read.
 */
export function slugify(name: string): string {
  const matches = name.toLowerCase().match(/[a-z0-9]+/g);
  return matches ? matches.join("-") : "";
}

// Reserved slug for the auto-captured pre-apply revert snapshot (see
// app/sets.py RESERVED_SLUG) — never produced by slugify() for a
// user-authored name, so it's kept as its own constant rather than derived.
export const PREVIOUS_SLUG = "_previous";

// ---------------------------------------------------------------------------
// Storage tiering (/api/storage/*, see app/routers/storage.py) — hot/cold
// location registry, unit catalog, and move jobs. Types mirror the router's
// response shapes exactly (bytes throughout, never pre-converted).
// ---------------------------------------------------------------------------

export interface StorageLocation {
  name: string; path: string; role: "hot" | "cold";
  store_type: "gguf" | "hf" | "comfy" | "plain";
  engine: "lemonade" | "comfyui" | "none";
  watermark_gb: number | null; archive_to: string | null; readonly: boolean;
  uuid: string; available: boolean;
  free_bytes: number | null; total_bytes: number | null;
}

export interface StorageUnit {
  id: string; type: "gguf" | "hf_repo" | "comfy" | "plain";
  name: string; location: string; relpath: string; size: number; mtime: number;
  state: "resident" | "moving" | "unavailable";
  pinned: boolean; last_used: number | null;
}

export interface StorageJob {
  id: string; unit_id: string; from: string; to: string; label: string;
  state: "queued" | "copying" | "verifying" | "done" | "failed" | "cancelled";
  bytes_done: number; bytes_total: number; error: string | null; created_ts: number;
}

export interface StorageState {
  locations: StorageLocation[]; units: StorageUnit[];
  jobs: StorageJob[]; policy: { auto: boolean };
}

export function getStorageState(): Promise<StorageState> {
  return request<StorageState>("/api/storage/state");
}
export function postStorageMove(unitId: string, dest: string): Promise<{ job: StorageJob }> {
  return request<{ job: StorageJob }>("/api/storage/moves", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ unit_id: unitId, dest }),
  });
}
export function cancelStorageJob(id: string): Promise<{ cancelled: boolean }> {
  return request<{ cancelled: boolean }>(`/api/storage/moves/${encodeURIComponent(id)}`, { method: "DELETE" });
}
export function putUnitPinned(unitId: string, pinned: boolean): Promise<StorageUnit> {
  return request<StorageUnit>(`/api/storage/units/${unitId}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pinned }),
  });
}
export function putStoragePolicy(policy: { auto: boolean }): Promise<{ auto: boolean }> {
  return request<{ auto: boolean }>("/api/storage/policy", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(policy),
  });
}
export function registerLocation(spec: Omit<StorageLocation, "uuid" | "available" | "free_bytes" | "total_bytes">): Promise<StorageLocation> {
  return request<StorageLocation>("/api/storage/locations", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
}
export function updateLocation(name: string, patch: Partial<Pick<StorageLocation, "role" | "watermark_gb" | "archive_to" | "readonly">>): Promise<StorageLocation> {
  return request<StorageLocation>(`/api/storage/locations/${encodeURIComponent(name)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}
export function deleteLocation(name: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/storage/locations/${encodeURIComponent(name)}`, { method: "DELETE" });
}
export function postStorageRescan(): Promise<{ units: number }> {
  return request<{ units: number }>("/api/storage/rescan", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Node registry (/api/nodes, see app/routers/nodes.py) — CRUD + the
// test-connection probe. The credential is write-only: accepted on
// create/update, never echoed back (only `credential_set` reports whether
// one is stored).
// ---------------------------------------------------------------------------

export interface NodeTestResult {
  ok: boolean;
  error?: string;
  name?: string;
  platform?: string;
  capabilities?: string[];
  gpu_count?: number;
}

/** The CRUD wire shape — what `app/routers/nodes.py::_public` actually
 * returns for create/update, over `NodeStore.add()`/`.update()`
 * (app/node_store.py:104-139): the stored spec (id, label, agent_kind,
 * address, serving_address, added_ts) plus `credential_set`. Deliberately
 * NOT `DeckNodeEntry`: that type mirrors a DIFFERENT producer
 * (status.py's `_nodes_block`, which folds in the node-observer pass) and
 * carries status/last_seen/gpus/serving/error — none of which this route
 * ever puts on the wire. Two producers, two types, so a `.status` read off
 * a createNode/updateNode result fails to compile instead of silently
 * resolving `undefined`. */
export interface NodeRegistryEntry {
  id: string;
  label: string;
  agent_kind: "local" | "node-agent";
  // Optional, not just nullable: `_public` spreads the stored dict as-is,
  // and the seeded local node is stored with NO address/serving_address
  // key at all (node_store.py:179's seed spec), so a PUT that never
  // touches either field returns a response missing both keys.
  address?: string | null;
  serving_address?: string | null;
  added_ts: string;
  credential_set: boolean;
  // Same field, same producer as DeckNodeEntry's — app/routers/nodes.py's
  // `_public` calls the same app/node_binding.py rule the state block does.
  actuation_stale: boolean;
}

export function createNode(body: {
  id: string; label: string; address: string;
  serving_address?: string | null; credential?: string;
}): Promise<NodeRegistryEntry> {
  return request<NodeRegistryEntry>("/api/nodes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function updateNode(id: string, body: {
  label?: string; address?: string; serving_address?: string | null;
  credential?: string;
}): Promise<NodeRegistryEntry> {
  return request<NodeRegistryEntry>(`/api/nodes/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function deleteNode(id: string): Promise<unknown> {
  return request<unknown>(`/api/nodes/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function testNode(body: {
  node_id?: string; address?: string; credential?: string;
}): Promise<NodeTestResult> {
  return request<NodeTestResult>("/api/nodes/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
