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

/** One declared local engine's live status, keyed by RESOURCE NAME in
 * World.tenants (E1 Task 3/11: replaces the fixed lemonade/comfyui/hipfire
 * triple — any number of any-kind declared resources now).
 *
 * `engine` and `gpu_index` are stamped uniformly on every entry regardless
 * of kind (app/state.py's World.snapshot: `obs["engine"] = kind` /
 * `obs["gpu_index"] = entry["gpu_index"]`, app/state.py:158-159). Every
 * other field is that resource's declared KIND's own
 * `app.engine_kinds.ENGINE_KINDS[kind].observe()` shape — a union of every
 * kind's fields rather than a discriminated one, because `kind` is data
 * (`KNOWN_KINDS`, app/engine_kinds.py:90-94), not a TS literal the compiler
 * can narrow on. Fields a given kind never sets are simply absent:
 * - lemonade-kind: state "loaded"|"unloaded"|"loading"|"unknown", model,
 *   footprint, idle_s (app/engine_kinds.py `_LemonadeAdapter.observe`,
 *   :211-214).
 * - comfyui-kind: state "busy"|"idle"|"unknown", queue, idle_s
 *   (`_ComfyAdapter.observe`, :478).
 * - hipfire-kind: state "running"|"loading"|"parked"|"unknown", model,
 *   footprint (never null — 0 when not running), queue_depth
 *   (`_HipfireAdapter.observe`, :613).
 */
export interface ResourceTenant {
  engine: string;
  gpu_index: number;
  state: string;
  model?: string | null;
  footprint?: number | null;
  idle_s?: number | null;
  queue?: number | null;
  /** In-flight requests holding hipfire's single admission slot (from the
   * daemon's /stats); null when parked/unreachable. > 0 means a
   * conversation turn is being served RIGHT NOW — park/apply will refuse
   * without force. */
  queue_depth?: number | null;
}

export interface World {
  gpus: Gpu[];
  /** Keyed by resource name (app/state.py's World.snapshot `tenants[resource]
   * = obs`) — never a fixed lemonade/comfyui/hipfire shape. Empty when
   * nothing is declared (spec §1: absence is representable). */
  tenants: Record<string, ResourceTenant>;
  externals: ExternalProc[];
  default_route: string | null;
  /** resource -> declared GPU index (app/state.py's World.snapshot
   * `"placement": {resource: entry["gpu_index"] ...}`) — redundant with
   * each tenant's own `gpu_index` field; kept because it is still on the
   * wire, even though `nodes.ts` reads `gpu_index` off the tenant directly. */
  placement: Record<string, number>;
}

export interface TenantPolicy {
  priority: number;
  pinned: boolean;
  idle_ttl: number;
}

export type PolicyMap = Record<string, TenantPolicy>;

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
  /** Who authored this record (app/intent.py's VALID_ACTORS). Optional: a
   * pre-upgrade intent.json has no such key, and every backend reader treats
   * a missing one as "operator" (app/routers/control.py's supersession
   * check). Carried on /api/state's lifecycle map (app/routers/__init__.py's
   * raw intent record, delivered by app/routers/status.py). */
  actor?: "operator" | "deck";
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
 * `<node>/<resource>` (e.g. "local/hipfire", "boxa/slot0"). `intent` is
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
  /** Declared operability (app/node_store.py _CONTROLS): "swap" nodes get
   * serving verbs and a board card with controls; "none" nodes are
   * observe-only regardless of what data they carry — declared, never
   * inferred. */
  control: "none" | "swap";
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
// app/argline.py:8-13, bare flags normalize as true end-to-end. The list arm
// admits null because a null INSIDE a list survives ladder resolution
// (app/argline.py:151-154 — legacy poison the wire refuses NEW instances of,
// but persisted stores may still hold) and GET /settings/effective serves it
// verbatim; a type that cannot model the wire hides the poison from every
// consumer instead of surfacing it.
export type ArgValue = string | (string | null)[] | boolean;

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

/** One resource's drafted goal (app/sets.py's `ResourceDesired`). Which
 * `desired` values a resource's declared KIND actually accepts is verb
 * membership in that kind's `human_verbs()` crossed with app/sets.py's
 * `_DESIRED_VERBS` table (:129-134) — never a hardcoded kind name there,
 * mirrored the same verb-generic way in `ui/src/model/setDraft.ts`'s
 * `KIND_DRAFT_SPEC`. `model` is accepted only when the kind supports
 * "load" (today: lemonade-kind alone) — app/sets.py:203-204. */
export interface ResourceDesired {
  desired: "loaded" | "unloaded" | "parked" | "freed";
  model?: string | null;
}

/** app/sets.py's `Ephemeral` — resource -> desired state. An ABSENT
 * resource means "don't touch it" (E1 Task 8: replaces the old fixed
 * lemonade/comfyui/hipfire sub-sections — any number of any-kind DECLARED
 * resources now, keyed by name, not by kind). */
export interface Ephemeral {
  resources: Record<string, ResourceDesired>;
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

/** One pydantic validation error, as the app-wide RequestValidationError
 * handler emits them (app/main.py's handler, which strips the `input` key
 * from every item — see the nodes router's write-only-credential note). */
interface ValidationItem {
  type?: string;
  loc?: unknown[];
  msg?: string;
}

interface ErrorBody {
  /** A plain string for HTTPException and every hand-raised error; a LIST of
   * ValidationItem for a 422 from the validation handler. Typing it as
   * `string` alone was not merely imprecise — the list then reached
   * `new ApiError(status, detail)` as an array and rendered
   * "[object Object]" to the operator [max-review c32]. */
  detail?: string | ValidationItem[];
}

/** The ONE place an error body becomes an operator-facing sentence. */
export function errorMessage(
  body: ErrorBody | null,
  status: number,
  statusText: string,
): string {
  const detail = body?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        item && typeof item === "object" && typeof item.msg === "string"
          ? item.msg
          : JSON.stringify(item),
      )
      .filter(Boolean)
      .join("; ");
    if (messages) return messages;
  }
  return `${status} ${statusText}`;
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
    throw new ApiError(res.status, errorMessage(body, res.status, res.statusText));
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
// Node serving (single-slot swap control) — /api/nodes/{id}/serving/*, see
// app/routers/serving.py. The Spark* type names/shapes are unchanged (design
// §7): this is the same wire contract the removed /api/spark/* routes spoke,
// now addressed per node.
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

/** GET /api/nodes/{id}/serving/status — null when the node is not operable
 * (503, app/routers/serving.py:_client). Every other failure propagates: a
 * declared-operable node we cannot reach is an error to surface, not an
 * absence. */
export async function getNodeServingStatus(nodeId: string): Promise<SparkStatus | null> {
  try {
    return await request<SparkStatus>(
      `/api/nodes/${encodeURIComponent(nodeId)}/serving/status`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 503) return null;
    throw err;
  }
}

/** POST /api/nodes/{id}/serving/swap. 409s: busy guard (force helps),
 * mid-boot guard (force helps, button withheld — SparkSwap.tsx docstring),
 * litellm default-route guard (force-proof). */
export function postNodeSwap(nodeId: string, profile: string, force = false): Promise<unknown> {
  return request<unknown>(`/api/nodes/${encodeURIComponent(nodeId)}/serving/swap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, force }),
  });
}

/** POST /api/nodes/{id}/serving/reload — JSON body required (matches swap). */
export function postNodeReload(nodeId: string): Promise<unknown> {
  return request<unknown>(`/api/nodes/${encodeURIComponent(nodeId)}/serving/reload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
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
 * null when no such engine/node (request()'s `(await res.json()) as T` passes
 * a JSON null body through unchanged). */
export function getCatalog(node: string, engine: string): Promise<Catalog | null> {
  return request<Catalog | null>(
    `/api/settings/catalog/${encodeURIComponent(node)}/${encodeURIComponent(engine)}`
  );
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

/** POST /api/settings/preview — one wire helper for both directions of the
 * same endpoint (parse ships {argline}, render ships {args}) so the shared
 * ctx contract can't drift between them. */
function postPreview(
  body: { argline: string } | { args: Record<string, ArgValue> },
  ctx?: { node?: string; engine?: string; model?: string }
): Promise<SettingsPreviewResponse> {
  const payload: Record<string, unknown> = { ...body };
  if (ctx?.node) payload.node = ctx.node;
  if (ctx?.engine) payload.engine = ctx.engine;
  if (ctx?.model) payload.model = ctx.model;
  return request<SettingsPreviewResponse>("/api/settings/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** POST /api/settings/preview with {argline} body */
export function previewParse(
  argline: string,
  ctx?: { node?: string; engine?: string; model?: string }
): Promise<SettingsPreviewResponse> {
  return postPreview({ argline }, ctx);
}

/** POST /api/settings/preview with {args} body */
export function previewRender(
  args: Record<string, ArgValue>,
  ctx?: { node?: string; engine?: string; model?: string }
): Promise<SettingsPreviewResponse> {
  return postPreview({ args }, ctx);
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
  // encodeURIComponent, like every other path param in this file: unit ids
  // come from catalog scans of real FILENAMES, so a "#" or "?" in one
  // silently truncated the path [max-review c34].
  return request<StorageUnit>(`/api/storage/units/${encodeURIComponent(unitId)}`, {
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
  // `_public` spreads the stored NodeStore entry as-is, which always
  // carries `control` (app/node_store.py _CONTROLS, healed by
  // `_heal_control` if ever missing/invalid).
  control: "none" | "swap";
  // Optional, present ONLY on the local entry (app/node_store.py's
  // `_heal_engines` strips this key from every non-local row) — the local
  // node's declared local-engine list, verbatim off NodeStore (E1 Task 10).
  // `_public` (app/routers/nodes.py:88-89) spreads the stored dict as-is,
  // so this is the one GET this file exposes that carries `engines[]` —
  // `/api/state`'s `nodes` block (DeckNodeEntry, status.py's `_nodes_block`)
  // never does, which is why the Engines editor reads `listNodeRegistry()`
  // rather than the board's own node list.
  engines?: DeclaredEngine[];
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

/** GET /api/nodes (app/routers/nodes.py's `list_nodes`) — the registry's
 * OWN listing. Used by the Engines editor (E1 Task 12) specifically because
 * it is the one producer whose local entry carries `engines[]`; every other
 * screen reading node identity uses `/api/state`'s `nodes` block instead. */
export function listNodeRegistry(): Promise<{ nodes: NodeRegistryEntry[] }> {
  return request<{ nodes: NodeRegistryEntry[] }>("/api/nodes");
}

// ---------------------------------------------------------------------------
// Declared local engines (/api/nodes/local/engines, /api/engine-kinds — see
// app/routers/nodes.py's E1 Task 10 section, :200-339) — the local node's
// declared-engine CRUD, plus the kind catalog that drives the UI's kind
// picker and per-kind connection fields (spec §5: never a UI literal).
// ---------------------------------------------------------------------------

/** One connection field's requiredness for a given kind — GET
 * /api/engine-kinds's per-kind `connection` map
 * (app/routers/nodes.py:334-336, sourced from `KNOWN_KINDS`,
 * app/engine_kinds.py:90-94). */
export interface EngineConnectionFieldSchema {
  required: boolean;
}

export interface EngineKindDef {
  kind: string;
  connection: Record<string, EngineConnectionFieldSchema>;
  human_verbs: string[];
}

/** GET /api/engine-kinds's body (app/routers/nodes.py:324-339's
 * `list_engine_kinds`). */
export interface EngineKindsResponse {
  kinds: EngineKindDef[];
}

export interface EnginePolicyDefaults {
  priority: number;
  pinned: boolean;
  idle_ttl: number;
}

/** One declared local engine — the exact shape
 * `app.engine_kinds.validate_engines` accepts (app/engine_kinds.py:103-150:
 * resource, kind, connection, gpu_index, policy_defaults, no other field),
 * what POST/PUT `/api/nodes/local/engines*` both accept and echo back, and
 * what `NodeRegistryEntry.engines[]` carries verbatim off the local
 * NodeStore row. */
export interface DeclaredEngine {
  resource: string;
  kind: string;
  connection: Record<string, string>;
  gpu_index: number;
  policy_defaults: EnginePolicyDefaults;
}

/** GET /api/engine-kinds (app/routers/nodes.py:324-339) — the UI's kind
 * picker + per-kind connection-field source; never a UI literal (spec §5). */
export function getEngineKinds(): Promise<EngineKindsResponse> {
  return request<EngineKindsResponse>("/api/engine-kinds");
}

/** POST /api/nodes/local/engines (app/routers/nodes.py:221-241's
 * `add_engine`). 422 (via the redacting handler — `app.main`'s
 * RequestValidationError handler) for a shape defect, with
 * `app.engine_kinds.validate_engines`' own one-line message as `detail`;
 * 409 when the resource is already declared. */
export function addEngine(body: DeclaredEngine): Promise<DeclaredEngine> {
  return request<DeclaredEngine>("/api/nodes/local/engines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** PUT /api/nodes/local/engines/{resource} (app/routers/nodes.py:244-270's
 * `update_engine`) — full-entry replace. `resource` in `body` must equal
 * `resource` here or the route 422s the rename ("forget and re-add
 * instead") — callers never offer renaming, they keep the path resource and
 * the body resource in lockstep. */
export function updateEngine(resource: string, body: DeclaredEngine): Promise<DeclaredEngine> {
  return request<DeclaredEngine>(
    `/api/nodes/local/engines/${encodeURIComponent(resource)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/** DELETE /api/nodes/local/engines/{resource} (app/routers/nodes.py:273-321's
 * `forget_engine`) — FORGET, bookkeeping only: drops the declaration plus
 * the deck's own intent record and stored policy row. Never calls the
 * engine itself — a running process, if any, is left exactly as it is. */
export function forgetEngine(resource: string): Promise<{ status: string }> {
  return request<{ status: string }>(
    `/api/nodes/local/engines/${encodeURIComponent(resource)}`,
    { method: "DELETE" },
  );
}
