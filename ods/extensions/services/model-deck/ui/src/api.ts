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
  state: "loaded" | "unloaded" | "unknown";
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

export interface StateResponse {
  world: World;
  policy: PolicyMap;
  models: ModelFile[];
}

export interface EventEntry {
  ts: string;
  kind: string;
  detail: Record<string, unknown>;
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

/** POST /api/spark/swap. Throws ApiError(409) for both guard refusals
 * (busy serving — force-retryable) and an already-running swap (not
 * force-retryable; the message says which). */
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
