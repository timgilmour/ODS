/**
 * Model Deck API client — typed wrappers over the FastAPI backend mounted
 * under /api (see app/routers/*.py). Every request re-reads the admin
 * token from localStorage at call time (rather than caching it), so a
 * token set/cleared via AdminGate takes effect on the very next call with
 * no extra plumbing.
 */

const TOKEN_KEY = "deck-token";

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

function authHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { "X-Deck-Token": token } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers ?? {}) },
  });

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

export function applySet(slug: string): Promise<ApplyReport> {
  return request<ApplyReport>(`/api/sets/${encodeURIComponent(slug)}/apply`, {
    method: "POST",
  });
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

export { TOKEN_KEY };
