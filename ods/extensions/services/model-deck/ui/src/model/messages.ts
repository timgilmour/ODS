/**
 * Every operator-visible string in the deck, in one place.
 *
 * Two reasons this exists rather than literals in components. First, tone is
 * attached to the message, so one condition cannot be red in one screen and
 * amber in another — the exact inconsistency the R1 drift card showed.
 * Second, backend detail strings ("busy — 2 requests in flight") arrive
 * lowercase and mid-sentence; Banner capitalizes at render, so nothing here
 * mutates a payload to make it presentable.
 *
 * Pure: no React, no formatting of live clocks. Callers pass in already
 * humanized ages.
 */

export type Tone = "neutral" | "warning" | "danger";

export interface Message {
  tone: Tone;
  title: string;
  body?: string;
  action?: { label: string };
}

export const messages = {
  nodeUnreachable: (label: string, age: string | null): Message => ({
    tone: "danger",
    title: "Node is unreachable",
    body: age
      ? `${label} last answered ${age} ago. What follows is the last thing it told us.`
      : `${label} is not answering. What follows is the last thing it told us.`,
    action: { label: "Retry" },
  }),

  // The node answered; what it should be running is not running. Distinct
  // from nodeUnreachable (we cannot see the box at all) and from warming (a
  // boot is legitimately in flight). `detail` is the backend's own sentence
  // — a swap helper error, or lifecycle's reason — because without it a
  // failed asynchronous swap renders as a red pill and nothing else.
  nodeDown: (label: string, detail: string | null): Message => ({
    tone: "danger",
    title: "Node is down",
    body: detail
      ? `${label} answered, but nothing is serving — ${detail}`
      : `${label} answered, but nothing is serving.`,
    action: { label: "Retry" },
  }),

  warmingFirstBoot: (): Message => ({
    tone: "neutral",
    title: "first boot can autotune for about 15 minutes — this is normal",
  }),

  // Distinct from nodeUnreachable, which reports what the BACKEND says about
  // a node. This one says THIS PAGE could not reach the deck's own endpoint
  // for that node — so everything below it may be minutes old even while the
  // backend's own view still calls the node reachable.
  nodeFetchFailed: (label: string, detail: string): Message => ({
    tone: "danger",
    title: `Cannot reach ${label} from this page`,
    body: `${detail} — what follows may be out of date.`,
  }),

  guardRefused: (detail: string): Message => ({
    tone: "danger",
    title: "Refused",
    body: detail,
  }),

  // Warning, not danger: this is asking the operator for a decision, which
  // is what warning means here. Nothing has failed or been refused — the
  // red-outlined button above it is what carries the danger.
  forceConfirm: (): Message => ({
    tone: "warning",
    title: "Click again to confirm",
  }),

  modelIsCold: (sizeGb: string): Message => ({
    tone: "warning",
    title: "Model is cold",
    body: `pull ${sizeGb} GB to hot storage, then load?`,
    action: { label: "Pull + load" },
  }),

  pullingFromCold: (): Message => ({
    tone: "neutral",
    title: "Pulling from cold storage",
    body: "the model will load once it lands on hot storage.",
  }),

  stateRefreshFailed: (detail: string): Message => ({
    tone: "danger",
    title: "State refresh failed",
    body: detail,
  }),

  eventsFetchFailed: (detail: string): Message => ({
    tone: "danger",
    title: "Could not load events",
    body: detail,
  }),

  noEvents: (): Message => ({ tone: "neutral", title: "no events yet" }),

  lastKnown: (): Message => ({ tone: "neutral", title: "last known" }),

  // The state pill already reads "last known"; this answers the next
  // question, which is *how* stale — a different fact, not a redundant one.
  lastSeen: (age: string): Message => ({
    tone: "neutral",
    title: `last seen ${age} ago`,
  }),
};

/** Short labels — control text, badges, captions and the `title` tooltips
 * that explain them. Not notices, hence not `Message`s (there is no tone to
 * carry), but every one of them is operator-visible text and an ARIA name or
 * a tooltip is read out loud, so they are centralized here for the same
 * reason the notices are. Parameterized entries are plain pure functions;
 * they format a fact, they do not decide anything. */
export const labels = {
  dismiss: "Dismiss",
  close: "Close",
  cancel: "Cancel",
  confirm: "Confirm",
  applying: "Applying…",
  forceApply: "Force apply",
  forceApplyTitle:
    "override the hipfire conversation-guard — the live/recent conversation will lose its cache and its next turn will re-read the whole history",
  loadingPreview: "Loading preview…",
  forcePark: "Force park",
  forceSwap: "Force swap",
  filterEvents: "Filter events…",

  /** Caption for a resource with nothing on it. Deliberately says nothing
   * about WHAT the resource is — the panel above it is already titled
   * "GPU 0" or "Serving slot", and the previous copy ("Serving slot", taken
   * from a Message meant for the spark) rendered as "GPU 0 / Serving slot"
   * on the most common empty state the board has. A caption is a label, not
   * a notice: it carries no tone, so it is not a Message. */
  nothingPlaced: "nothing placed",

  // Top-level chrome and navigation. `events` was the only tab that ever
  // came from here, which is precisely how a half-applied rule dies: the
  // next author copies whichever neighbour they happened to read first.
  appTitle: "Model Deck",
  appSubtitle: "GPU/VRAM control for lemonade, ComfyUI, hipfire",
  deck: "On Deck",
  setBuilder: "Set Builder",
  storage: "Storage",
  events: "Events",
  policy: "Policy",
  loading: "loading…",

  // Placement controls.
  modelToLoad: "model to load",
  selectModel: "select a model…",
  noModels: "no models found",
  coldGroup: "❄ cold",
  coldOption: (name: string, sizeGb: string) => `❄ ${name} (${sizeGb} GB)`,
  load: "Load",
  unload: "Unload",
  free: "Free",
  comfyuiBlockedTitle: "ComfyUI is busy or has a non-empty queue",
  park: "Park",
  resume: "Resume",

  // Spark's profile picker.
  swapTo: "swap to…",
  swap: "Swap",
  /** A profile whose engine is not the node's usual one says so in the
   * picker, the same fact the chip's engine badge carries. */
  swapOption: (profile: string, engine: string | null) =>
    engine ? `${profile} (${engine})` : profile,

  // Placement facts (see Placement's "operational facts" block in nodes.ts).
  pinned: "📌",
  pinnedTitle: "pinned — exempt from idle eviction",
  priority: (n: number) => `P${n}`,
  priorityTitle: "eviction priority",
  inUse: "in use",
  inUseTitle:
    "a conversation turn is being served right now — park/apply will refuse without force",
  queue: (n: number | null) => `queue ${n ?? "—"}`,
  queueTitle: "jobs waiting on this engine",
  idle: (seconds: number) => `idle ${Math.round(seconds)} s`,
  idleTitle: "time since last activity — counts towards the idle-TTL eviction",

  // Set-apply step vocabulary — one label per "step" kind app/sets.py's
  // plan_apply() emits: unload_lemonade (sets.py:237), free_comfyui (:244),
  // warn (:246,:265,:296), park_hipfire (:258), activate (:263),
  // resume_hipfire (:288), load_lemonade (:298), policy_patch (:302).
  stepUnload: "Unload",
  stepLoad: "Load",
  stepFreeComfyui: "Free ComfyUI VRAM",
  stepParkHipfire: "Park hipfire",
  stepResumeHipfire: "Resume hipfire",
  stepActivate: "Activate catalog model",
  stepPolicyPatch: "Apply policy overrides",
  stepWarn: "Warning",
  estimate: (s: number) => `about ${s}s`,
  noChanges: "no changes — already matches this set",
  stepsCompleted: (n: number) => `${n} step(s) completed`,
};

/** "26h", "4m", "3d" — a compact age for a timestamp, or null when there is
 * no timestamp to age. NodeCard computes this once per node and threads it
 * into every place that fact is shown: the header's own age span, the
 * `nodeUnreachable` banner body, and `staleAge` on ResourcePanel (which
 * feeds the per-chip `lastSeen` caption) — one computed age, several
 * renderings, rather than each caller re-deriving it.
 *
 * `now` is injected so this stays pure and testable; production callers omit
 * it. Ages clamp at zero: a node whose clock runs ahead of ours must not be
 * reported as last seen in the future.
 *
 * Stays in hours through the second day (< 48h) before switching to days —
 * a 24h cutoff would print a 26-hour-old reading as "1d", which throws away
 * exactly the precision an operator needs right after a node drops. */
export function humanizeAge(iso: string | null, now: number = Date.now()): string | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;

  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 2 * 86_400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86_400)}d`;
}
