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

  noEvents: (): Message => ({ tone: "neutral", title: "no events yet" }),

  emptySlot: (): Message => ({ tone: "neutral", title: "Serving slot" }),

  lastKnown: (): Message => ({ tone: "neutral", title: "last known" }),

  // The state pill already reads "last known"; this answers the next
  // question, which is *how* stale — a different fact, not a redundant one.
  lastSeen: (age: string): Message => ({
    tone: "neutral",
    title: `last seen ${age} ago`,
  }),
};

/** Short imperative labels for controls. Not notices, hence not `Message`s —
 * but still operator-visible text, so still centralized here. */
export const labels = {
  dismiss: "Dismiss",
  forcePark: "Force park",
  forceSwap: "Force swap",
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
