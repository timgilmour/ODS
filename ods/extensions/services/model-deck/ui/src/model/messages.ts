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
};

/** Short imperative labels for controls. Not notices, hence not `Message`s —
 * but still operator-visible text, so still centralized here. */
export const labels = {
  dismiss: "Dismiss",
};
