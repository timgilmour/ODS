/**
 * Event severity classification for the Events tab.
 *
 * Backend event kinds (see app/arbiter.py, app/sets.py, app/storage.py,
 * app/mover.py, app/routers/*.py's log_event/_log call sites) are not a
 * closed enumeration — new kinds get added as new call sites get added.
 * The first cut of this feature hardcoded a kind->CSS-class map lifted from
 * the design mockup ("refused", "error", "reconciled", "load", "unload",
 * "pull") — illustrative labels that don't appear anywhere in the real
 * backend, so every real event silently fell through to the neutral
 * default and the colouring never fired against production data.
 *
 * This classifies by the suffix/naming conventions the real vocabulary
 * already follows instead, which degrades gracefully: a kind nobody
 * enumerated here still lands in the right bucket as long as it follows
 * the convention, and an unrecognized one is neutral rather than invisible.
 *
 * The vocabulary itself mixes `-` and `_` as separators (e.g. "load-failed"
 * vs. "move_failed" vs. "storage_shortfall") depending on which module
 * logged it — normalizing to one separator before matching means either
 * convention classifies the same way, rather than requiring every call
 * site to agree on a style this file doesn't own.
 */

export type Severity = "failure" | "success" | "attention" | "neutral";

const FAILURE_SUFFIXES = ["-failed", "-error", "-vetoed", "-unreachable", "-quarantined"];
const SUCCESS_SUFFIXES = ["-done", "-end", "-restore"];
const ATTENTION_SUFFIXES = ["-warn"];

// Kinds that carry a severity but don't follow the suffix convention.
const SUCCESS_EXACT = new Set(["reconciled"]);
const ATTENTION_EXACT = new Set([
  "storage-shortfall", "host-agent-busy", "free-raced", "origin-moved",
  // The operator asked for a load and it did NOT happen: an action of
  // theirs overtook the pull-through mid-copy (app/routers/control.py).
  // Neutral would read as "nothing to see here".
  "pull-through-superseded",
  // A PUT named a tenant outside DEFAULT_POLICIES. Deliberately ACCEPTED
  // (defaults are seed data, not an allowlist), so the event is the only
  // feedback that a typo'd tenant name is policying nothing —
  // app/routers/policy.py's put_policy, NOT the store's boundary gate.
  "policy-unknown-tenant",
]);

function normalize(kind: string): string {
  return kind.replace(/_/g, "-");
}

/** Checked in this order — failed-restore ("lifecycle-restore-failed") must
 * classify as a failure, not a success, even though it contains "restore";
 * ending in "-failed" is checked first so it wins.
 *
 * `detail` is optional: pass it wherever it is available, because an
 * explicit outcome outranks the naming convention entirely (see below). */
export function eventSeverity(kind: string, detail?: unknown): Severity {
  // An explicit failure outcome beats the suffix rule. The backend logs BOTH
  // terminal results of a set apply under the ONE kind "apply-end"
  // (app/sets.py's run_apply logs {"outcome": "failed"} on a failed step and
  // {"outcome": "ok"} at the end), so the kind alone cannot classify it —
  // and "-end" reads as success, which rendered a FAILED apply green
  // [max-review #14]. Only "failed" is honoured: a detail must be able to
  // escalate a mis-suffixed kind, never to launder an unrelated one into
  // looking fine.
  if (detail !== null && typeof detail === "object") {
    if ((detail as Record<string, unknown>).outcome === "failed") return "failure";
  }

  const normalized = normalize(kind);

  if (FAILURE_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) return "failure";
  if (SUCCESS_EXACT.has(normalized) || SUCCESS_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) {
    return "success";
  }
  if (ATTENTION_EXACT.has(normalized) || ATTENTION_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) {
    return "attention";
  }
  return "neutral";
}
