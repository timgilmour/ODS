/**
 * Reserve-GB input parsing — commit only what parses.
 *
 * Extracted for the same reason parseWatermark was, in the same commit's
 * sibling fix and one recurrence later: logic inline in a component is logic
 * no test can reach (vitest runs `environment: "node"` here — there is no
 * component harness, so a value computed inside SetBuilder is unreachable).
 *
 * The rule: an in-progress edit updates what the operator SEES without
 * rewriting the committed draft. The old inline version did
 * `Math.max(1, Math.round(value || 0))` on every keystroke, and `Number("")`
 * is 0 — so clearing the field snapped it to 1 under the cursor
 * [max-review c35].
 */

/** Whole GB, at least 1 — fractional or zero input causes an avoidable
 * server 422, so those are treated as "not a value yet" rather than
 * silently rounded into something the operator did not type. */
export function parseReserveGb(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;          // mid-edit, not a value
  const n = Number(trimmed);
  if (!Number.isInteger(n) || n < 1) return null;
  return n;
}
