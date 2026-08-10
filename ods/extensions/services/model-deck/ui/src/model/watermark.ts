/**
 * Watermark input parsing — refuse, never coerce.
 *
 * Lives in model/ rather than inline in PolicyModal for the reason this
 * codebase has now learned four times: logic inline in a component is logic
 * no test can reach (there is no component harness here — vitest runs in a
 * "node" environment with no testing-library).
 *
 * The rule it enforces: a watermark that does not parse must BLOCK the save.
 * The previous inline version returned `null` for anything unparseable, and
 * `watermark_gb: null` is a legal, meaningful value the backend reads as "no
 * watermark on this drive" — so a typo like "50 GB" silently DISABLED
 * auto-archiving for a hot location, reporting success [max-review #15].
 * Empty and unparseable are different answers and must not collapse.
 */

/** The one value that means "explicitly no watermark" — a deliberate,
 * legal setting, distinct from an input we could not understand. */
export const NO_WATERMARK = null;

export type WatermarkParse = number | null | "invalid";

/**
 * `""`/whitespace -> `null` (explicitly no watermark).
 * A non-negative finite number -> that number.
 * Anything else -> `"invalid"`, which callers must REFUSE rather than
 * translate into `null`.
 *
 * Negative is invalid rather than clamped: a negative GB threshold is not a
 * threshold the operator could have meant, and clamping would silently write
 * a number they never typed.
 */
export function parseWatermark(raw: string | undefined | null): WatermarkParse {
  const trimmed = raw?.trim();
  if (!trimmed) return NO_WATERMARK;
  // Number("") is 0 and Number("  ") is 0, both already handled above.
  // Number("50 GB") is NaN — the case that motivated all of this.
  const n = Number(trimmed);
  return Number.isFinite(n) && n >= 0 ? n : "invalid";
}
