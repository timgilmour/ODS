/**
 * GPU-Monitor-format stats — the tone/formatting rules `GpuStatsBlock`
 * renders, kept here rather than inline so the thresholds have exactly one
 * definition and a full boundary test (KISS "pure functions" for
 * business/display logic).
 *
 * Every function takes `number | null` — `nodes.ts`'s `GpuStats` folds an
 * absent wire reading and an explicit null into the same "no reading" fact
 * (its own docstring), so a null here means the producer took no reading,
 * never a fabricated zero. Each formatter renders GPUCard.jsx's own
 * unavailable pattern for that case: "—".
 *
 * Thresholds are a straight port of core ODS's
 * dashboard/src/components/GPUCard.jsx — temp red >=85, amber >=70
 * (:31-38); util bar red >90, amber >70 (:4-13). Cited rather than
 * re-derived: these are somebody else's judgment calls about what counts as
 * "hot" or "busy", carried over verbatim so the deck's board agrees with the
 * dashboard's about the same GPU.
 */

/** Temperature tone — GPUCard.jsx:31-38's `tempColor` ladder, restated as a
 * tone name rather than a Tailwind class so `GpuStatsBlock` can map it onto
 * the deck's own `.ui-pill-*` tokens (deck.css) instead of importing a new
 * color. `"na"` is this module's own addition: GPUCard.jsx's zinc-600
 * "no reading" case, named rather than left to fall out of the ladder. */
export function tempTone(c: number | null): "good" | "warn" | "bad" | "na" {
  if (c === null) return "na";
  if (c >= 85) return "bad";
  if (c >= 70) return "warn";
  return "good";
}

/** Utilization bar fill class — GPUCard.jsx:4-13's `Bar` color ladder,
 * returned as the same `"meter-fill meter-<tone>"` pair `api.ts`'s
 * `meterFillClass` already uses for the VRAM meter, so `GpuStatsBlock`'s
 * util bar and `Meter`'s VRAM bar share one class vocabulary rather than
 * inventing a second. A null reading (no telemetry) renders the neutral
 * class — the bar's width is 0 in that case (`formatUtil` renders "—"
 * beside it), so the color is moot, but a defined return keeps every call
 * site free of a null branch. Deliberately `>` not `>=` at both thresholds,
 * matching GPUCard.jsx's own strict-greater-than ladder — 70 and 90 stay
 * the lower tone, 71 and 91 cross into the next one. */
export function utilFillClass(pct: number | null): string {
  if (pct !== null && pct > 90) return "meter-fill meter-red";
  if (pct !== null && pct > 70) return "meter-fill meter-amber";
  return "meter-fill meter-neutral";
}

/** "3%" | "—" — GPUCard.jsx's own util row format. Rounded: telemetry can
 * arrive fractional even though the reading is conceptually a whole
 * percent, and a bare decimal would read as false precision. */
export function formatUtil(pct: number | null): string {
  if (pct === null) return "—";
  return `${Math.round(pct)}%`;
}

/** "29°C" | "—" — GPUCard.jsx's own temp format, minus its inline "!" alert
 * marker (the tone-colored pill this renders into already carries that
 * signal, so a second glyph would say the same thing twice). */
export function formatTemp(c: number | null): string {
  if (c === null) return "—";
  return `${Math.round(c)}°C`;
}

/** "15.0W" | "—" — one decimal, always shown (never `Math.round`), because
 * a bare-watt reading loses exactly the precision that separates a truly
 * idle card from one at its floor. */
export function formatPower(w: number | null): string {
  if (w === null) return "—";
  return `${w.toFixed(1)}W`;
}
