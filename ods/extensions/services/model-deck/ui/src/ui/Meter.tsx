import { bytesToGB, meterFillClass } from "../api";

/** A capacity bar. `capacity: null` means no reading is available — rendered
 * hatched, because an empty track would claim zero bytes used.
 *
 * `tone` picks what the fill color means. "capacity" (default) runs the
 * value through `meterFillClass`'s amber/red-at-95% thresholds — right for
 * "how full is this resource", wrong for "how far along is this job".
 * Design decision 5: a healthy job at 82% must not render amber ("this
 * wants a decision") or a nearly-done one at 96% render red ("this
 * failed") — jobs are neutral/in-progress until their own terminal state
 * says otherwise, which the state chip beside the meter already carries.
 * "neutral" pins the fill to `meter-neutral` regardless of pct, for any
 * meter (job progress today, others later) whose number is a fraction
 * completed rather than a fraction of capacity consumed. */
export default function Meter({
  capacity,
  watermarkPct,
  watermarkTitle,
  tone = "capacity",
}: {
  capacity: { used: number; total: number } | null;
  watermarkPct?: number;
  /** Tooltip for the watermark tick — carried over from the old
   * LocationColumn tick's `title="watermark N GB"`, lost in the Meter
   * migration until restored here. */
  watermarkTitle?: string;
  tone?: "capacity" | "neutral";
}) {
  if (capacity === null) {
    return (
      <div className="ui-meter">
        <div className="ui-meter-track ui-meter-unknown" />
        <span className="ui-meter-label">—</span>
      </div>
    );
  }

  const pct = capacity.total > 0 ? (capacity.used / capacity.total) * 100 : 0;
  const fillClass = tone === "neutral" ? "meter-fill meter-neutral" : meterFillClass(pct);

  return (
    <div className="ui-meter">
      <div className="ui-meter-track">
        <div
          className={`ui-meter-fill ${fillClass}`}
          // Clamp stays unconditional: backend bytes_done can transiently
          // exceed bytes_total mid-copy, and an unclamped width would blow
          // out of the track regardless of which tone is driving the color.
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
        {watermarkPct != null && (
          <div
            className="ui-meter-tick"
            style={{ left: `${Math.min(watermarkPct, 100)}%` }}
            title={watermarkTitle}
          />
        )}
      </div>
      <span className="ui-meter-label">
        {bytesToGB(capacity.used)} / {bytesToGB(capacity.total)} GB
      </span>
    </div>
  );
}
