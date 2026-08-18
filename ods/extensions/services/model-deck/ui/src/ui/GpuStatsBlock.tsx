import { formatPower, formatTemp, formatUtil, tempTone, utilFillClass } from "../model/gpuStats";
import { labels } from "../model/messages";
import type { GpuStats } from "../model/nodes";

/** `tempTone`'s four-way answer, mapped onto the deck's existing pill tokens
 * (deck.css's `.ui-pill-*`) — no new colors, per the styling rule. `"na"`
 * (no reading) takes the "off" token, the same tone an unloaded resource's
 * own pill already carries for "nothing to report". */
const TEMP_PILL_CLASS: Record<ReturnType<typeof tempTone>, string> = {
  good: "ui-pill-good",
  warn: "ui-pill-warn",
  bad: "ui-pill-bad",
  na: "ui-pill-off",
};

/** GPU-Monitor-format stats (design §D) — utilization bar, temperature,
 * power. Render-only: every tone/format decision lives in `model/
 * gpuStats.ts` (thresholds ported from core ODS's own GPUCard.jsx, cited
 * there). ResourcePanel renders this directly above the card's ONE VRAM bar
 * (`<Meter capacity={resource.capacity} />`) — this block never touches
 * capacity, and never uses `Meter`, because `Meter`'s label speaks GB while
 * this row speaks percent.
 *
 * The hardware name (`stats.name`) is NOT rendered here — ResourcePanel
 * puts it beside the panel title (a muted `.gpu-name` span next to
 * `resource.label`), the one place a GPU's identity belongs, so it is not
 * repeated a second time in this block's footer. */
export default function GpuStatsBlock({ stats }: { stats: GpuStats }) {
  const { utilizationPercent, temperatureC, powerW } = stats;
  return (
    <div className="gpu-stats">
      <div className="gpu-stats-row">
        <span className="gpu-stats-label">{labels.gpuUtil}</span>
        <div className="ui-meter-track">
          <div
            className={`ui-meter-fill ${utilFillClass(utilizationPercent)}`}
            style={{ width: `${Math.min(utilizationPercent ?? 0, 100)}%` }}
          />
        </div>
        <span className="gpu-stats-value">{formatUtil(utilizationPercent)}</span>
      </div>
      <div className="gpu-stats-footer">
        <span
          className={`ui-pill ${TEMP_PILL_CLASS[tempTone(temperatureC)]}`}
          title={labels.gpuTemp}
        >
          {formatTemp(temperatureC)}
        </span>
        <span className="gpu-stats-power" title={labels.gpuPower}>
          {formatPower(powerW)}
        </span>
      </div>
    </div>
  );
}
