import { bytesToGB, meterFillClass } from "../api";

/** A capacity bar. `capacity: null` means no reading is available — rendered
 * hatched, because an empty track would claim zero bytes used. */
export default function Meter({
  capacity,
  watermarkPct,
}: {
  capacity: { used: number; total: number } | null;
  watermarkPct?: number;
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

  return (
    <div className="ui-meter">
      <div className="ui-meter-track">
        <div
          className={`ui-meter-fill ${meterFillClass(pct)}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
        {watermarkPct != null && (
          <div className="ui-meter-tick" style={{ left: `${Math.min(watermarkPct, 100)}%` }} />
        )}
      </div>
      <span className="ui-meter-label">
        {bytesToGB(capacity.used)} / {bytesToGB(capacity.total)} GB
      </span>
    </div>
  );
}
