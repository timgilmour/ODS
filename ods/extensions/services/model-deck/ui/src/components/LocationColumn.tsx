import { useState, type DragEvent } from "react";
import {
  bytesToGB,
  meterFillClass,
  putUnitPinned,
  truncateMiddle,
  type StorageLocation,
  type StorageUnit,
} from "../api";
import { labels } from "../model/messages";

interface LocationColumnProps {
  location: StorageLocation;
  /** Units already filtered to this location by the parent (StorageView
   * owns the full catalog; a column only renders its own slice). */
  units: StorageUnit[];
  /** dest is the preselected destination — a column-drop names itself
   * (location.name); the per-unit "Move…" button passes null and lets the
   * modal own eligibility. */
  onRequestMove: (unitId: string, dest: string | null) => void;
  onChanged: () => void;
}

/** One storage location: header (name/role/store_type/engine), a capacity
 * meter (reusing GpuColumn's meter-track/meterFillClass, plus a watermark
 * tick when set), and its resident unit cards. The whole column is also an
 * HTML5 drop target (ModelLibrary/SetBuilder idiom) for units dragged in
 * from another column. */
export default function LocationColumn({
  location,
  units,
  onRequestMove,
  onChanged,
}: LocationColumnProps) {
  const [pinBusy, setPinBusy] = useState<string | null>(null);
  const [pinError, setPinError] = useState<{ unitId: string; message: string } | null>(null);

  const usedBytes =
    location.total_bytes != null && location.free_bytes != null
      ? location.total_bytes - location.free_bytes
      : null;
  const pct =
    location.total_bytes && location.total_bytes > 0 && usedBytes != null
      ? (usedBytes / location.total_bytes) * 100
      : 0;
  const watermarkPct =
    location.watermark_gb != null && location.total_bytes
      ? Math.min(((location.watermark_gb * 1e9) / location.total_bytes) * 100, 100)
      : null;

  async function handlePinToggle(unit: StorageUnit) {
    setPinBusy(unit.id);
    try {
      await putUnitPinned(unit.id, !unit.pinned);
      setPinError(null);
    } catch (err) {
      setPinError({ unitId: unit.id, message: err instanceof Error ? err.message : String(err) });
    } finally {
      setPinBusy(null);
      onChanged();
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const unitId = e.dataTransfer.getData("text/plain");
    // Ignore malformed drops, drops back onto the unit's current column (a
    // no-op move), and drops onto an unavailable (mount-missing) column —
    // mirrors SetBuilder's silent-ignore-on-bad-drop idiom; a doomed move
    // is never submitted, same rule the modal applies when listing eligible
    // destinations for the per-unit "Move…" path.
    if (!unitId || !location.available || units.some((u) => u.id === unitId)) return;
    onRequestMove(unitId, location.name);
  }

  return (
    <div
      className={`location-column${!location.available ? " location-unavailable" : ""}`}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <div className="location-column-head">
        <span className="location-name" title={location.name}>
          {truncateMiddle(location.name, 22)}
        </span>
        <span className={`chip ${location.role === "hot" ? "chip-running" : "chip-parked"}`}>
          {location.role}
        </span>
      </div>
      <div className="location-meta">
        {location.store_type} · {location.engine}
        {location.readonly && " · read-only"}
      </div>

      {!location.available && (
        <div className="banner-error location-unavailable-banner">
          <span>mount missing — files retained</span>
        </div>
      )}

      <div className="location-meter">
        <div className="meter-track">
          <div className={meterFillClass(pct)} style={{ width: `${Math.min(pct, 100)}%` }} />
          {watermarkPct != null && (
            <div
              className="meter-watermark"
              style={{ left: `${watermarkPct}%` }}
              title={`watermark ${location.watermark_gb} GB`}
            />
          )}
        </div>
        <div className="meter-label">
          {bytesToGB(usedBytes)} / {bytesToGB(location.total_bytes)} GB ({pct.toFixed(0)}%)
        </div>
      </div>

      <ul className="unit-list">
        {units.length === 0 && <li className="unit-empty">no units</li>}
        {units.map((unit) => {
          const moving = unit.state === "moving";
          return (
            <li
              key={unit.id}
              className={`unit-card${moving ? " unit-moving" : ""}`}
              draggable={!moving}
              onDragStart={(e) => {
                e.dataTransfer.setData("text/plain", unit.id);
                e.dataTransfer.effectAllowed = "move";
              }}
            >
              <div className="unit-card-row">
                <span className="unit-name" title={unit.name}>
                  {truncateMiddle(unit.name)}
                </span>
                {moving && <span className="chip chip-busy">moving</span>}
              </div>

              <div className="unit-card-row">
                <span className="unit-size">{bytesToGB(unit.size)} GB</span>
                <button
                  className={`unit-pin-btn${unit.pinned ? " unit-pinned" : ""}`}
                  aria-pressed={unit.pinned}
                  title={unit.pinned ? "pinned — click to unpin" : "click to pin (exempt from tiering)"}
                  onClick={() => handlePinToggle(unit)}
                  disabled={moving || pinBusy === unit.id}
                >
                  📌
                </button>
              </div>

              {pinError && pinError.unitId === unit.id && (
                <div className="banner-error">
                  <span>{pinError.message}</span>
                  <button
                    onClick={() => setPinError(null)}
                    aria-label="dismiss error"
                  >
                    ×
                  </button>
                </div>
              )}

              {/* Drag is never the only path to a move — a per-card button
                  covers keyboard/touch users. Opens with no destination
                  preselected; the modal owns eligibility and lets the
                  operator pick. */}
              <button
                type="button"
                disabled={moving}
                onClick={() => onRequestMove(unit.id, null)}
              >
                {labels.moveTo}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
