import { useState, type DragEvent } from "react";
import {
  bytesToGB,
  putUnitPinned,
  truncateMiddle,
  type StorageLocation,
  type StorageUnit,
} from "../api";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Meter from "../ui/Meter";
import Panel from "../ui/Panel";

interface LocationCardProps {
  location: StorageLocation;
  /** Units already filtered to this location by the parent (StorageView
   * owns the full catalog; a card only renders its own slice). */
  units: StorageUnit[];
  /** dest is the preselected destination — a card-drop names itself
   * (location.name); the per-unit "Move…" button passes null and lets the
   * modal own eligibility. */
  onRequestMove: (unitId: string, dest: string | null) => void;
  onChanged: () => void;
}

/** One storage location: a Panel titled with its name and a role pill, a
 * capacity meter (with a watermark tick when set), and its resident unit
 * rows. The whole card is also an HTML5 drop target (ModelLibrary/SetBuilder
 * idiom) for units dragged in from another card. */
export default function LocationCard({
  location,
  units,
  onRequestMove,
  onChanged,
}: LocationCardProps) {
  const [pinBusy, setPinBusy] = useState<string | null>(null);
  const [pinError, setPinError] = useState<{ unitId: string; message: string } | null>(null);

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
    // Ignore malformed drops, drops back onto the unit's current card (a
    // no-op move), and drops onto an unavailable (mount-missing) card —
    // mirrors SetBuilder's silent-ignore-on-bad-drop idiom; a doomed move
    // is never submitted, same rule the modal applies when listing eligible
    // destinations for the per-unit "Move…" path.
    if (!unitId || !location.available || units.some((u) => u.id === unitId)) return;
    onRequestMove(unitId, location.name);
  }

  return (
    <Panel
      className={`location-card${!location.available ? " location-unavailable" : ""}`}
      title={
        <span className="location-name" title={location.name}>
          {truncateMiddle(location.name, 22)}
        </span>
      }
      actions={
        // The hot-role pill uses ui-pill-warn (amber) deliberately: the R1
        // storage render shows the hot chip amber, and role labeling is a
        // fact, not a notice — decision 5 governs notices; the render
        // governs the chip. Cold gets the neutral off pill.
        <span className={`ui-pill ${location.role === "hot" ? "ui-pill-warn" : "ui-pill-off"}`}>
          {location.role}
        </span>
      }
    >
      <div
        className="location-card-body"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="location-meta">
          {location.store_type} · {location.engine}
          {location.readonly && ` · ${labels.readonly}`}
        </div>

        {!location.available && <Banner message={messages.mountMissing()} />}

        <Meter
          capacity={
            location.total_bytes != null && location.free_bytes != null
              ? {
                  used: location.total_bytes - location.free_bytes,
                  total: location.total_bytes,
                }
              : null
          }
          watermarkPct={watermarkPct ?? undefined}
        />

        <ul className="unit-list">
          {units.length === 0 && <li className="unit-empty">{labels.noUnits}</li>}
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
                  {moving && <span className="ui-pill ui-pill-busy">{unit.state}</span>}
                </div>

                <div className="unit-card-row">
                  <span className="unit-size">{bytesToGB(unit.size)} GB</span>
                  <button
                    type="button"
                    className={`unit-pin-btn${unit.pinned ? " unit-pinned" : ""}`}
                    aria-pressed={unit.pinned}
                    title={unit.pinned ? labels.pinTitlePinned : labels.pinTitleUnpinned}
                    onClick={() => handlePinToggle(unit)}
                    disabled={moving || pinBusy === unit.id}
                  >
                    📌
                  </button>
                </div>

                {pinError && pinError.unitId === unit.id && (
                  <Banner
                    message={messages.guardRefused(pinError.message)}
                    onDismiss={() => setPinError(null)}
                  />
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
    </Panel>
  );
}
