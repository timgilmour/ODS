import { useEffect, useState } from "react";
import {
  bytesToGB,
  cancelStorageJob,
  postStorageMove,
  truncateMiddle,
  type StorageJob,
  type StorageLocation,
  type StorageUnit,
} from "../api";
import { moveRail } from "../model/moveRail";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Meter from "../ui/Meter";
import Modal from "../ui/Modal";
import StepRail from "../ui/StepRail";

type Phase = "confirm" | "progress" | "done" | "cancelled" | "error";

const NON_TERMINAL: ReadonlySet<StorageJob["state"]> = new Set(["queued", "copying", "verifying"]);

interface MoveModalProps {
  unit: StorageUnit;
  /** Every registered location; the modal derives eligible destinations
   * itself: not the unit's own location, and available only — a move onto a
   * missing mount is guaranteed to fail, so it is never offered (same rule
   * LocationCard applied when it owned the select). */
  locations: StorageLocation[];
  /** Preselected destination (a drag onto a specific card) or null (the
   * per-unit Move… path — the operator picks in the modal). */
  initialDest: string | null;
  /** storageState.jobs, read live via the PARENT's own 3s poll — this modal
   * never polls on its own; it just re-derives its job from whatever the
   * parent last fetched. */
  jobs: StorageJob[];
  onModalOpenChange: (open: boolean) => void;
  onClose: () => void;
  onChanged: () => void;
}

/** ApplyModal's phase idiom (confirm -> progress -> done/cancelled/error),
 * adapted for a single unit move. Unlike ApplyModal (whose parent owns
 * onModalOpenChange around mount/unmount), THIS modal drives it itself off
 * `phase` — polling must keep running once the move is submitted so the
 * progress bar can advance, and only the confirm step needs the pause. */
export default function MoveModal({
  unit,
  locations,
  initialDest,
  jobs,
  onModalOpenChange,
  onClose,
  onChanged,
}: MoveModalProps) {
  const eligible = locations.filter((l) => l.name !== unit.location && l.available);
  const [dest, setDest] = useState<string | null>(initialDest);
  const [phase, setPhase] = useState<Phase>("confirm");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Cleanup is load-bearing, not cosmetic: StorageView's onClose unmounts
  // this modal directly (no intermediate "closing" phase), most commonly
  // from the "confirm" phase via the Cancel button. Without restoring
  // false on unmount, that path would leave the parent's modalOpen stuck
  // true forever, freezing the 3s poll for every tab. A phase change still
  // re-runs this (cleanup-then-effect), so confirm -> progress correctly
  // resumes polling before any unmount could occur.
  useEffect(() => {
    onModalOpenChange(phase === "confirm");
    return () => onModalOpenChange(false);
  }, [phase, onModalOpenChange]);

  const job = jobId ? (jobs.find((j) => j.id === jobId) ?? null) : null;

  // Follow the job to its terminal state as the parent's poll refreshes
  // `jobs` — this is the only place phase advances past "progress".
  useEffect(() => {
    if (!job) return;
    if (job.state === "done") setPhase("done");
    else if (job.state === "cancelled") setPhase("cancelled");
    else if (job.state === "failed") setPhase("error");
  }, [job]);

  async function handleConfirm() {
    if (!dest) return;
    setBusy(true);
    setError(null);
    try {
      const { job: created } = await postStorageMove(unit.id, dest);
      setJobId(created.id);
      setPhase("progress");
    } catch (err) {
      // ApiError's .message IS the backend's `detail` string (see
      // api.ts request()) — a 409 (e.g. a conflicting move already
      // in flight for this unit) surfaces here with no special-casing.
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  async function handleCancel() {
    if (!jobId) return;
    setBusy(true);
    try {
      await cancelStorageJob(jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  function handleClose() {
    onChanged();
    onClose();
  }

  const rail = job ? moveRail(job.state) : null;

  return (
    <Modal
      title={labels.moveModel}
      subtitle={`${truncateMiddle(unit.name)} · ${bytesToGB(unit.size)} GB`}
      onClose={phase === "confirm" ? onClose : undefined}
      footer={
        <>
          {phase === "confirm" && (
            <>
              <button type="button" onClick={onClose} disabled={busy}>
                {labels.cancel}
              </button>
              <button
                type="button"
                className="primary"
                onClick={handleConfirm}
                disabled={busy || !dest}
              >
                {busy ? labels.starting : labels.startMove}
              </button>
            </>
          )}
          {phase === "progress" && (
            <button
              type="button"
              onClick={handleCancel}
              disabled={busy || !job || !NON_TERMINAL.has(job.state)}
            >
              {labels.cancelMove}
            </button>
          )}
          {(phase === "done" || phase === "cancelled" || phase === "error") && (
            <button type="button" className="primary" onClick={handleClose}>
              {labels.close}
            </button>
          )}
        </>
      }
    >
      {/* source → destination, the R1 render's route line */}
      <div className="move-route">
        <span className="move-route-loc">{unit.location}</span>
        <span className="move-route-arrow" aria-hidden="true">→</span>
        {phase === "confirm" ? (
          eligible.length > 0 && (
            <select
              aria-label={labels.destination}
              value={dest ?? ""}
              onChange={(e) => setDest(e.target.value || null)}
            >
              <option value="">{labels.moveTo}</option>
              {eligible.map((l) => (
                <option key={l.name} value={l.name}>{l.name}</option>
              ))}
            </select>
          )
        ) : (
          <span className="move-route-loc">{dest}</span>
        )}
      </div>

      {phase === "confirm" && eligible.length === 0 && (
        <Banner message={messages.noEligibleDestination()} />
      )}

      {error && (
        <Banner
          message={messages.moveFailed(error)}
          onDismiss={() => setError(null)}
        />
      )}

      {phase === "progress" && (
        <>
          {rail && <StepRail stops={rail} />}
          <Meter
            capacity={
              job ? { used: job.bytes_done, total: job.bytes_total } : null
            }
            tone="neutral"
          />
        </>
      )}

      {phase === "done" && rail && <StepRail stops={rail} />}
      {phase === "done" && <Banner message={messages.moveComplete()} />}
      {phase === "cancelled" && <Banner message={messages.moveCancelled()} />}
      {phase === "error" && job?.error && (
        <Banner message={messages.moveFailed(job.error)} />
      )}
    </Modal>
  );
}
