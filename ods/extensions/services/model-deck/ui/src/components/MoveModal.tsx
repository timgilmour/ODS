import { useEffect, useState } from "react";
import {
  bytesToGB,
  cancelStorageJob,
  meterFillClass,
  postStorageMove,
  truncateMiddle,
  type StorageJob,
  type StorageLocation,
  type StorageUnit,
} from "../api";

type Phase = "confirm" | "progress" | "done" | "cancelled" | "error";

const NON_TERMINAL: ReadonlySet<StorageJob["state"]> = new Set(["queued", "copying", "verifying"]);

interface MoveModalProps {
  unit: StorageUnit;
  dest: StorageLocation;
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
export default function MoveModal({ unit, dest, jobs, onModalOpenChange, onClose, onChanged }: MoveModalProps) {
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
    setBusy(true);
    setError(null);
    try {
      const { job: created } = await postStorageMove(unit.id, dest.name);
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

  const pct = job && job.bytes_total > 0 ? Math.min((job.bytes_done / job.bytes_total) * 100, 100) : 0;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-box">
        <h3>Move {truncateMiddle(unit.name)}</h3>
        <p className="modal-notes">
          {unit.location} → {dest.name} · {bytesToGB(unit.size)} GB
        </p>

        {error && (
          <div className="banner-error">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="dismiss error">
              ×
            </button>
          </div>
        )}

        {phase === "progress" && (
          <div className="move-progress">
            <div className="meter-track">
              <div className={meterFillClass(pct)} style={{ width: `${pct}%` }} />
            </div>
            <div className="meter-label">
              {job ? (
                <>
                  {bytesToGB(job.bytes_done)} / {bytesToGB(job.bytes_total)} GB — {job.state}
                </>
              ) : (
                "starting…"
              )}
            </div>
          </div>
        )}

        {phase === "done" && <p>Move complete.</p>}
        {phase === "cancelled" && <p>Move cancelled.</p>}
        {phase === "error" && job?.error && <p className="failed">{job.error}</p>}

        <div className="modal-actions">
          {phase === "confirm" && (
            <>
              <button onClick={onClose} disabled={busy}>
                Cancel
              </button>
              <button className="primary" onClick={handleConfirm} disabled={busy}>
                {busy ? "Starting…" : "Confirm"}
              </button>
            </>
          )}
          {phase === "progress" && (
            <button onClick={handleCancel} disabled={busy || !job || !NON_TERMINAL.has(job.state)}>
              Cancel move
            </button>
          )}
          {(phase === "done" || phase === "cancelled" || phase === "error") && (
            <button className="primary" onClick={handleClose}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
