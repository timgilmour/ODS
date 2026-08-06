import { useState } from "react";
import {
  bytesToGB,
  cancelStorageJob,
  postStorageRescan,
  registerLocation,
  type StorageJob,
  type StorageLocation,
  type StorageState,
} from "../api";
import LocationColumn from "./LocationColumn";
import MoveModal from "./MoveModal";

interface StorageViewProps {
  storageState: StorageState | null;
  error: string | null;
  onModalOpenChange: (open: boolean) => void;
  onChanged: () => void;
}

interface MoveRequest {
  unit: StorageState["units"][number];
  /** Preselected destination (a drag onto a specific card) or null (the
   * per-unit "Move…" path — the operator picks in the modal). */
  dest: string | null;
}

const ROLES: StorageLocation["role"][] = ["hot", "cold"];
const STORE_TYPES: StorageLocation["store_type"][] = ["gguf", "hf", "comfy", "plain"];
const ENGINES: StorageLocation["engine"][] = ["lemonade", "comfyui", "none"];

/** Storage tab root: onboarding when no locations are registered yet,
 * otherwise a row of LocationColumns + a jobs panel + Rescan. Owns the
 * move-confirmation flow (which unit/dest opens MoveModal) since that's
 * the one piece of state shared across columns (a drag from column A can
 * be dropped on column B). */
export default function StorageView({ storageState, error, onModalOpenChange, onChanged }: StorageViewProps) {
  const [moveRequest, setMoveRequest] = useState<MoveRequest | null>(null);
  const [rescanning, setRescanning] = useState(false);
  const [rescanError, setRescanError] = useState<string | null>(null);

  function requestMove(unitId: string, destName: string | null) {
    if (!storageState) return;
    const unit = storageState.units.find((u) => u.id === unitId);
    if (!unit || unit.state === "moving") return;
    setMoveRequest({ unit, dest: destName });
  }

  async function handleRescan() {
    setRescanning(true);
    setRescanError(null);
    try {
      await postStorageRescan();
    } catch (err) {
      setRescanError(err instanceof Error ? err.message : String(err));
    } finally {
      setRescanning(false);
      onChanged();
    }
  }

  return (
    <div className="storage-view">
      {error && (
        <div className="banner-error">
          <span>storage state failed: {error}</span>
        </div>
      )}

      {!storageState ? (
        <div className="panel">loading…</div>
      ) : storageState.locations.length === 0 ? (
        <OnboardingPanel onChanged={onChanged} />
      ) : (
        <>
          <div className="storage-toolbar">
            <button onClick={handleRescan} disabled={rescanning}>
              {rescanning ? "Rescanning…" : "Rescan"}
            </button>
            {rescanError && (
              <div className="banner-error">
                <span>{rescanError}</span>
                <button onClick={() => setRescanError(null)} aria-label="dismiss error">
                  ×
                </button>
              </div>
            )}
          </div>

          <div className="storage-row">
            {storageState.locations.map((loc) => (
              <LocationColumn
                key={loc.name}
                location={loc}
                units={storageState.units.filter((u) => u.location === loc.name)}
                onRequestMove={requestMove}
                onChanged={onChanged}
              />
            ))}
          </div>

          <JobsPanel jobs={storageState.jobs} onChanged={onChanged} />
        </>
      )}

      {moveRequest && (
        <MoveModal
          unit={moveRequest.unit}
          locations={storageState?.locations ?? []}
          initialDest={moveRequest.dest}
          jobs={storageState?.jobs ?? []}
          onModalOpenChange={onModalOpenChange}
          onClose={() => setMoveRequest(null)}
          onChanged={onChanged}
        />
      )}
    </div>
  );
}

function OnboardingPanel({ onChanged }: { onChanged: () => void }) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [role, setRole] = useState<StorageLocation["role"]>("hot");
  const [storeType, setStoreType] = useState<StorageLocation["store_type"]>("gguf");
  const [engine, setEngine] = useState<StorageLocation["engine"]>("none");
  const [registering, setRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);

  async function handleRegister() {
    setRegistering(true);
    setRegisterError(null);
    try {
      await registerLocation({
        name: name.trim(),
        path: path.trim(),
        role,
        store_type: storeType,
        engine,
        watermark_gb: null,
        archive_to: null,
        readonly: false,
      });
      setName("");
      setPath("");
      onChanged();
    } catch (err) {
      setRegisterError(err instanceof Error ? err.message : String(err));
    } finally {
      setRegistering(false);
    }
  }

  return (
    <div className="panel">
      <h2>No storage locations yet</h2>
      <p className="helper-text">
        1) Bind-mount a drive into the model-deck container in compose. 2) Register it here.
      </p>

      {registerError && (
        <div className="banner-error">
          <span>{registerError}</span>
          <button onClick={() => setRegisterError(null)} aria-label="dismiss error">
            ×
          </button>
        </div>
      )}

      <div className="storage-register-form">
        <label className="builder-field">
          name
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. cold-nvme"
          />
        </label>
        <label className="builder-field">
          path
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/mnt/cold/models"
          />
        </label>
        <label className="builder-field">
          role
          <select value={role} onChange={(e) => setRole(e.target.value as StorageLocation["role"])}>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label className="builder-field">
          store type
          <select
            value={storeType}
            onChange={(e) => setStoreType(e.target.value as StorageLocation["store_type"])}
          >
            {STORE_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label className="builder-field">
          engine
          <select value={engine} onChange={(e) => setEngine(e.target.value as StorageLocation["engine"])}>
            {ENGINES.map((e2) => (
              <option key={e2} value={e2}>
                {e2}
              </option>
            ))}
          </select>
        </label>
        <button
          className="primary"
          onClick={handleRegister}
          disabled={registering || !name.trim() || !path.trim()}
        >
          {registering ? "Registering…" : "Register"}
        </button>
      </div>
    </div>
  );
}

function JobsPanel({ jobs, onChanged }: { jobs: StorageJob[]; onChanged: () => void }) {
  const active = jobs.filter((j) => j.state !== "done");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);

  async function handleCancel(id: string) {
    setBusyId(id);
    try {
      await cancelStorageJob(id);
      setRowError(null);
    } catch (err) {
      setRowError({ id, message: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusyId(null);
      onChanged();
    }
  }

  return (
    <div className="panel storage-jobs">
      <h2>Moves</h2>
      {active.length === 0 ? (
        <div>no active moves</div>
      ) : (
        <ul className="storage-jobs-list">
          {active.map((job) => {
            const pct = job.bytes_total > 0 ? Math.min((job.bytes_done / job.bytes_total) * 100, 100) : 0;
            const cancellable = job.state === "queued" || job.state === "copying" || job.state === "verifying";
            return (
              <li key={job.id} className="storage-job-row">
                <div className="storage-job-head">
                  <span>{job.label}</span>
                  <span className={`chip chip-${job.state === "failed" ? "unknown" : job.state === "cancelled" ? "unloaded" : "busy"}`}>
                    {job.state}
                  </span>
                </div>
                <div className="storage-job-route">
                  {job.from} → {job.to}
                </div>
                <div className="meter-track">
                  <div className="meter-fill meter-neutral" style={{ width: `${pct}%` }} />
                </div>
                <div className="meter-label">
                  {bytesToGB(job.bytes_done)} / {bytesToGB(job.bytes_total)} GB
                </div>
                {job.error && <div className="failed">{job.error}</div>}
                {rowError && rowError.id === job.id && (
                  <div className="banner-error">
                    <span>{rowError.message}</span>
                    <button onClick={() => setRowError(null)} aria-label="dismiss error">
                      ×
                    </button>
                  </div>
                )}
                {cancellable && (
                  <button onClick={() => handleCancel(job.id)} disabled={busyId === job.id}>
                    Cancel
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
