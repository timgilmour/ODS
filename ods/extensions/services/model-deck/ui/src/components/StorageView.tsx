import { useState } from "react";
import {
  cancelStorageJob,
  postStorageRescan,
  registerLocation,
  type StorageJob,
  type StorageLocation,
  type StorageState,
} from "../api";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Meter from "../ui/Meter";
import Panel from "../ui/Panel";
import LocationCard from "./LocationCard";
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
 * otherwise a HOT band above a COLD band of LocationCards (build-design
 * decision 3 — hot-above-cold is visible structure, and each band scales to
 * any number of locations) + a jobs panel + Rescan. Owns the
 * move-confirmation flow (which unit/dest opens MoveModal) since that's
 * the one piece of state shared across cards (a drag from card A can be
 * dropped on card B). */
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

  const hot = storageState?.locations.filter((l) => l.role === "hot") ?? [];
  const cold = storageState?.locations.filter((l) => l.role === "cold") ?? [];

  return (
    <div className="storage-view">
      {error && <Banner message={messages.storageFetchFailed(error)} />}

      {!storageState ? (
        <div className="panel">{labels.loading}</div>
      ) : storageState.locations.length === 0 ? (
        <OnboardingPanel onChanged={onChanged} />
      ) : (
        <>
          <div className="storage-toolbar">
            <button type="button" onClick={handleRescan} disabled={rescanning}>
              {rescanning ? labels.rescanning : labels.rescan}
            </button>
            {rescanError && (
              <Banner
                message={messages.guardRefused(rescanError)}
                onDismiss={() => setRescanError(null)}
              />
            )}
          </div>

          {[
            { key: "hot", title: labels.hotBand, locations: hot },
            { key: "cold", title: labels.coldBand, locations: cold },
          ]
            .filter((band) => band.locations.length > 0)
            .map((band) => (
              <section key={band.key} className="storage-band">
                <div className="storage-band-title">{band.title}</div>
                <div className="storage-grid">
                  {band.locations.map((loc) => (
                    <LocationCard
                      key={loc.name}
                      location={loc}
                      units={storageState.units.filter((u) => u.location === loc.name)}
                      onRequestMove={requestMove}
                      onChanged={onChanged}
                    />
                  ))}
                </div>
              </section>
            ))}

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
    <Panel title="No storage locations yet">
      <p className="helper-text">
        1) Bind-mount a drive into the model-deck container in compose. 2) Register it here.
      </p>

      {registerError && (
        <Banner
          message={messages.guardRefused(registerError)}
          onDismiss={() => setRegisterError(null)}
        />
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
          type="button"
          className="primary"
          onClick={handleRegister}
          disabled={registering || !name.trim() || !path.trim()}
        >
          {registering ? "Registering…" : "Register"}
        </button>
      </div>
    </Panel>
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
    <Panel title={labels.jobsPanel} className="storage-jobs">
      {active.length === 0 ? (
        <div>{labels.noActiveMoves}</div>
      ) : (
        <ul className="storage-jobs-list">
          {active.map((job) => {
            const cancellable = job.state === "queued" || job.state === "copying" || job.state === "verifying";
            const stateClass =
              job.state === "failed" ? "ui-pill-bad" : job.state === "cancelled" ? "ui-pill-off" : "ui-pill-busy";
            return (
              <li key={job.id} className="storage-job-row">
                <div className="storage-job-head">
                  <span>{job.label}</span>
                  <span className={`ui-pill ${stateClass}`}>{job.state}</span>
                </div>
                <div className="storage-job-route">
                  {job.from} → {job.to}
                </div>
                {/* tone="neutral": this is a job PROGRESS fraction, not a
                    capacity reading — decision 5 says jobs stay
                    blue/in-progress regardless of how close to done they
                    are; the state chip beside it already carries
                    failed/cancelled. */}
                <Meter
                  capacity={{ used: job.bytes_done, total: job.bytes_total }}
                  tone="neutral"
                />
                {job.error && <div className="failed">{job.error}</div>}
                {rowError && rowError.id === job.id && (
                  <Banner
                    message={messages.guardRefused(rowError.message)}
                    onDismiss={() => setRowError(null)}
                  />
                )}
                {cancellable && (
                  <button type="button" onClick={() => handleCancel(job.id)} disabled={busyId === job.id}>
                    {labels.cancel}
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
