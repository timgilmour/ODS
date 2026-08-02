import { useCallback, useEffect, useState } from "react";
import { getState, getStorageState, type StateResponse, type StorageState } from "./api";
import EventLog from "./components/EventLog";
import GpuColumn from "./components/GpuColumn";
import PolicyModal from "./components/PolicyModal";
import SetBuilder from "./components/SetBuilder";
import SparkCard from "./components/SparkCard";
import SetStrip from "./components/SetStrip";
import StorageView from "./components/StorageView";

const POLL_MS = 3000;

type View = "deck" | "builder" | "storage";

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [storageState, setStorageState] = useState<StorageState | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [view, setView] = useState<View>("deck");
  const [modalOpen, setModalOpen] = useState(false);
  const [policyModalOpen, setPolicyModalOpen] = useState(false);
  // Bumped on every poll tick and after any mutating action; EventLog
  // re-fetches its own window whenever this changes (see EventLog.tsx).
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Fetches both in parallel (Promise.all-style), but each leg's own
  // .then/.catch means a storage failure can never reject the whole call
  // or clobber the main `state` — the Deck and Set Builder tabs must keep
  // working even when the storage backend is unreachable/misconfigured;
  // only the Storage tab shows storageError.
  const refreshState = useCallback(async () => {
    await Promise.all([
      getState().then(
        (s) => {
          setState(s);
          setLoadError(null);
        },
        (err) => setLoadError(err instanceof Error ? err.message : String(err)),
      ),
      getStorageState().then(
        (ss) => {
          setStorageState(ss);
          setStorageError(null);
        },
        (err) => {
          setStorageState(null);
          setStorageError(err instanceof Error ? err.message : String(err));
        },
      ),
    ]);
  }, []);

  const refreshAll = useCallback(async () => {
    await refreshState();
    setRefreshTrigger((t) => t + 1);
  }, [refreshState]);

  // Poll every 3s; paused entirely while an apply confirmation modal (Deck's
  // SetStrip, SetBuilder's own preview, or the policy editor) is open.
  useEffect(() => {
    if (modalOpen || policyModalOpen) return;
    refreshAll();
    const id = setInterval(refreshAll, POLL_MS);
    return () => clearInterval(id);
  }, [modalOpen, policyModalOpen, refreshAll]);

  // Resident GGUFs sitting on a location that isn't lemonade's hot mount —
  // i.e. cold from lemonade's point of view. Defaults to [] so the lemonade
  // card's Load dropdown renders fine even when storage is down/unconfigured.
  const coldGgufs =
    storageState?.units.filter(
      (u) =>
        u.type === "gguf" &&
        u.state === "resident" &&
        storageState.locations.find((l) => l.name === u.location)?.engine !== "lemonade",
    ) ?? [];

  return (
    <>
      <header className="deck-header">
        <div>
          <h1>Model Deck</h1>
          <div className="deck-subtitle">GPU/VRAM control for lemonade, ComfyUI, hipfire</div>
        </div>
        <nav className="view-tabs">
          <button
            className={view === "deck" ? "primary" : undefined}
            onClick={() => setView("deck")}
          >
            Deck
          </button>
          <button
            className={view === "builder" ? "primary" : undefined}
            onClick={() => setView("builder")}
          >
            Set Builder
          </button>
          <button
            className={view === "storage" ? "primary" : undefined}
            onClick={() => setView("storage")}
          >
            Storage
          </button>
        </nav>
        <button onClick={() => setPolicyModalOpen(true)} disabled={!state}>
          Policy
        </button>
      </header>

      {loadError && (
        <div className="load-error">state refresh failed: {loadError}</div>
      )}

      {view === "deck" && (
        <>
          <SetStrip onModalOpenChange={setModalOpen} onChanged={refreshAll} />

          {state && (
            <div className="gpu-row">
              {state.world.gpus.map((gpu) => (
                <GpuColumn
                  key={gpu.index}
                  gpu={gpu}
                  world={state.world}
                  policy={state.policy}
                  models={state.models}
                  coldGgufs={coldGgufs}
                  onRefresh={refreshAll}
                />
              ))}
            </div>
          )}

          <SparkCard refreshTrigger={refreshTrigger} onChanged={refreshAll} />
        </>
      )}

      {view === "builder" &&
        (state ? (
          <SetBuilder
            models={state.models}
            gpus={state.world.gpus}
            world={state.world}
            onModalOpenChange={setModalOpen}
          />
        ) : (
          <div className="panel">loading…</div>
        ))}

      {view === "storage" && (
        <StorageView
          storageState={storageState}
          error={storageError}
          onModalOpenChange={setModalOpen}
          onChanged={refreshAll}
        />
      )}

      {policyModalOpen && state && (
        <PolicyModal
          policy={state.policy}
          storageState={storageState}
          onClose={() => setPolicyModalOpen(false)}
          onSaved={refreshAll}
        />
      )}

      <EventLog refreshTrigger={refreshTrigger} />
    </>
  );
}
