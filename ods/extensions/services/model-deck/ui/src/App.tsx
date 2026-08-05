import { useCallback, useEffect, useState } from "react";
import {
  getSparkStatus,
  getState,
  getStorageState,
  type SparkStatus,
  type StateResponse,
  type StorageState,
} from "./api";
import Board from "./components/Board";
import EventLog from "./components/EventLog";
import PolicyModal from "./components/PolicyModal";
import SetBuilder from "./components/SetBuilder";
import SetStrip from "./components/SetStrip";
import StorageView from "./components/StorageView";
import { messages } from "./model/messages";
import { buildNodes } from "./model/nodes";
import Banner from "./ui/Banner";

const POLL_MS = 3000;

type View = "deck" | "builder" | "storage";

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [storageState, setStorageState] = useState<StorageState | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [spark, setSpark] = useState<SparkStatus | null>(null);
  const [view, setView] = useState<View>("deck");
  const [modalOpen, setModalOpen] = useState(false);
  const [policyModalOpen, setPolicyModalOpen] = useState(false);
  // Bumped on every poll tick and after any mutating action; EventLog
  // re-fetches its own window whenever this changes (see EventLog.tsx).
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Fetches all three in parallel (Promise.all-style), but each leg's own
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
      getSparkStatus().then(
        (s) => setSpark(s),
        // Deliberately does NOT clear `spark`. getSparkStatus resolves to
        // null only for a 503 — "no spark configured on this deployment",
        // which correctly removes the card. Every other failure means a
        // CONFIGURED node we currently cannot reach, and the adapter's whole
        // contract is that such a node keeps its card and its last-known
        // placements, marked stale. Overwriting with null here would blank
        // the node on the first failed poll and silently defeat that rule.
        () => {},
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

      {loadError && <Banner message={messages.stateRefreshFailed(loadError)} />}

      {view === "deck" && (
        <>
          <SetStrip onModalOpenChange={setModalOpen} onChanged={refreshAll} />

          {state && (
            <Board
              nodes={buildNodes(state, spark)}
              world={state.world}
              models={state.models}
              coldGgufs={coldGgufs}
              spark={spark}
              onRefresh={refreshAll}
            />
          )}
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
