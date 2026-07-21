import { useCallback, useEffect, useState } from "react";
import { getState, TOKEN_KEY, type StateResponse } from "./api";
import AdminGate from "./components/AdminGate";
import EventLog from "./components/EventLog";
import GpuColumn from "./components/GpuColumn";
import PolicyModal from "./components/PolicyModal";
import SetBuilder from "./components/SetBuilder";
import SetStrip from "./components/SetStrip";

const POLL_MS = 3000;

type View = "deck" | "builder";

export default function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) ?? "");
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [view, setView] = useState<View>("deck");
  const [modalOpen, setModalOpen] = useState(false);
  const [policyModalOpen, setPolicyModalOpen] = useState(false);
  // Bumped on every poll tick and after any mutating action; EventLog
  // re-fetches its own window whenever this changes (see EventLog.tsx).
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const refreshState = useCallback(async () => {
    try {
      const s = await getState();
      setState(s);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
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
        </nav>
        <button
          onClick={() => setPolicyModalOpen(true)}
          disabled={!token || !state}
          title={!token ? "admin token required" : undefined}
        >
          Policy
        </button>
        <AdminGate token={token} onTokenChange={setToken} />
      </header>

      {loadError && (
        <div className="load-error">state refresh failed: {loadError}</div>
      )}

      {view === "deck" && (
        <>
          <SetStrip token={token} onModalOpenChange={setModalOpen} onChanged={refreshAll} />

          {state && (
            <div className="gpu-row">
              {state.world.gpus.map((gpu) => (
                <GpuColumn
                  key={gpu.index}
                  gpu={gpu}
                  world={state.world}
                  policy={state.policy}
                  models={state.models}
                  token={token}
                  onRefresh={refreshAll}
                />
              ))}
            </div>
          )}
        </>
      )}

      {view === "builder" &&
        (state ? (
          <SetBuilder
            models={state.models}
            gpus={state.world.gpus}
            token={token}
            onModalOpenChange={setModalOpen}
          />
        ) : (
          <div className="panel">loading…</div>
        ))}

      {policyModalOpen && state && (
        <PolicyModal
          policy={state.policy}
          onClose={() => setPolicyModalOpen(false)}
          onSaved={refreshAll}
        />
      )}

      <EventLog refreshTrigger={refreshTrigger} />
    </>
  );
}
