import { useCallback, useEffect, useState } from "react";
import { getState, TOKEN_KEY, type StateResponse } from "./api";
import AdminGate from "./components/AdminGate";
import EventLog from "./components/EventLog";
import GpuColumn from "./components/GpuColumn";
import SetStrip from "./components/SetStrip";

const POLL_MS = 3000;

export default function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) ?? "");
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
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

  // Poll every 3s; paused entirely while an apply confirmation modal is open.
  useEffect(() => {
    if (modalOpen) return;
    refreshAll();
    const id = setInterval(refreshAll, POLL_MS);
    return () => clearInterval(id);
  }, [modalOpen, refreshAll]);

  return (
    <>
      <header className="deck-header">
        <div>
          <h1>Model Deck</h1>
          <div className="deck-subtitle">GPU/VRAM control for lemonade, ComfyUI, hipfire</div>
        </div>
        <AdminGate token={token} onTokenChange={setToken} />
      </header>

      {loadError && (
        <div className="load-error">state refresh failed: {loadError}</div>
      )}

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

      <EventLog refreshTrigger={refreshTrigger} />
    </>
  );
}
