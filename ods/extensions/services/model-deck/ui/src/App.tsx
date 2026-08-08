import { useCallback, useEffect, useState } from "react";
import {
  getCatalog,
  getSparkStatus,
  getState,
  getStorageState,
  type SparkStatus,
  type StateResponse,
  type StorageState,
} from "./api";
import Board from "./components/Board";
import EventsView from "./components/EventsView";
import PolicyModal from "./components/PolicyModal";
import SetBuilder from "./components/SetBuilder";
import SettingsModal, { type SettingsTarget } from "./components/SettingsModal";
import SetStrip from "./components/SetStrip";
import StorageView from "./components/StorageView";
import { labels, messages } from "./model/messages";
import { buildNodes, SPARK_NODE_ID } from "./model/nodes";
import Banner from "./ui/Banner";

const POLL_MS = 3000;

type View = "deck" | "builder" | "storage" | "events";

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [storageState, setStorageState] = useState<StorageState | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [spark, setSpark] = useState<SparkStatus | null>(null);
  // Separate from `spark` on purpose: the last-known status must survive a
  // failed poll (see the rejection handler below), but the failure itself
  // still has to reach the screen.
  const [sparkError, setSparkError] = useState<string | null>(null);
  const [view, setView] = useState<View>("deck");
  const [modalOpen, setModalOpen] = useState(false);
  const [policyModalOpen, setPolicyModalOpen] = useState(false);
  const [settingsTarget, setSettingsTarget] = useState<SettingsTarget | null>(null);
  // nodeId -> engine for the nodes that have an engine-level Settings entry.
  // Presence of a harvested option catalog IS the configurability signal —
  // there is no separate "configurable" flag anywhere in the backend, by
  // design — so this is a probe, not a lookup.
  const [engineSettingsNodes, setEngineSettingsNodes] = useState<Record<string, string>>({});
  // Bumped on every poll tick and after any mutating action; EventsView
  // re-fetches its own window whenever this changes (see EventsView.tsx).
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
        (s) => {
          setSpark(s);
          setSparkError(null);
        },
        // Deliberately does NOT clear `spark`. getSparkStatus resolves to
        // null only for a 503 — "no spark configured on this deployment",
        // which correctly removes the card. Every other failure means a
        // CONFIGURED node we currently cannot reach, and the adapter's whole
        // contract is that such a node keeps its card and its last-known
        // placements, marked stale. Overwriting with null here would blank
        // the node on the first failed poll and silently defeat that rule.
        //
        // Keeping the data is not the same as hiding the failure, though.
        // buildSparkNode derives reachability from the BACKEND's lifecycle
        // view, so if THIS page's fetch is what broke, the node would keep
        // rendering a confident status pill over data that stopped updating
        // minutes ago. The error is recorded here and banner-ed on the node
        // it belongs to, which is the one thing this handler must not skip.
        (err) => setSparkError(err instanceof Error ? err.message : String(err)),
      ),
    ]);
  }, []);

  const refreshAll = useCallback(async () => {
    await refreshState();
    setRefreshTrigger((t) => t + 1);
  }, [refreshState]);

  // Poll every 3s; paused entirely while an apply confirmation modal (Deck's
  // SetStrip, SetBuilder's own preview, the policy editor, or the settings
  // panel) is open. Settings joins the list for the same reason the others
  // are on it: it holds uncommitted operator edits, and a refresh underneath
  // one is a re-render of the surface being typed into.
  useEffect(() => {
    if (modalOpen || policyModalOpen || settingsTarget) return;
    refreshAll();
    const id = setInterval(refreshAll, POLL_MS);
    return () => clearInterval(id);
  }, [modalOpen, policyModalOpen, settingsTarget, refreshAll]);

  // One probe, once, on mount. `(spark_node_id(), "vllm")` is the only
  // configurable engine pair the deck wires today (app/main.py:245-247 builds
  // `configurable_engines` from exactly that one route), and a non-null
  // catalog for it is what earns the node its Engine settings button. A null
  // catalog (never harvested) or a failed probe leaves the map empty, so the
  // button is absent rather than present-and-broken.
  useEffect(() => {
    let alive = true;
    getCatalog(SPARK_NODE_ID, "vllm").then(
      (catalog) => {
        if (alive && catalog) setEngineSettingsNodes({ [SPARK_NODE_ID]: "vllm" });
      },
      () => {},
    );
    return () => {
      alive = false;
    };
  }, []);

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
          <h1>{labels.appTitle}</h1>
          <div className="deck-subtitle">{labels.appSubtitle}</div>
        </div>
        <nav className="view-tabs">
          <button
            className={view === "deck" ? "primary" : undefined}
            onClick={() => setView("deck")}
          >
            {labels.deck}
          </button>
          <button
            className={view === "builder" ? "primary" : undefined}
            onClick={() => setView("builder")}
          >
            {labels.setBuilder}
          </button>
          <button
            className={view === "storage" ? "primary" : undefined}
            onClick={() => setView("storage")}
          >
            {labels.storage}
          </button>
          <button
            className={view === "events" ? "primary" : undefined}
            onClick={() => setView("events")}
          >
            {labels.events}
          </button>
        </nav>
        <button onClick={() => setPolicyModalOpen(true)} disabled={!state}>
          {labels.policy}
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
              // App owns the fetches, so App is what knows which node an
              // error belongs to. Board stays node-agnostic: it just looks
              // each node up by id.
              nodeErrors={{ [SPARK_NODE_ID]: sparkError }}
              engineSettingsNodes={engineSettingsNodes}
              onOpenSettings={setSettingsTarget}
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
            nodeLabel={state.node.label}
            onModalOpenChange={setModalOpen}
          />
        ) : (
          <div className="panel">{labels.loading}</div>
        ))}

      {view === "storage" && (
        <StorageView
          storageState={storageState}
          error={storageError}
          onModalOpenChange={setModalOpen}
          onChanged={refreshAll}
        />
      )}

      {view === "events" && <EventsView refreshTrigger={refreshTrigger} />}

      {policyModalOpen && state && (
        <PolicyModal
          policy={state.policy}
          storageState={storageState}
          onClose={() => setPolicyModalOpen(false)}
          onSaved={refreshAll}
        />
      )}

      {settingsTarget && (
        // Keyed on the target so a NEW target remounts the panel rather than
        // reusing the instance. Without it, opening Settings for a second
        // model while one is already open carries the whole panel's state
        // across: the edit buffer (pending edits for model A would save into
        // model B's scopes — the buffer is keyed by write KIND, which says
        // nothing about which model it was typed for), the selected kind (an
        // engine-only target stranded on "engine_models" makes toPuts throw
        // its developer-facing message into the save banner), the last good
        // argline, and the import warnings. Only the fetch effect re-runs on
        // a prop change; useState initializers do not. Unreachable from
        // today's single entry point — Tasks 10 and 11 add exactly the call
        // sites that arm it.
        <SettingsModal
          key={`${settingsTarget.node}/${settingsTarget.engine}/${settingsTarget.model ?? ""}`}
          target={settingsTarget}
          onClose={() => setSettingsTarget(null)}
          onSaved={refreshAll}
        />
      )}
    </>
  );
}
