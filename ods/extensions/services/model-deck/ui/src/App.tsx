import { useCallback, useEffect, useState } from "react";
import {
  getCatalog,
  getEngineKinds,
  getNodeServingStatus,
  getState,
  getStorageState,
  type EngineKindsResponse,
  type SparkStatus,
  type StateResponse,
  type StorageState,
} from "./api";
import Board from "./components/Board";
import EventsView from "./components/EventsView";
import ModelDetailDrawer from "./components/ModelDetailDrawer";
import NodesView from "./components/NodesView";
import PolicyModal from "./components/PolicyModal";
import SetBuilder from "./components/SetBuilder";
import SettingsModal, { type SettingsTarget } from "./components/SettingsModal";
import SetStrip from "./components/SetStrip";
import StorageView from "./components/StorageView";
import { resourceKindMap } from "./model/engineForm";
import { labels, messages } from "./model/messages";
import { buildNodes, findPlacement, swapNodes, type Placement } from "./model/nodes";
import Banner from "./ui/Banner";

const POLL_MS = 3000;

type View = "deck" | "builder" | "storage" | "nodes" | "events";

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [storageState, setStorageState] = useState<StorageState | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [serving, setServing] = useState<Record<string, SparkStatus | null>>({});
  // Last-known statuses survive failed polls per node (same contract the
  // single spark fetch had); failures land per node id for the banner.
  const [servingErrors, setServingErrors] = useState<Record<string, string | null>>({});
  const [view, setView] = useState<View>("deck");
  const [modalOpen, setModalOpen] = useState(false);
  const [policyModalOpen, setPolicyModalOpen] = useState(false);
  const [settingsTarget, setSettingsTarget] = useState<SettingsTarget | null>(null);
  // nodeId -> engine for the nodes that have an engine-level Settings entry.
  // Presence of a harvested option catalog IS the configurability signal —
  // there is no separate "configurable" flag anywhere in the backend, by
  // design — so this is a probe, not a lookup.
  const [engineSettingsNodes, setEngineSettingsNodes] = useState<Record<string, string>>({});
  // GET /api/engine-kinds — the verb vocabulary for every DECLARED REMOTE
  // engine on the board (spec §5: never a UI literal). One fetch, not per
  // poll: it is a static catalog of the backend's own KNOWN_KINDS
  // (app/routers/nodes.py:476-504), and nothing an operator does changes it.
  const [engineKinds, setEngineKinds] = useState<EngineKindsResponse | null>(null);
  const [engineKindsError, setEngineKindsError] = useState<string | null>(null);
  // The placement the detail drawer was opened on. Held as the LAST-KNOWN
  // object only: the drawer's live subject is re-derived from each poll by id
  // (see `detailSpot` below), so this never drives the status pill.
  const [detailPlacement, setDetailPlacement] = useState<Placement | null>(null);
  // Bumped on every poll tick and after any mutating action; EventsView
  // re-fetches its own window whenever this changes (see EventsView.tsx).
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  // The board's "+ add engine here" click (INST I1) — which node and GPU to
  // pre-seed the Nodes screen's create-instance form with. Set by
  // `onAddEngineHere` below and consumed exactly once by NodesView
  // (`onSeedConsumed`), which is also what clears it — App holds it only
  // long enough to hand it to the screen the operator is about to land on.
  const [instanceSeed, setInstanceSeed] =
    useState<{ nodeId: string; gpuIndex: number } | null>(null);

  // Fetches all three in parallel (Promise.all-style), but each leg's own
  // .then/.catch means a storage failure can never reject the whole call
  // or clobber the main `state` — the Deck and Set Builder tabs must keep
  // working even when the storage backend is unreachable/misconfigured;
  // only the Storage tab shows storageError.
  const refreshState = useCallback(async () => {
    await Promise.all([
      getState().then(
        async (s) => {
          setState(s);
          setLoadError(null);
          const ids = swapNodes(s).map((e) => e.id);
          // Prune nodes that left the registry/demoted, keep last-known for
          // the rest (the per-node analogue of the old don't-clear rule
          // below).
          setServing((prev) => Object.fromEntries(
            Object.entries(prev).filter(([id]) => ids.includes(id))));
          setServingErrors((prev) => Object.fromEntries(
            Object.entries(prev).filter(([id]) => ids.includes(id))));
          await Promise.all(ids.map((id) =>
            getNodeServingStatus(id).then(
              (st) => {
                setServing((prev) => ({ ...prev, [id]: st }));
                setServingErrors((prev) => ({ ...prev, [id]: null }));
              },
              // Deliberately does NOT clear this node's `serving` entry.
              // getNodeServingStatus resolves to null only for a 503 — "not
              // operable on this deployment", which correctly falls back to
              // the observe-only card. Every other failure means a
              // DECLARED-operable node we currently cannot reach, and the
              // adapter's whole contract is that such a node keeps its card
              // and its last-known placements, marked stale. Overwriting
              // with null here would blank the node on the first failed
              // poll and silently defeat that rule.
              //
              // Keeping the data is not the same as hiding the failure,
              // though. buildSwapNode derives reachability from the
              // BACKEND's lifecycle view, so if THIS page's fetch is what
              // broke, the node would keep rendering a confident status
              // pill over data that stopped updating minutes ago. The error
              // is recorded here and banner-ed on the node it belongs to,
              // which is the one thing this handler must not skip.
              (err) =>
                setServingErrors((prev) => ({
                  ...prev,
                  [id]: err instanceof Error ? err.message : String(err),
                })),
            )));
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

  // Its own fetch, and its own error: a failed catalog leaves every remote
  // engine card rendering its status with NO verbs (model/engineVerbs.ts
  // returns none for a null catalog — an invented verb would be worse than
  // an absent one), which is a thing to say out loud rather than a set of
  // silently missing buttons. Retried on demand from the banner; never on
  // the poll.
  const loadEngineKinds = useCallback(() => {
    getEngineKinds().then(
      (k) => {
        setEngineKinds(k);
        setEngineKindsError(null);
      },
      (err) => setEngineKindsError(err instanceof Error ? err.message : String(err)),
    );
  }, []);

  useEffect(loadEngineKinds, [loadEngineKinds]);

  const closeDetail = useCallback(() => setDetailPlacement(null), []);

  /** Switching tabs closes the drawer. It is a detail OF the board — leaving
   * it floating over Storage or Events would put a model's verbs on a screen
   * that has nothing to do with them. */
  const showView = useCallback((next: View) => {
    setView(next);
    setDetailPlacement(null);
  }, []);

  // The board's "+ add engine here" entry point (INST I1) — seeds the
  // Nodes screen's create-instance form with the card the operator clicked
  // and switches to it, in the same click. `showView` (above) already
  // clears `detailPlacement`, which this wants too: the seed and an open
  // model-detail drawer are two different reasons to be on the Nodes
  // screen, and only one of them makes sense at a time.
  const onAddEngineHere = useCallback((nodeId: string, gpuIndex: number) => {
    setInstanceSeed({ nodeId, gpuIndex });
    showView("nodes");
  }, [showView]);

  // Poll every 3s; paused entirely while an apply confirmation modal (Deck's
  // SetStrip, SetBuilder's own preview, the policy editor, or the settings
  // panel) is open. Settings joins the list for the same reason the others
  // are on it: it holds uncommitted operator edits, and a refresh underneath
  // one is a re-render of the surface being typed into.
  //
  // The model detail drawer is deliberately NOT on that list. Its headline is
  // a live status pill over a placement the deck may unload out from under it,
  // so a paused poll would freeze the one thing it exists to report. Its
  // free-text fields are seeded once per FACTS KEY rather than re-read on
  // every tick, which is what makes a live refresh safe there — and, because
  // the key (not a one-shot flag) is what gates it, a spark swap under a
  // stable placement id still re-seeds onto the new model
  // (ModelDetailDrawer's `seededKey`).
  useEffect(() => {
    if (modalOpen || policyModalOpen || settingsTarget) return;
    refreshAll();
    const id = setInterval(refreshAll, POLL_MS);
    return () => clearInterval(id);
  }, [modalOpen, policyModalOpen, settingsTarget, refreshAll]);

  // One probe per swap node, re-run whenever the swap-node id list changes.
  // `(nodeId, "vllm")` is the only configurable engine pair the deck wires
  // today (app/main.py:290-301's routes loop builds `deck["configurable_engines"]`
  // as exactly one `(node_id, "vllm")` pair per control:"swap" node), and a
  // non-null catalog for a node is what earns it the Engine settings button.
  // A null catalog (never harvested) or a failed probe leaves that node out
  // of the map, so the button is absent rather than present-and-broken.
  // Keyed on the joined id list rather than `state` itself, so an unrelated
  // poll (world/lifecycle changing, the swap-node set staying put) does not
  // re-fire every node's probe.
  const swapIds = swapNodes(state).map((e) => e.id).join(",");
  useEffect(() => {
    if (!swapIds) return;
    let alive = true;
    for (const id of swapIds.split(",")) {
      getCatalog(id, "vllm").then(
        (catalog) => {
          if (alive && catalog)
            setEngineSettingsNodes((prev) => ({ ...prev, [id]: "vllm" }));
        },
        () => {},
      );
    }
    return () => {
      alive = false;
    };
  }, [swapIds]);

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

  const nodes = buildNodes(state, serving, engineKinds);
  // Re-derived on EVERY render, i.e. after every poll: the drawer must read
  // its subject out of the freshest board rather than out of the object the
  // chip click captured, or it would keep showing "serving" over a model that
  // was unloaded minutes ago. `null` means the placement left the board
  // entirely — the drawer then keeps the last-known object and says so.
  const detailSpot = detailPlacement ? findPlacement(nodes, detailPlacement.id) : null;

  return (
    <>
      <header className="deck-header">
        <div>
          <h1>{labels.appTitle}</h1>
          <div className="deck-subtitle">{labels.appSubtitle}</div>
        </div>
        <nav className="view-tabs">
          <button type="button"
            className={view === "deck" ? "primary" : undefined}
            onClick={() => showView("deck")}
          >
            {labels.deck}
          </button>
          <button type="button"
            className={view === "builder" ? "primary" : undefined}
            onClick={() => showView("builder")}
          >
            {labels.setBuilder}
          </button>
          <button type="button"
            className={view === "storage" ? "primary" : undefined}
            onClick={() => showView("storage")}
          >
            {labels.storage}
          </button>
          <button type="button"
            className={view === "nodes" ? "primary" : undefined}
            onClick={() => showView("nodes")}
          >
            {labels.nodes}
          </button>
          <button type="button"
            className={view === "events" ? "primary" : undefined}
            onClick={() => showView("events")}
          >
            {labels.events}
          </button>
        </nav>
        <button type="button" onClick={() => setPolicyModalOpen(true)} disabled={!state}>
          {labels.policy}
        </button>
      </header>

      {loadError && <Banner message={messages.stateRefreshFailed(loadError)} />}
      {engineKindsError && (
        <Banner
          message={messages.engineKindsFailed(engineKindsError)}
          onAction={loadEngineKinds}
        />
      )}

      {view === "deck" && (
        <>
          <SetStrip onModalOpenChange={setModalOpen} onChanged={refreshAll} />

          {state && (
            <Board
              nodes={nodes}
              world={state.world}
              models={state.models}
              coldGgufs={coldGgufs}
              serving={serving}
              // App owns the fetches, so App is what knows which node an
              // error belongs to. Board stays node-agnostic: it just looks
              // each node up by id.
              nodeErrors={servingErrors}
              engineSettingsNodes={engineSettingsNodes}
              onChipClick={setDetailPlacement}
              onOpenSettings={setSettingsTarget}
              onRefresh={refreshAll}
              onAddEngineHere={onAddEngineHere}
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

      {view === "nodes" && (
        <NodesView
          nodes={state?.nodes ?? []}
          gpus={state?.world.gpus ?? []}
          policy={state?.policy ?? {}}
          onChanged={refreshAll}
          instanceSeed={instanceSeed}
          onSeedConsumed={() => setInstanceSeed(null)}
        />
      )}

      {view === "events" && <EventsView refreshTrigger={refreshTrigger} />}

      {policyModalOpen && state && (
        <PolicyModal
          policy={state.policy}
          storageState={storageState}
          kinds={engineKinds?.kinds ?? null}
          resourceKinds={resourceKindMap(state?.world?.tenants, state?.world?.remote_tenants)}
          onClose={() => setPolicyModalOpen(false)}
          onSaved={refreshAll}
        />
      )}

      {detailPlacement && state && (
        // Keyed on the placement id for the same reason SettingsModal is keyed
        // on its target: clicking a second chip while one drawer is open must
        // START the new one, not inherit the first's fetched facts and seeded
        // text drafts (useState initializers and the `seededKey` ref do not
        // re-run on a prop change).
        //
        // `placement` is the freshly re-derived one wherever the board still
        // carries it, and the last-known object otherwise — which is what
        // `placedOn: null` then tells the drawer, so it drops its verbs and
        // says the placement is gone rather than offering actions against
        // something that is no longer there.
        <ModelDetailDrawer
          key={detailPlacement.id}
          placement={detailSpot?.placement ?? detailPlacement}
          placedOn={
            detailSpot ? { nodeId: detailSpot.node.id, resource: detailSpot.resource } : null
          }
          world={state.world}
          models={state.models}
          coldGgufs={coldGgufs}
          refreshTrigger={refreshTrigger}
          engineSettingsNodes={engineSettingsNodes}
          suspended={settingsTarget !== null}
          onOpenSettings={setSettingsTarget}
          onRefresh={refreshAll}
          onClose={closeDetail}
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
