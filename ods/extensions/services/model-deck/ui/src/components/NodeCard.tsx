import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { humanizeAge, labels, messages } from "../model/messages";
import type { DeckNode, Placement } from "../model/nodes";
import Banner from "../ui/Banner";
import Panel from "../ui/Panel";
import HeaderEngineMenu from "./HeaderEngineMenu";
import ResourcePanel from "./ResourcePanel";
import type { SettingsTarget } from "./SettingsModal";

const DOT: Record<DeckNode["status"], string> = {
  reachable: "ui-pill-good",
  warming: "ui-pill-busy",
  unreachable: "ui-pill-bad",
  down: "ui-pill-bad",
};

export default function NodeCard({
  node,
  world,
  models,
  coldGgufs,
  spark,
  fetchError,
  settingsEngine,
  onChipClick,
  onOpenSettings,
  onRefresh,
}: {
  node: DeckNode;
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  spark: SparkStatus | null;
  /** This page's own fetch for the node failed. Note this can be set while
   * `node.status` still reads "reachable": that pill is the backend's
   * verdict, and disagreeing with it is exactly the point — everything below
   * is as old as the last successful poll. */
  fetchError: string | null;
  /** The node's configurable engine, or null when it has none. Non-null is
   * what puts the Engine settings button in the header; App decided it from
   * a catalog probe, because the presence of a harvested option catalog IS
   * the configurability signal (no invented flag anywhere in the payload). */
  settingsEngine: string | null;
  /** Threaded straight through to every ResourcePanel — the card has no
   * opinion about which chip was clicked, only App does. */
  onChipClick: (placement: Placement) => void;
  onOpenSettings: (target: SettingsTarget) => void;
  onRefresh: () => void;
}) {
  const unreachable = node.status === "unreachable";
  const age = humanizeAge(node.lastSeen);

  // Engine-level entry: no model in context, so the panel opens with the two
  // model-scoped tabs disabled. Rendered even while the node is unreachable
  // — settings are the deck's declared intent, which is exactly what an
  // operator wants to read when a node stops answering.
  const settingsButton = settingsEngine ? (
    <button
      type="button"
      title={labels.engineSettingsTitle}
      onClick={() => onOpenSettings({ node: node.id, engine: settingsEngine, model: null })}
    >
      {labels.engineSettings}
    </button>
  ) : undefined;

  // INTERIM SURFACE (Task 5) — see HeaderEngineMenu's own doc. Renders
  // nothing itself when the node has no hidden declared engine with a
  // usable load verb, so it is always safe to include here.
  const menu = <HeaderEngineMenu hiddenEngines={node.hiddenEngines} onRefresh={onRefresh} />;

  return (
    // The whole node desaturates as ONE unit when unreachable, so it reads as
    // a box going quiet rather than as individually greyed-out widgets. Its
    // placements stay on screen: unreachable is not empty.
    <Panel
      className={`node-card ${unreachable ? "node-stale" : ""}`}
      title={
        <>
          <span className="node-label">{node.label}</span>
          <span className={`ui-pill ${DOT[node.status]}`}>{node.status}</span>
          {/* Same fact as the per-chip caption below, so it comes from the
              same catalog entry. Any parenthesising is markup/CSS, never
              baked into the string — two hand-written renderings of one
              fact drift. */}
          {unreachable && age && (
            <span className="node-age">{messages.lastSeen(age).title}</span>
          )}
        </>
      }
      actions={
        <>
          {menu}
          {settingsButton}
        </>
      }
    >
      {node.servingLine && (
        <div className="node-serving-line">{node.servingLine}</div>
      )}
      {fetchError && <Banner message={messages.nodeFetchFailed(node.label, fetchError)} />}
      {unreachable && (
        <Banner message={messages.nodeUnreachable(node.label, age)} onAction={onRefresh} />
      )}
      {/* Without this, `down` was the one node status with a red pill and no
          explanation — an asynchronous swap failure (helper died, container
          never came up) rendered with nothing anywhere on screen saying so.
          The detail is the backend's own sentence; the adapter picks it. */}
      {node.status === "down" && (
        <Banner
          message={messages.nodeDown(node.label, node.detail ?? null)}
          onAction={onRefresh}
        />
      )}
      {node.status === "warming" && <Banner message={messages.warmingFirstBoot()} />}

      <div className="node-resources">
        {node.resources.map((r) => (
          <ResourcePanel
            key={r.id}
            resource={r}
            world={world}
            models={models}
            coldGgufs={coldGgufs}
            spark={spark}
            stale={unreachable}
            staleAge={age}
            nodeId={node.id}
            settingsEngine={settingsEngine}
            onChipClick={onChipClick}
            onOpenSettings={onOpenSettings}
            onRefresh={onRefresh}
          />
        ))}
      </div>
    </Panel>
  );
}
