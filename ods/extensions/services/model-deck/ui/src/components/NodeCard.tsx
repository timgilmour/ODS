import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { humanizeAge, messages } from "../model/messages";
import type { DeckNode } from "../model/nodes";
import Banner from "../ui/Banner";
import Panel from "../ui/Panel";
import ResourcePanel from "./ResourcePanel";

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
  onRefresh: () => void;
}) {
  const unreachable = node.status === "unreachable";
  const age = humanizeAge(node.lastSeen);

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
    >
      {fetchError && <Banner message={messages.nodeFetchFailed(node.label, fetchError)} />}
      {unreachable && (
        <Banner message={messages.nodeUnreachable(node.label, age)} onAction={onRefresh} />
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
            onRefresh={onRefresh}
          />
        ))}
      </div>
    </Panel>
  );
}
