import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { messages } from "../model/messages";
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
  return (
    <Panel
      className="node-card"
      title={
        <>
          <span className="node-label">{node.label}</span>
          <span className={`ui-pill ${DOT[node.status]}`}>{node.status}</span>
        </>
      }
    >
      {fetchError && <Banner message={messages.nodeFetchFailed(node.label, fetchError)} />}

      <div className="node-resources">
        {node.resources.map((r) => (
          <ResourcePanel
            key={r.id}
            resource={r}
            world={world}
            models={models}
            coldGgufs={coldGgufs}
            spark={spark}
            stale={node.status === "unreachable"}
            onRefresh={onRefresh}
          />
        ))}
      </div>
    </Panel>
  );
}
