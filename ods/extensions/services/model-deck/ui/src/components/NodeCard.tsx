import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import type { DeckNode } from "../model/nodes";
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
  onRefresh,
}: {
  node: DeckNode;
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  spark: SparkStatus | null;
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
