import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import type { DeckNode } from "../model/nodes";
import NodeCard from "./NodeCard";

/** Nodes stack vertically, arbitrary count. Resources within a node lay out
 * as a responsive grid, so a two-GPU box reads as two columns and a
 * single-slot node reads as one wide panel — with no per-node special
 * casing anywhere in this file. */
export default function Board({
  nodes,
  world,
  models,
  coldGgufs,
  spark,
  onRefresh,
}: {
  nodes: DeckNode[];
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  spark: SparkStatus | null;
  onRefresh: () => void;
}) {
  return (
    <div className="board">
      {nodes.map((node) => (
        <NodeCard
          key={node.id}
          node={node}
          world={world}
          models={models}
          coldGgufs={coldGgufs}
          spark={spark}
          onRefresh={onRefresh}
        />
      ))}
    </div>
  );
}
