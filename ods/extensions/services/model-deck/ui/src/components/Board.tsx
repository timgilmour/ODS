import { useState } from "react";
import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import type { DeckNode } from "../model/nodes";
import { applyOrder, loadOrder, saveOrder } from "../model/nodeOrder";
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
  nodeErrors,
  onRefresh,
}: {
  nodes: DeckNode[];
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  spark: SparkStatus | null;
  /** Per-node fetch failures, keyed by node id: "this page could not reach
   * that node's endpoint", which is a different claim from the backend's own
   * reachability verdict in `node.status`. A plain map rather than a
   * spark-shaped prop, so a real node registry needs no change here. */
  nodeErrors: Record<string, string | null>;
  onRefresh: () => void;
}) {
  const [order, setOrder] = useState<string[]>(loadOrder);
  const [dragging, setDragging] = useState<string | null>(null);
  const ordered = applyOrder(nodes, order);

  function dropOn(targetId: string) {
    if (!dragging || dragging === targetId) return;
    const ids = ordered.map((n) => n.id).filter((id) => id !== dragging);
    ids.splice(ids.indexOf(targetId), 0, dragging);
    setOrder(ids);
    saveOrder(ids);
    setDragging(null);
  }

  return (
    <div className="board">
      {ordered.map((node) => (
        <div
          key={node.id}
          draggable
          onDragStart={() => setDragging(node.id)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={() => dropOn(node.id)}
          className={dragging === node.id ? "node-dragging" : undefined}
        >
          <NodeCard
            node={node}
            world={world}
            models={models}
            coldGgufs={coldGgufs}
            spark={spark}
            fetchError={nodeErrors[node.id] ?? null}
            onRefresh={onRefresh}
          />
        </div>
      ))}
    </div>
  );
}
