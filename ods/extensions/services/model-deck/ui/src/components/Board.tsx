import { useState } from "react";
import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import type { DeckNode, Placement } from "../model/nodes";
import { applyOrder, loadOrder, reorder, saveOrder } from "../model/nodeOrder";
import NodeCard from "./NodeCard";
import type { SettingsTarget } from "./SettingsModal";

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
  engineSettingsNodes,
  onChipClick,
  onOpenSettings,
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
  /** nodeId -> engine, for the nodes whose engine is configurable (App holds
   * the catalog probe that decides that). Same shape and same reason as
   * `nodeErrors`: Board stays node-agnostic and just looks each node up by
   * id, so a real node registry needs no change here. */
  engineSettingsNodes: Record<string, string>;
  /** A chip was clicked: App opens the model detail drawer on it. Threaded
   * rather than handled here because the drawer's subject has to be
   * re-derived from every later poll (App holds the state; see
   * findPlacement in model/nodes.ts). */
  onChipClick: (placement: Placement) => void;
  onOpenSettings: (target: SettingsTarget) => void;
  onRefresh: () => void;
}) {
  const [order, setOrder] = useState<string[]>(loadOrder);
  const [dragging, setDragging] = useState<string | null>(null);
  const ordered = applyOrder(nodes, order);

  function dropOn(targetId: string) {
    if (!dragging || dragging === targetId) return;
    const ids = reorder(ordered.map((n) => n.id), dragging, targetId);
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
          // Fires on every drag termination — successful drop, drop on a
          // non-card target, or an Escape-cancelled drag — unlike onDrop,
          // which only fires on a successful drop. Without it, "pick up a
          // card and let go outside any drop target" (or drop it back on
          // itself, which dropOn bails out of before clearing state) left
          // that card stuck at half opacity indefinitely.
          onDragEnd={() => setDragging(null)}
          className={dragging === node.id ? "node-dragging" : undefined}
        >
          <NodeCard
            node={node}
            world={world}
            models={models}
            coldGgufs={coldGgufs}
            spark={spark}
            fetchError={nodeErrors[node.id] ?? null}
            settingsEngine={engineSettingsNodes[node.id] ?? null}
            onChipClick={onChipClick}
            onOpenSettings={onOpenSettings}
            onRefresh={onRefresh}
          />
        </div>
      ))}
    </div>
  );
}
