import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { messages } from "../model/messages";
import { isTenantName, type DeckResource } from "../model/nodes";
import Meter from "../ui/Meter";
import ModelChip from "../ui/ModelChip";
import Panel from "../ui/Panel";
import PlacementActions from "./PlacementActions";
import SparkSwap from "./SparkSwap";

export default function ResourcePanel({
  resource,
  world,
  models,
  coldGgufs,
  spark,
  stale,
  onRefresh,
}: {
  resource: DeckResource;
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  spark: SparkStatus | null;
  /** True when the owning node is unreachable: show what we last knew, but
   * offer no verbs, because none of them can currently reach anything. */
  stale: boolean;
  onRefresh: () => void;
}) {
  return (
    <Panel title={resource.label} className="resource-panel">
      <Meter capacity={resource.capacity} />

      {resource.placements.length === 0 ? (
        <div className="dropzone-empty">{messages.emptySlot().title}</div>
      ) : (
        resource.placements.map((p) => (
          <div key={p.id} className="resource-placement">
            <ModelChip placement={p} />
          </div>
        ))
      )}

      {!stale &&
        resource.controls.map((control) =>
          control === "spark" ? (
            spark && <SparkSwap key={control} spark={spark} onChanged={onRefresh} />
          ) : isTenantName(control) ? (
            <PlacementActions
              key={control}
              tenant={control}
              world={world}
              models={models}
              // Whether this tenant already has a chip on this resource. A
              // parked hipfire deliberately has none, and then the control
              // row is the only thing that can say which tenant it is and
              // what state it is in. Computed here because the placements
              // are already in hand.
              hasPlacement={resource.placements.some((p) => p.engine === control)}
              coldGgufs={coldGgufs}
              onRefresh={onRefresh}
            />
          ) : null,
        )}
    </Panel>
  );
}
