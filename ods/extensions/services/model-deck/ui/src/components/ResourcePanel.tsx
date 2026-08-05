import type { ModelFile, SparkStatus, StorageUnit, TenantName, World } from "../api";
import { messages } from "../model/messages";
import type { DeckResource } from "../model/nodes";
import Meter from "../ui/Meter";
import ModelChip from "../ui/ModelChip";
import Panel from "../ui/Panel";
import PlacementActions from "./PlacementActions";
import SparkSwap from "./SparkSwap";

const TENANTS: readonly string[] = ["lemonade", "comfyui", "hipfire"];

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
          ) : TENANTS.includes(control) ? (
            <PlacementActions
              key={control}
              tenant={control as TenantName}
              world={world}
              models={models}
              coldGgufs={coldGgufs}
              onRefresh={onRefresh}
            />
          ) : null,
        )}
    </Panel>
  );
}
