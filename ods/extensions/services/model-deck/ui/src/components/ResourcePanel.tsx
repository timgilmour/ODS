import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { labels, messages } from "../model/messages";
import { isTenantName, SPARK_CONTROL, type DeckResource, type Placement } from "../model/nodes";
import Meter from "../ui/Meter";
import ModelChip from "../ui/ModelChip";
import Panel from "../ui/Panel";
import PlacementActions from "./PlacementActions";
import SparkSwap from "./SparkSwap";

/** The facts that do not fit on the chip: which engine (when it is not the
 * node's usual one), whether a turn is in flight, queue depth, idle time,
 * and the two policy fields that decide what gets evicted first.
 *
 * Every value is read straight off the placement — the adapter decides what
 * is knowable and what is merely zero, so an absent field renders nothing at
 * all rather than a misleading "0". */
function PlacementMeta({ placement }: { placement: Placement }) {
  const { pinned, priority, busy, queue, idleSeconds, engineBadge } = placement;

  const empty =
    !engineBadge &&
    !busy &&
    queue === undefined &&
    idleSeconds == null &&
    !pinned &&
    priority == null;
  if (empty) return null;

  return (
    <div className="tenant-meta">
      {engineBadge && <span className="ui-pill ui-pill-off">{engineBadge}</span>}
      {busy && (
        <span className="ui-pill ui-pill-busy" title={labels.inUseTitle}>
          {labels.inUse}
        </span>
      )}
      {queue !== undefined && <span title={labels.queueTitle}>{labels.queue(queue)}</span>}
      {idleSeconds != null && <span title={labels.idleTitle}>{labels.idle(idleSeconds)}</span>}
      {pinned && (
        <span className="badge-pinned" title={labels.pinnedTitle}>
          {labels.pinned}
        </span>
      )}
      {priority != null && (
        <span title={labels.priorityTitle}>{labels.priority(priority)}</span>
      )}
    </div>
  );
}

export default function ResourcePanel({
  resource,
  world,
  models,
  coldGgufs,
  spark,
  stale,
  staleAge,
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
  /** Humanized age of the owning node's last successful contact, or null
   * when unknown. Rendered as a caption under each chip: the state pill
   * already says "last known", this answers the next question, which is
   * *how* stale — a different fact, not a redundant one. */
  staleAge: string | null;
  onRefresh: () => void;
}) {
  return (
    <Panel title={resource.label} className="resource-panel">
      <Meter capacity={resource.capacity} />

      {resource.placements.length === 0 ? (
        <div className="dropzone-empty">{labels.nothingPlaced}</div>
      ) : (
        resource.placements.map((p) => (
          <div key={p.id} className="resource-placement">
            <ModelChip placement={p} />
            <PlacementMeta placement={p} />
            {stale && staleAge && (
              <span className="ui-chip-stale-note">
                {messages.lastSeen(staleAge).title}
              </span>
            )}
          </div>
        ))
      )}

      {!stale &&
        resource.controls.map((control) =>
          control === SPARK_CONTROL ? (
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
