import type { ModelFile, SparkStatus, StorageUnit, World } from "../api";
import { labels, messages } from "../model/messages";
import { controlHasPlacement, SPARK_CONTROL, type DeckResource, type Placement } from "../model/nodes";
import GpuStatsBlock from "../ui/GpuStatsBlock";
import Meter from "../ui/Meter";
import ModelChip from "../ui/ModelChip";
import Panel from "../ui/Panel";
import DriftCard from "./DriftCard";
import PlacementActions from "./PlacementActions";
import RemoteEngineActions from "./RemoteEngineActions";
import type { SettingsTarget } from "./SettingsModal";
import SparkSwap from "./SparkSwap";

/** The facts that do not fit on the chip: which engine is serving it, whether
 * a turn is in flight, queue depth, idle time, and the two policy fields
 * that decide what gets evicted first.
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
  nodeId,
  settingsEngine,
  onChipClick,
  onOpenSettings,
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
  /** The node this resource belongs to — threaded to the drift card, which
   * needs it (with `settingsEngine`) for the same Settings-target
   * translation ModelDetailDrawer performs. */
  nodeId: string;
  /** The node's configurable engine (App's catalog probe), or null when it
   * has none. Same gate NodeCard's Engine settings button and
   * ModelDetailDrawer's Settings button use. */
  settingsEngine: string | null;
  /** Opens the model detail drawer for a chip. Optional so a chip stays a
   * plain, unclickable div wherever no handler is threaded (ModelChip renders
   * a button only when it has an onClick) — the affordance never appears
   * without something behind it. A STALE chip stays clickable on purpose: the
   * drawer is a read surface, and last-known facts are exactly what an
   * operator wants while a node is dark. */
  onChipClick?: (placement: Placement) => void;
  onOpenSettings: (target: SettingsTarget) => void;
  onRefresh: () => void;
}) {
  return (
    <Panel
      title={
        <>
          {resource.label}
          {resource.stats?.name && (
            <span className="gpu-name">{resource.stats.name}</span>
          )}
          {resource.unmanaged && (
            <span className="unmanaged-tag">{labels.unmanagedGpu}</span>
          )}
        </>
      }
      className="resource-panel"
    >
      {resource.stats && <GpuStatsBlock stats={resource.stats} />}
      <Meter capacity={resource.capacity} />

      {resource.placements.length === 0 ? (
        <div className="dropzone-empty">{labels.nothingPlaced}</div>
      ) : (
        resource.placements.map((p) => (
          <div key={p.id} className="resource-placement">
            <ModelChip
              placement={p}
              onClick={onChipClick ? () => onChipClick(p) : undefined}
            />
            <PlacementMeta placement={p} />
            {stale && staleAge && (
              <span className="ui-chip-stale-note">
                {messages.lastSeen(staleAge).title}
              </span>
            )}
            {p.settingsDrift && (
              <DriftCard
                placement={p}
                drift={p.settingsDrift}
                nodeId={nodeId}
                settingsEngine={settingsEngine}
                stale={stale}
                onOpenSettings={onOpenSettings}
                onRefresh={onRefresh}
              />
            )}
          </div>
        ))
      )}

      {/* The DECLARED REMOTE engines whose chips ride this GPU card (Task
          10b) — several, since one GPU can carry more than one. Outside the
          `stale` gate below on purpose: an unreachable node's card keeps
          showing what it last knew AND each engine's own state word, while
          every verb arrives already disabled — nodes.ts folds staleness into
          the verb list itself (model/engineVerbs.ts), so the disabled-ness
          travels with the button rather than being re-decided here. */}
      {resource.remoteEngines?.map((control) => (
        <RemoteEngineActions key={control.resource} control={control} onRefresh={onRefresh} />
      ))}

      {!stale &&
        resource.controls.map((control) =>
          control === SPARK_CONTROL ? (
            spark && (
              <SparkSwap key={control} nodeId={nodeId} spark={spark} onChanged={onRefresh} />
            )
          ) : (
            // Any non-spark control is a local resource's own name
            // (nodes.ts's DeckResource.controls carries the resources
            // DECLARED on this GPU — never an unrecognized string), so no
            // narrowing guard is needed here anymore.
            <PlacementActions
              key={control}
              resource={control}
              world={world}
              models={models}
              // Whether THIS control already has a chip on this card — the
              // per-control question, and the only real "has a placement"
              // question the board has (ModelDetailDrawer's own call site
              // has already proved its answer and passes the literal; see
              // nodes.ts's controlHasPlacement). A shared GPU can carry two
              // controls and one chip: an unloaded resource's row must not
              // read as "has a chip" just because a co-resident neighbour's
              // does. A parked hipfire-kind resource deliberately has none,
              // and then the control row is the only thing that can say what
              // state it is in.
              hasPlacement={controlHasPlacement(resource, control)}
              coldGgufs={coldGgufs}
              onRefresh={onRefresh}
            />
          ),
        )}
    </Panel>
  );
}
