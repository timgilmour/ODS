import { useEffect, useRef, useState } from "react";
import {
  bytesToGB,
  getFacts,
  getFactsDrift,
  putDeclared,
  sparkReload,
  type FactEntry,
  type FactsDriftItem,
  type FactsMap,
  type Layer,
  type ModelFile,
  type StorageUnit,
  type World,
} from "../api";
import {
  declaredTags,
  declaredText,
  factRows,
  factValueText,
  partitionDrift,
} from "../model/factsView";
import { labels, messages } from "../model/messages";
import { isTenantName, SPARK_SLOT_KEY, type DeckResource, type Placement } from "../model/nodes";
import { settingsIdentityFor } from "../model/settingsView";
import Banner from "../ui/Banner";
import ProvenanceDot from "../ui/ProvenanceDot";
import StatePill from "../ui/StatePill";
import PlacementActions from "./PlacementActions";
import type { SettingsTarget } from "./SettingsModal";

/** Which provenance dot a fact ORIGIN borrows. DISPLAY-ONLY REUSE, not layer
 * semantics: a fact has two origins (app/facts.py:resolve_facts sets exactly
 * "derived"/"declared"), the settings ladder has five layers
 * (app/ladder.py:48), and nothing maps one onto the other. What is shared is
 * the palette's meaning — dim grey for "the machine read this", the brightest
 * accent for "a human declared it" — which is the same distinction facts
 * carry. The dot's NAME is overridden at every call site (labels.factOrigin),
 * because the layer sentence would be false here. */
const FACT_DOT_LAYER: Record<FactEntry["origin"], Layer> = {
  derived: "engine_defaults",
  declared: "engine_model",
};

/** Where a placement is placed — the two things a verb needs that the
 * placement itself does not carry. `null` means App could not re-derive the
 * placement from the latest poll (it left the board), which is exactly when
 * there is nothing left to act on. */
export interface PlacedOn {
  nodeId: string;
  resource: DeckResource;
}

/**
 * Everything about one placement that does not fit on its chip: the facts the
 * deck derived about the model, the five fields a human may declare about it
 * (app/declared.py:32-38), and the verbs that act on it.
 *
 * Polling is deliberately NOT paused while this is open — unlike every modal
 * in the deck, which pauses because it holds uncommitted edits over a surface
 * that would re-render underneath. This drawer's headline IS a live status
 * pill, and a paused poll would freeze it at whatever it read when the chip
 * was clicked. App re-derives `placement` from each fresh `buildNodes` by id
 * (model/nodes.ts:findPlacement) so the pill tracks reality; the two text
 * fields that ARE typed into are seeded once and never re-seeded (see
 * `seeded` below), so a refetch cannot land in the middle of a sentence.
 */
export default function ModelDetailDrawer({
  placement,
  placedOn,
  world,
  models,
  coldGgufs,
  refreshTrigger,
  engineSettingsNodes,
  suspended,
  onOpenSettings,
  onRefresh,
  onClose,
}: {
  /** Re-derived from the freshest poll by App; falls back to the last-known
   * object when the placement has left the board. */
  placement: Placement;
  placedOn: PlacedOn | null;
  world: World;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  /** Bumped by App on every poll and after every mutating action; refetches
   * the facts, same contract EventsView consumes it under. */
  refreshTrigger: number;
  /** nodeId -> engine for the nodes whose engine is configurable — App's
   * catalog probe. Gating the Settings button on THIS rather than on a node
   * name is the point: a node earns the button by having a harvested option
   * catalog, which is the only configurability signal the backend has. */
  engineSettingsNodes: Record<string, string>;
  /** True when a modal is stacked above this drawer. The drawer keeps
   * rendering (it is the context that modal was opened from) but stops
   * answering Escape, which belongs to the topmost surface. */
  suspended: boolean;
  onOpenSettings: (target: SettingsTarget) => void;
  onRefresh: () => void;
  onClose: () => void;
}) {
  // Facts are keyed `<kind>/<id>` (app/routers/facts.py:64-71 unions the
  // characteristics and declared stores, both keyed that way).
  const factsKey = `model/${placement.name}`;

  const [facts, setFacts] = useState<FactsMap | null>(null);
  const [drift, setDrift] = useState<Record<string, FactsDriftItem[]>>({});
  const [factsError, setFactsError] = useState<string | null>(null);
  const [declaredBusy, setDeclaredBusy] = useState(false);
  const [declaredError, setDeclaredError] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);
  const [reloadError, setReloadError] = useState<string | null>(null);
  const [tagDraft, setTagDraft] = useState("");
  const [labelDraft, setLabelDraft] = useState("");
  const [notesDraft, setNotesDraft] = useState("");
  const [enginePrefDraft, setEnginePrefDraft] = useState("");

  // The three free-text fields are seeded from the FIRST facts response and
  // never again. Facts refetch on every poll tick (below), and re-seeding on
  // each of those would delete whatever the operator was typing three
  // characters in. App keys this component on the placement id, so clicking a
  // different chip remounts it and seeds afresh.
  const seeded = useRef(false);

  useEffect(() => {
    let alive = true;
    Promise.all([getFacts(), getFactsDrift()]).then(
      ([f, d]) => {
        if (!alive) return;
        setFacts(f);
        setDrift(d);
        setFactsError(null);
      },
      (err) => {
        // Deliberately keeps whatever facts are already on screen: a failed
        // refetch makes them stale, not wrong, and blanking the table would
        // read as "this model has no facts".
        if (alive) setFactsError(err instanceof Error ? err.message : String(err));
      },
    );
    return () => {
      alive = false;
    };
  }, [refreshTrigger]);

  useEffect(() => {
    if (seeded.current || facts === null) return;
    seeded.current = true;
    const entry = facts[factsKey] ?? {};
    setLabelDraft(declaredText(entry.label?.value));
    setNotesDraft(declaredText(entry.notes?.value));
    setEnginePrefDraft(declaredText(entry.engine_preference?.value));
  }, [facts, factsKey]);

  useEffect(() => {
    if (suspended) return;
    // Window-level rather than a handler on the panel: the drawer does not
    // trap focus (the board behind it stays live on purpose), so there is no
    // element guaranteed to be focused for a React key handler to fire on.
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, suspended]);

  const entry = facts?.[factsKey];
  const rows = factRows(entry);
  const { byName, unmatched } = partitionDrift(
    drift[factsKey],
    rows.map(([name]) => name),
  );

  const tags = declaredTags(entry?.tags?.value);
  const toolsVerified = entry?.tools_verified?.value === true;
  const gone = placedOn === null;

  // Every declared write REPLACES the whole field — `tags` most of all — so
  // none of this form may be touched before the current values are in hand.
  // Editing against a facts map that never arrived would silently delete
  // whatever is already declared. A failed fetch leaves it locked for the
  // same reason, with the banner above saying why.
  const declaredLocked = declaredBusy || facts === null;

  /** One declared field, one PUT. The route takes the fields to merge and
   * writes atomically (app/declared.py), so a per-field save is a complete
   * save — and `tags` is always sent as the FULL new array, because the merge
   * is per FIELD, not per list element. `onRefresh` runs either way: it bumps
   * refreshTrigger, which is what pulls the server's own view of what landed
   * back onto the screen. */
  async function saveDeclared(fields: Record<string, unknown>) {
    setDeclaredBusy(true);
    try {
      await putDeclared(factsKey, fields);
      setDeclaredError(null);
    } catch (err) {
      setDeclaredError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeclaredBusy(false);
      onRefresh();
    }
  }

  /** Saves a text field only when the draft actually differs from what the
   * server holds — a blur is not an edit, and a PUT per focus-and-leave would
   * write a fact nobody changed.
   *
   * Emptying a field declares it EMPTY rather than deleting it: the store
   * merges fields and its only removal verb drops a whole key
   * (app/declared.py's `put` vs `forget`), so "" is the honest thing a text
   * box can express here. */
  function commitText(field: "label" | "notes" | "engine_preference", draft: string) {
    if (draft === declaredText(entry?.[field]?.value)) return;
    saveDeclared({ [field]: draft });
  }

  function addTag() {
    const tag = tagDraft.trim();
    setTagDraft("");
    if (tag === "" || tags.includes(tag)) return;
    saveDeclared({ tags: [...tags, tag] });
  }

  async function handleReload() {
    setReloading(true);
    try {
      await sparkReload();
      setReloadError(null);
    } catch (err) {
      setReloadError(err instanceof Error ? err.message : String(err));
    } finally {
      setReloading(false);
      onRefresh();
    }
  }

  // A tenant control belonging to THIS placement. The resource's controls are
  // the authority on what is a local tenant (isTenantName narrows the
  // deliberately-`string[]` list), but a GPU can host two tenants and this
  // drawer is about one placement — the other tenant's verbs belong to its
  // own chip, not to this one.
  const tenantControl = placedOn
    ? (placedOn.resource.controls.filter(isTenantName).find((c) => c === placement.engine) ?? null)
    : null;
  // Spark is a single-slot node whose lifecycle key is fixed backend-side
  // (app/observe.py SPARK_SLOT_KEY, mirrored in nodes.ts).
  const isSparkSlot = placement.id === SPARK_SLOT_KEY;
  const settingsEngine = placedOn ? (engineSettingsNodes[placedOn.nodeId] ?? null) : null;
  // Verbs need a reachable node AND a placement still on the board: a stale
  // reading is a memory, and nothing can act on a memory.
  const showActions =
    placedOn !== null &&
    !placement.stale &&
    (tenantControl !== null || isSparkSlot || settingsEngine !== null);

  return (
    <>
      {/* Decorative: closing is also on the × and on Escape, so the overlay
          adds no keyboard-only path and needs no role. */}
      <div className="ui-drawer-overlay" aria-hidden="true" onClick={onClose} />
      <aside className="ui-drawer" role="dialog" aria-label={labels.modelDetail}>
        <div className="ui-drawer-head">
          <div className="drawer-hero">
            {/* Never truncated, unlike the chip: the checkpoint name IS the
                model's identity, and this is the surface that exists to show
                it in full. */}
            <div className="drawer-name">{placement.name}</div>
            <div className="drawer-hero-meta">
              <StatePill status={placement.status} stale={placement.stale || gone} />
              <span>{placement.engine}</span>
              {placement.bytes != null && <span>{bytesToGB(placement.bytes)} GB</span>}
            </div>
          </div>
          <button type="button" aria-label={labels.close} onClick={onClose}>
            ×
          </button>
        </div>

        <div className="ui-drawer-body">
          {gone && <Banner message={messages.placementGone()} />}
          {factsError && <Banner message={messages.factsLoadFailed(factsError)} />}

          <section className="drawer-section">
            <h4 className="drawer-section-head">
              {labels.factsSection}
              <span className="drawer-section-hint">{labels.factsSectionHint}</span>
            </h4>

            {facts === null && factsError === null && (
              <div className="drawer-empty">{labels.loading}</div>
            )}
            {facts !== null && rows.length === 0 && (
              // The tags/declared form below stays available regardless: a
              // declared put CREATES the entry (the route unions both stores),
              // so "no facts yet" is a starting point, not a dead end.
              <div className="drawer-empty">{labels.noFactsYet}</div>
            )}

            {rows.length > 0 && (
              <div className="fact-table">
                {rows.map(([name, fact]) => {
                  const item = byName.get(name);
                  return (
                    <div className="fact-row" key={name}>
                      <div className="fact-line">
                        <ProvenanceDot
                          layer={FACT_DOT_LAYER[fact.origin]}
                          label={labels.factOrigin(fact.origin, fact.source)}
                        />
                        <span className="fact-name">{name}</span>
                        <span className={item ? "fact-value fact-drifted" : "fact-value"}>
                          {factValueText(fact.value)}
                        </span>
                        {/* Present only when a declaration overrode something
                            the deck had derived (app/facts.py:resolve_facts
                            keeps the loser) — struck through, because it is
                            what the machine read and no longer what wins. */}
                        {fact.shadowed_value !== undefined && (
                          <span className="fact-shadowed" title={labels.shadowedTitle}>
                            {factValueText(fact.shadowed_value)}
                          </span>
                        )}
                      </div>
                      {item && (
                        <div className="fact-drift">
                          {labels.driftDetail(
                            factValueText(item.expected),
                            factValueText(item.actual),
                            item.severity,
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Drift the facts table has no row for — which, today, is ALL of
                it: every rule in app/facts.py's DRIFT_RULES reports a runtime
                field name that differs from the fact it was compared against
                (see partitionDrift). Rendered here rather than mapped onto a
                row by a second copy of that table. */}
            {unmatched.length > 0 && (
              <div className="fact-table">
                {unmatched.map((item, i) => (
                  <div className="fact-row" key={`${item.field}-${i}`}>
                    {/* The RUNTIME field's name, verbatim from the payload —
                        not dressed up as a fact row, because no fact by that
                        name exists. The values live in the line beneath. */}
                    <div className="fact-line">
                      <span className="fact-name fact-drifted">{labels.driftIn(item.field)}</span>
                    </div>
                    <div className="fact-drift">
                      {labels.driftDetail(
                        factValueText(item.expected),
                        factValueText(item.actual),
                        item.severity,
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="drawer-section">
            <h4 className="drawer-section-head">
              {labels.declaredFactsSection}
              <span className="drawer-section-hint">{labels.declaredFactsHint}</span>
            </h4>

            {declaredError && (
              <Banner
                message={messages.declaredSaveFailed(declaredError)}
                onDismiss={() => setDeclaredError(null)}
              />
            )}

            <div className="drawer-field">
              <span className="drawer-field-label">{labels.tagsField}</span>
              <div className="drawer-tags">
                {tags.map((tag) => (
                  <span className="drawer-tag" key={tag}>
                    {tag}
                    <button
                      type="button"
                      aria-label={labels.removeTagTitle(tag)}
                      title={labels.removeTagTitle(tag)}
                      disabled={declaredLocked}
                      onClick={() => saveDeclared({ tags: tags.filter((t) => t !== tag) })}
                    >
                      ✕
                    </button>
                  </span>
                ))}
                <input
                  className="drawer-input"
                  type="text"
                  aria-label={labels.addTag}
                  placeholder={labels.addTag}
                  value={tagDraft}
                  disabled={declaredLocked}
                  onChange={(e) => setTagDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key !== "Enter") return;
                    // Enter in a text input can submit an enclosing form;
                    // there is none today, and this keeps it that way.
                    e.preventDefault();
                    addTag();
                  }}
                />
              </div>
            </div>

            <label className="drawer-field">
              <span className="drawer-field-label">{labels.factLabel}</span>
              <input
                className="drawer-input"
                type="text"
                value={labelDraft}
                disabled={declaredLocked}
                onChange={(e) => setLabelDraft(e.target.value)}
                onBlur={() => commitText("label", labelDraft)}
              />
            </label>

            <label className="drawer-field">
              <span className="drawer-field-label">{labels.factNotes}</span>
              <textarea
                className="drawer-input"
                rows={3}
                value={notesDraft}
                disabled={declaredLocked}
                onChange={(e) => setNotesDraft(e.target.value)}
                onBlur={() => commitText("notes", notesDraft)}
              />
            </label>

            <label className="drawer-checkbox" title={labels.factToolsVerifiedTitle}>
              <input
                type="checkbox"
                checked={toolsVerified}
                disabled={declaredLocked}
                onChange={(e) => saveDeclared({ tools_verified: e.target.checked })}
              />
              <span>{labels.factToolsVerified}</span>
            </label>

            <label className="drawer-field">
              <span className="drawer-field-label" title={labels.factEnginePreferenceTitle}>
                {labels.factEnginePreference}
              </span>
              <input
                className="drawer-input"
                type="text"
                value={enginePrefDraft}
                disabled={declaredLocked}
                onChange={(e) => setEnginePrefDraft(e.target.value)}
                onBlur={() => commitText("engine_preference", enginePrefDraft)}
              />
            </label>
          </section>

          {showActions && placedOn && (
            <section className="drawer-section">
              <h4 className="drawer-section-head">{labels.actionsSection}</h4>

              {reloadError && (
                <Banner
                  message={messages.reloadFailed(reloadError)}
                  onDismiss={() => setReloadError(null)}
                />
              )}

              <div className="drawer-actions">
                {settingsEngine && (
                  <button
                    type="button"
                    title={labels.modelSettingsTitle}
                    onClick={() =>
                      onOpenSettings({
                        node: placedOn.nodeId,
                        engine: settingsEngine,
                        // The board names a spark placement by its PROFILE;
                        // settings live under the checkpoint identity. See
                        // settingsIdentityFor — a local placement's name is
                        // already the model and passes through unchanged.
                        model: settingsIdentityFor(
                          facts ?? {},
                          placedOn.nodeId,
                          settingsEngine,
                          placement.name,
                        ),
                      })
                    }
                  >
                    {labels.modelSettings}
                  </button>
                )}

                {isSparkSlot && (
                  <button
                    type="button"
                    disabled={reloading}
                    title={labels.reloadTitle}
                    onClick={handleReload}
                  >
                    {reloading ? labels.reloading : labels.reload}
                  </button>
                )}
              </div>

              {/* The same component the board mounts, with the same props:
                  the verbs, their guards, and their armed confirmations are
                  its business, and a second rendering of them here would be a
                  second set of rules to keep in step. */}
              {tenantControl && (
                <PlacementActions
                  tenant={tenantControl}
                  world={world}
                  models={models}
                  hasPlacement={placedOn.resource.placements.some(
                    (p) => p.engine === tenantControl,
                  )}
                  coldGgufs={coldGgufs}
                  onRefresh={onRefresh}
                />
              )}
            </section>
          )}
        </div>
      </aside>
    </>
  );
}
