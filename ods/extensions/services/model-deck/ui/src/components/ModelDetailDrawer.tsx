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
  tagsSettled,
  tagsWith,
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
 * (model/nodes.ts:findPlacement) so the pill tracks reality; the free-text
 * fields that ARE typed into are seeded once PER FACTS KEY rather than on
 * every tick (see `seededKey` below), so a refetch cannot land in the middle
 * of a sentence — while a swap that changes WHICH model this drawer is
 * pointed at does re-seed, instead of holding the previous model's text over
 * the new one's facts.
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
  // The tag list the last successful PUT actually sent, held until the
  // server's own view catches up (tagsSettled). Edits build on THIS, or the
  // refresh window between a write and its refetch would let a second edit
  // ship the pre-write array — see tagsSettled's docstring.
  const [writtenTags, setWrittenTags] = useState<string[] | null>(null);

  // The facts key whose values the three free-text drafts were seeded from —
  // NOT a bare "have we seeded yet" flag. Facts refetch on every poll tick
  // (below) and re-seeding on each of those would delete whatever the
  // operator was typing three characters in; but the KEY itself can change
  // under a stable component identity, because App keys this component on the
  // placement ID and a spark swap changes the model NAME while `sparky/slot0`
  // persists. Seeded once per key, the drawer would then hold the previous
  // model's label/notes over the new model's facts and write them onto it at
  // the next blur — cross-model corruption from a component that never
  // remounted.
  const seededKey = useRef<string | null>(null);
  const tagInputRef = useRef<HTMLInputElement>(null);

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
    if (facts === null || seededKey.current === factsKey) return;
    seededKey.current = factsKey;
    const seed = facts[factsKey] ?? {};
    setLabelDraft(declaredText(seed.label?.value));
    setNotesDraft(declaredText(seed.notes?.value));
    setEnginePrefDraft(declaredText(seed.engine_preference?.value));
    // Both belong to the key that was just left behind: an unsent tag and an
    // optimistic list written against the PREVIOUS model must not carry over
    // onto the new one.
    setTagDraft("");
    setWrittenTags(null);
  }, [facts, factsKey]);

  // Hands the tag list back to the server the moment its own view reflects
  // what was written. Functional update so this cannot race the seeding
  // effect above, which resets the same state on a key change.
  useEffect(() => {
    if (facts === null) return;
    const serverTags = facts[factsKey]?.tags?.value;
    setWrittenTags((cur) => (tagsSettled(cur, serverTags) ? null : cur));
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

  // What this screen last wrote wins over what the last refetch happened to
  // carry, until the two agree (tagsSettled).
  const tags = writtenTags ?? declaredTags(entry?.tags?.value);
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
  async function saveDeclared(fields: Record<string, unknown>): Promise<boolean> {
    setDeclaredBusy(true);
    try {
      await putDeclared(factsKey, fields);
      setDeclaredError(null);
      return true;
    } catch (err) {
      setDeclaredError(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setDeclaredBusy(false);
      onRefresh();
    }
  }

  /** Tag writes go through here rather than calling saveDeclared directly:
   * the array that was actually sent has to be remembered on success, or the
   * next edit composes against whatever the last refetch delivered — which,
   * for the whole width of an App refresh, is still the pre-write list. The
   * unlock does NOT wait for that refetch (it is three fetches away, one of
   * them to a remote node, and freezing the form for that long to protect one
   * field is the wrong trade); remembering what was sent closes the same
   * window without the freeze. */
  async function saveTags(next: string[]) {
    const key = factsKey;
    const ok = await saveDeclared({ tags: next });
    // `seededKey` is "which model this form is currently showing", which is
    // exactly the question here: a swap that completed while the PUT was in
    // flight has already re-pointed the drawer, and remembering this list
    // would then hold the PREVIOUS model's tags over the new one — the
    // cross-model carry-over the seeding effect exists to prevent. The write
    // still landed on `key`, correctly; only the optimistic display is
    // dropped.
    if (ok && seededKey.current === key) setWrittenTags(next);
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
    const next = tagsWith(tags, tagDraft);
    if (next === null) {
      // A duplicate (or a blank) is not an edit, so the box KEEPS what was
      // typed rather than blanking silently — the previous version cleared
      // first and left an operator watching their word disappear with no
      // chip to show for it. Selecting it points at the reason without a
      // banner for something that has not failed.
      tagInputRef.current?.select();
      return;
    }
    setTagDraft("");
    saveTags(next);
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
                      onClick={() => saveTags(tags.filter((t) => t !== tag))}
                    >
                      ✕
                    </button>
                  </span>
                ))}
                <input
                  ref={tagInputRef}
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
                    // Refuses while the facts are outstanding — the whole
                    // first tick, and permanently when /api/facts is failing,
                    // since the error path deliberately leaves `facts` as it
                    // found it. settingsIdentityFor's fallback is the
                    // placement name, which for a spark profile is the
                    // UNTRANSLATED "heretic": Settings would then open on
                    // `sparky/vllm|heretic`, a scope key nothing resolves,
                    // with nothing on screen saying so. That is the D11
                    // defect the helper exists to prevent, and a fallback is
                    // the wrong answer to "we do not know yet".
                    disabled={facts === null}
                    title={
                      facts === null ? labels.modelSettingsNoFactsTitle : labels.modelSettingsTitle
                    }
                    onClick={() =>
                      onOpenSettings({
                        node: placedOn.nodeId,
                        engine: settingsEngine,
                        // The board names a spark placement by its PROFILE;
                        // settings live under the checkpoint identity. See
                        // settingsIdentityFor — a local placement's name is
                        // already the model and passes through unchanged.
                        // `?? {}` is unreachable while the button is enabled.
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
