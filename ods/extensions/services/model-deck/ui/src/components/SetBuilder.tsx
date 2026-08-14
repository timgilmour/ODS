import { useEffect, useState, type DragEvent } from "react";
import {
  ApiError,
  bytesToGB,
  deleteSet,
  getSets,
  saveSet,
  slugify,
  truncateMiddle,
  type ConfigSet,
  type Durable,
  type Gpu,
  type ModelFile,
  type ResourceDesired,
  type TenantPolicy,
  type World,
} from "../api";
import { labels, messages } from "../model/messages";
import {
  buildDraft,
  derivePlacedModel,
  draftChoiceFor,
  draftedFootprintBytes,
  draftEquals,
  EXTRA_PREFIX,
  fieldsFromSet,
  KIND_DRAFT_SPEC,
  setResourceChoice,
} from "../model/setDraft";
import Banner from "../ui/Banner";
import Meter from "../ui/Meter";
import Panel from "../ui/Panel";
import ApplyModal from "./ApplyModal";
import ModelLibrary from "./ModelLibrary";
import SavedSets from "./SavedSets";

interface SetBuilderProps {
  models: ModelFile[];
  gpus: Gpu[];
  world: World;
  /** Display name of the box serving this UI (App passes
   * `state.node.label`, from app/routers/status.py's `node` block) — the
   * draft card is a draft OF THAT NODE, so it is titled with the same name
   * the board's local node card carries rather than a generic "draft". */
  nodeLabel: string;
  onModalOpenChange: (open: boolean) => void;
}

// Backend's registry constant (app/registry.py:HIPFIRE_FOOTPRINT) — hipfire
// has no live per-model footprint reading like lemonade does, so its
// budgeted cost is this fixed figure whenever a hipfire-kind resource is
// drafted "loaded" (resume).
const HIPFIRE_FOOTPRINT_BYTES = 33_000_000_000;

// GPU real-VRAM fallback when a GPU index isn't present in the live world
// snapshot (e.g. previewing a set on a box with fewer cards than the set
// assumes) — matches GpuColumn's un-fallback-guarded read, made explicit
// here since the builder must always render every declared resource's card
// regardless.
const DEFAULT_GPU_BYTES = 32_000_000_000;

// Above this share of a card's VRAM the drafted footprint is called over
// budget. A threshold, not a colour: Meter's own amber/red thresholds say
// how full the bar looks, this says whether the operator gets a banner.
const OVER_BUDGET_PCT = 90;

// Button copy for the generic (non-model) desired-choice row — today the
// only non-model kind offering "loaded" is hipfire-kind (resume), so that
// entry reads "Running" rather than the generic verb; "unloaded" is
// unreachable through this row (a load-verb kind takes the model-drop path
// below instead) but listed for completeness/exhaustiveness.
const DESIRED_BUTTON_LABEL: Record<ResourceDesired["desired"], string> = {
  loaded: "Running",
  unloaded: "Unload",
  parked: "Parked",
  freed: "Free",
};

// Snapshot taken the instant a 409 fires on save: the exact draft + the
// slug it collided with. Frozen at that moment rather than re-derived from
// live state later, so Overwrite can never act on a draft edited (or a
// different name typed) after the confirm banner appeared — see CRITICAL 1
// in the review that produced this fix.
interface OverwriteSnapshot {
  draft: ConfigSet;
  slug: string;
}

/** Drag-and-drop set editor: one dashed DRAFT card standing for the local
 * node, holding one control block per DECLARED resource (E1 Task 11:
 * replaces the fixed hipfire/lemonade/comfyui two-GPU layout — any number
 * of any-kind declared resources now, each its own card, mirroring
 * `nodes.ts`'s board redesign). A load-verb kind's resource (today:
 * lemonade alone, `KIND_DRAFT_SPEC`) gets the model-library drop target;
 * every other kind gets a plain choice row built from its own
 * `KIND_DRAFT_SPEC.options`. This is a *draft* of desired state, not a live
 * status card, which is what the dashed border and the DRAFT pill say out
 * loud.
 *
 * Scope: `ConfigSet` (see api.ts `Ephemeral`) has one leg per DECLARED
 * resource, keyed by name — there is no spark leg — so the builder drafts
 * the local node and nothing else. A second node card here would be a
 * control for a set field that does not exist. */
export default function SetBuilder({
  models,
  gpus,
  world,
  nodeLabel,
  onModalOpenChange,
}: SetBuilderProps) {
  // Existing sets, listed by SavedSets (load/duplicate/delete per row).
  const [sets, setSets] = useState<ConfigSet[]>([]);
  const [listError, setListError] = useState<string | null>(null);

  // Draft fields.
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [durable, setDurable] = useState<Durable | null>(null);
  const [resources, setResources] = useState<Record<string, ResourceDesired>>({});
  const [placedModel, setPlacedModel] = useState<string | null>(null);
  const [catalogId, setCatalogId] = useState("");
  // policy_overrides is not editable in v1, but must survive load->save
  // verbatim (a loaded set that carries overrides keeps them on the next
  // save). New drafts keep it null; a badge surfaces its presence.
  const [policyOverrides, setPolicyOverrides] = useState<Record<string, TenantPolicy> | null>(null);

  // Save flow.
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [overwriteSnapshot, setOverwriteSnapshot] = useState<OverwriteSnapshot | null>(null);
  const [savedSlug, setSavedSlug] = useState<string | null>(null);
  const [savedSet, setSavedSet] = useState<ConfigSet | null>(null);

  // Preview (reuses ApplyModal — only reachable once the current draft is
  // saved and unchanged, since preview/apply both act on a slug on disk).
  const [previewOpen, setPreviewOpen] = useState(false);

  function refreshSets() {
    getSets()
      .then((r) => {
        setSets(r.sets);
        setListError(null);
      })
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(refreshSets, []);

  function resetDraft() {
    setName("");
    setNotes("");
    setDurable(null);
    setResources({});
    setPlacedModel(null);
    setCatalogId("");
    setPolicyOverrides(null);
    setSaveError(null);
    setOverwriteSnapshot(null);
  }

  function populateFromSet(cfgset: ConfigSet, clearName: boolean) {
    const f = fieldsFromSet(cfgset, clearName);
    setName(f.name);
    setNotes(f.notes);
    setDurable(f.durable);
    setResources(f.resources);
    setPolicyOverrides(f.policyOverrides);
    setCatalogId(cfgset.durable?.activate_model_id ?? "");
    setPlacedModel(derivePlacedModel(cfgset));
    setSaveError(null);
    setOverwriteSnapshot(null);
  }

  function handleLoad(slug: string) {
    const cfgset = sets.find((s) => slugify(s.name) === slug);
    if (!cfgset) return;
    populateFromSet(cfgset, false);
    setSavedSlug(slug);
    setSavedSet(cfgset);
  }

  function handleDuplicate(slug: string) {
    const cfgset = sets.find((s) => slugify(s.name) === slug);
    if (!cfgset) return;
    populateFromSet(cfgset, true);
    setSavedSlug(null);
    setSavedSet(null);
  }

  const draft = buildDraft({ name, notes, durable, resources, policyOverrides });
  const isSavedUnchanged = savedSet !== null && draftEquals(draft, savedSet);

  // overwrite=true always saves the frozen snapshot from the moment the 409
  // fired, never the live `draft` — the form is disabled while a snapshot
  // is pending (see the fieldset below), but this keeps the save call
  // itself provably safe even if that ever changes. See CRITICAL 1.
  async function handleSave(overwrite: boolean) {
    // Re-entry guard: the confirm banner's Overwrite button lives in a
    // Banner (which takes no `disabled`), so the "no second save while one
    // is in flight" rule the old inline banner got from `disabled={saving}`
    // is enforced here instead.
    if (saving) return;
    const target = overwrite ? overwriteSnapshot?.draft : draft;
    if (!target || !target.name) return;
    setSaving(true);
    setSaveError(null);
    try {
      const { slug } = await saveSet(target, overwrite);
      setSavedSlug(slug);
      setSavedSet(target);
      setOverwriteSnapshot(null);
      refreshSets();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setOverwriteSnapshot({ draft, slug: slugify(draft.name) });
      } else {
        setSaveError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setSaving(false);
    }
  }

  function cancelOverwrite() {
    setOverwriteSnapshot(null);
  }

  // Delete is armed per row inside SavedSets (two-click, no window.confirm);
  // by the time this runs the operator has already confirmed. Only deleting
  // the set the draft was loaded FROM touches the draft — deleting some
  // other row must not throw away work in progress.
  async function handleDelete(slug: string) {
    try {
      await deleteSet(slug);
      if (slug === savedSlug) {
        setSavedSlug(null);
        setSavedSet(null);
        resetDraft();
      }
      refreshSets();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    }
  }

  // --- Model placement (drag or click-to-place; both funnel through here) -

  // The one declared resource whose kind carries a model (E1: any number of
  // declared resources now — `KIND_DRAFT_SPEC.supportsModel` is true only
  // for a load-verb kind, lemonade today). The model-library drop
  // target/picker is keyed to it; with none declared, there is nothing to
  // place a model onto and the library panel does not render (below).
  const loadResource = Object.entries(world.tenants).find(
    ([, tenant]) => KIND_DRAFT_SPEC[tenant.engine]?.supportsModel,
  )?.[0] ?? null;

  function placeModel(file: string) {
    if (!loadResource) return;
    setPlacedModel(file);
    setResources((r) => setResourceChoice(r, loadResource, "loaded"));
    setDurable({
      default_route_model: `${EXTRA_PREFIX}${file}`,
      activate_model_id: catalogId.trim() || null,
    });
  }

  function updateCatalogId(value: string) {
    setCatalogId(value);
    if (placedModel) {
      setDurable({
        default_route_model: `${EXTRA_PREFIX}${placedModel}`,
        activate_model_id: value.trim() || null,
      });
    }
  }

  function removeModel() {
    if (!loadResource) return;
    setPlacedModel(null);
    setResources((r) => setResourceChoice(r, loadResource, "unloaded"));
    setDurable(null);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    // The drop target isn't a form control, so the disabled <fieldset>
    // around the rest of the form doesn't block it natively — guard
    // explicitly so a drag can't mutate the draft while an overwrite
    // confirmation is pending (CRITICAL 1).
    if (overwriteSnapshot || !loadResource) return;
    const file = e.dataTransfer.getData("text/plain");
    // Malformed/unknown drops (empty payload, or a file not in the current
    // registry scan) are ignored silently rather than surfaced as an error
    // — a stray drag from elsewhere on the page shouldn't corrupt the draft.
    if (!file || !models.some((m) => m.file === file)) return;
    placeModel(file);
  }

  // --- Footprint budgets ----------------------------------------------------

  const placedFootprint = placedModel
    ? (models.find((m) => m.file === placedModel)?.footprint ?? 0)
    : 0;

  // One card per declared resource, ordered the same way the board is
  // (nodes.ts: gpu_index then resource name) so the draft and the live
  // board read the same left-to-right.
  const resourceEntries = Object.entries(world.tenants).sort(
    ([nameA, a], [nameB, b]) => a.gpu_index - b.gpu_index || nameA.localeCompare(nameB),
  );

  return (
    <>
      {/* Permanent, and outside the fieldset: what this whole screen is.
          Nothing below it deploys anything. */}
      <Banner message={messages.draftNothingDeployed()} />

      {/* Rendered OUTSIDE the disabled fieldset below so Overwrite/Cancel
          stay clickable while every draft-mutating control is locked — the
          target name is read from the frozen snapshot, never the
          (now-disabled, but belt-and-suspenders-safe) live draft. Banner's
          dismiss × is the cancel path. */}
      {overwriteSnapshot && (
        <Banner
          message={messages.overwriteSet(overwriteSnapshot.slug)}
          onAction={() => handleSave(true)}
          onDismiss={cancelOverwrite}
        />
      )}

      <div className="builder-layout">
        {/* Disabled as a whole (native fieldset cascade covers every
            input/textarea/select/button inside, across component
            boundaries: ModelLibrary's search and Place buttons, SavedSets'
            per-row Load/Duplicate/Delete, and every control in the draft
            card) whenever an overwrite confirmation is pending — see
            CRITICAL 1. The drop target isn't a form control, so handleDrop
            also short-circuits explicitly while a snapshot is pending. */}
        <fieldset className="builder-fieldset" disabled={overwriteSnapshot !== null}>
          <div className="builder-side">
            {loadResource && (
              <ModelLibrary
                models={models}
                onPlace={placeModel}
                targetGpu={world.tenants[loadResource].gpu_index}
              />
            )}
            <SavedSets
              sets={sets}
              listError={listError}
              onLoad={handleLoad}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
            />
          </div>

          <div className="builder-main">
            <Panel className="builder-meta">
              <label className="builder-field">
                name
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Chat mode"
                />
              </label>

              <label className="builder-field">
                notes
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                  placeholder="optional notes shown in the preview/apply modal"
                />
              </label>

              {policyOverrides !== null && (
                <div className="badge-policy-overrides" title="this set carries per-tenant policy overrides (not editable here in v1; preserved on save)">
                  policy overrides present
                </div>
              )}

              {saveError && (
                <Banner
                  message={messages.guardRefused(saveError)}
                  onDismiss={() => setSaveError(null)}
                />
              )}

              <div className="builder-actions">
                <button
                  type="button"
                  className="primary"
                  onClick={() => handleSave(false)}
                  disabled={saving || !draft.name}
                >
                  {saving ? labels.saving : labels.saveDraft}
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setPreviewOpen(true);
                    onModalOpenChange(true);
                  }}
                  disabled={!savedSlug || !isSavedUnchanged}
                  title={
                    !savedSlug
                      ? "save the set first"
                      : !isSavedUnchanged
                        ? "unsaved changes — save first"
                        : undefined
                  }
                >
                  {labels.previewSteps}
                </button>

                <button type="button" onClick={resetDraft}>
                  {labels.cancel}
                </button>
              </div>
            </Panel>

            <Panel
              draft
              className="builder-node"
              title={nodeLabel}
              actions={<span className="ui-pill ui-pill-busy">{labels.draftPill}</span>}
            >
              {resourceEntries.length === 0 ? (
                <p className="helper-text">nothing declared on this node yet</p>
              ) : (
                <div className="builder-gpu-row">
                  {resourceEntries.map(([resource, tenant]) => {
                    const spec = KIND_DRAFT_SPEC[tenant.engine];
                    const gpuTotal =
                      gpus.find((g) => g.index === tenant.gpu_index)?.total ?? DEFAULT_GPU_BYTES;
                    const choice = draftChoiceFor(resources, resource);
                    const bytes = draftedFootprintBytes(
                      tenant, choice, placedFootprint, HIPFIRE_FOOTPRINT_BYTES,
                    );
                    const pct = gpuTotal > 0 ? (bytes / gpuTotal) * 100 : 0;
                    const isLoadResource = resource === loadResource;

                    return (
                      <div
                        key={resource}
                        className={isLoadResource ? "builder-gpu set-builder-drop" : "builder-gpu"}
                        {...(isLoadResource
                          ? { onDragOver: (e: DragEvent<HTMLDivElement>) => e.preventDefault(), onDrop: handleDrop }
                          : {})}
                      >
                        <h3>{resource} · GPU {tenant.gpu_index}</h3>
                        {pct > OVER_BUDGET_PCT && <Banner message={messages.overBudget()} />}
                        {/* Subdued only at "don't touch": the bar is then
                            reporting the LIVE footprint, not a drafted one,
                            and the caption underneath says so in words. */}
                        <div className={choice === "none" ? "builder-meter-subdued" : undefined}>
                          <Meter capacity={{ used: bytes, total: gpuTotal }} />
                        </div>
                        {choice === "none" && <p className="helper-text">current live state</p>}

                        <div className="tenant-card">
                          <div className="tenant-card-head">
                            <span className="tenant-name">{resource}</span>
                          </div>

                          {isLoadResource ? (
                            placedModel ? (
                              <>
                                <div className="tenant-meta">
                                  <span title={placedModel}>{truncateMiddle(placedModel)}</span>
                                  <span>{bytesToGB(placedFootprint)} GB</span>
                                </div>
                                <p className="helper-text">Placed model becomes the default chat route.</p>
                                <input
                                  type="text"
                                  className="builder-catalog-id"
                                  placeholder="catalog id (optional, enables durable revert)"
                                  value={catalogId}
                                  onChange={(e) => updateCatalogId(e.target.value)}
                                />
                                <div className="tenant-actions">
                                  <button type="button" onClick={removeModel}>
                                    remove model
                                  </button>
                                </div>
                              </>
                            ) : durable ? (
                              // durable persists even though no placedModel
                              // could be derived from it (e.g.
                              // default_route_model without the "extra."
                              // prefix) — surfaced explicitly so it can
                              // never silently ride along on save.
                              <div className="durable-chip">
                                <span>durable route: {durable.default_route_model}</span>
                                <button
                                  type="button"
                                  onClick={() => setDurable(null)}
                                  aria-label="clear durable route"
                                >
                                  ✕
                                </button>
                              </div>
                            ) : (
                              <>
                                <div className="builder-dropzone">{labels.dropToAssign}</div>
                                {choice === "unloaded" && (
                                  <p className="helper-text">currently set to unload</p>
                                )}
                              </>
                            )
                          ) : (
                            <div className="tenant-actions">
                              <button
                                type="button"
                                className={choice === "none" ? "primary" : undefined}
                                onClick={() => setResources((r) => setResourceChoice(r, resource, "none"))}
                              >
                                don't touch
                              </button>
                              {spec?.options.map((opt) => (
                                <button
                                  key={opt}
                                  type="button"
                                  className={choice === opt ? "primary" : undefined}
                                  onClick={() => setResources((r) => setResourceChoice(r, resource, opt))}
                                >
                                  {DESIRED_BUTTON_LABEL[opt]}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </Panel>
          </div>
        </fieldset>
      </div>

      {previewOpen && savedSlug && savedSet && (
        <ApplyModal
          slug={savedSlug}
          cfgset={savedSet}
          onClose={() => {
            setPreviewOpen(false);
            onModalOpenChange(false);
          }}
        />
      )}
    </>
  );
}
