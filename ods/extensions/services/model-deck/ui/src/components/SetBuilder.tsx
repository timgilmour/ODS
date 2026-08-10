import { useEffect, useState, type DragEvent } from "react";
import {
  ApiError,
  bytesToGB,
  deleteSet,
  getSets,
  saveSet,
  slugify,
  truncateMiddle,
  type ComfyuiEphemeral,
  type ConfigSet,
  type Durable,
  type Gpu,
  type HipfireEphemeral,
  type LemonadeEphemeral,
  type ModelFile,
  type TenantPolicy,
  type World,
} from "../api";
import { labels, messages } from "../model/messages";
import {
  buildDraft,
  derivePlacedModel,
  draftEquals,
  EXTRA_PREFIX,
  fieldsFromSet,
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
// budgeted cost is this fixed figure whenever the draft's toggle is
// "running".
const HIPFIRE_FOOTPRINT_BYTES = 33_000_000_000;

// GPU real-VRAM fallback when a GPU index isn't present in the live world
// snapshot (e.g. previewing a set on a box with fewer cards than the set
// assumes) — matches GpuColumn's un-fallback-guarded read, made explicit
// here since the builder must always render both columns regardless.
const DEFAULT_GPU_BYTES = 32_000_000_000;

// Above this share of a card's VRAM the drafted footprint is called over
// budget. A threshold, not a colour: Meter's own amber/red thresholds say
// how full the bar looks, this says whether the operator gets a banner.
const OVER_BUDGET_PCT = 90;

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
 * node, holding the hipfire column (a toggle, not a drop target) and the
 * lemonade/comfyui column (a drop target). Both mirror GpuColumn's tenant
 * grouping but render bespoke controls per tenant rather than TenantCard's
 * live-status view — this is a *draft* of desired state, not a live status
 * card, which is what the dashed border and the DRAFT pill say out loud.
 * Column indices (which physical GPU each one is) come from the world
 * snapshot's placement map, not a hardcoded layout — GpuColumn no longer
 * hardcodes it either.
 *
 * Scope: `ConfigSet` (see api.ts `Ephemeral`) has legs for lemonade,
 * comfyui and hipfire only — there is no spark leg — so the builder drafts
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
  const [lemonade, setLemonade] = useState<LemonadeEphemeral | null>(null);
  const [comfyui, setComfyui] = useState<ComfyuiEphemeral | null>(null);
  // null = "don't touch" (matches lemonade/comfyui's pattern) — must never
  // be defaulted to a concrete state, see CRITICAL 2.
  const [hipfire, setHipfire] = useState<HipfireEphemeral | null>(null);
  const [placedModel, setPlacedModel] = useState<string | null>(null);
  const [catalogId, setCatalogId] = useState("");
  const [reserveGb, setReserveGb] = useState(24);
  // The RAW string is what the input renders. Committing only valid parses
  // means clearing the field no longer snaps a digit under the operator's
  // cursor: the old `Math.max(1, Math.round(value || 0))` turned an empty
  // field (Number("") === 0) into 1 mid-edit [max-review c35].
  const [reserveGbRaw, setReserveGbRaw] = useState("24");
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
    setLemonade(null);
    setComfyui(null);
    setHipfire(null);
    setPlacedModel(null);
    setCatalogId("");
    setReserveGb(24);
    setReserveGbRaw("24");   // both states, always together — see updateReserveGb
    setPolicyOverrides(null);
    setSaveError(null);
    setOverwriteSnapshot(null);
  }

  function populateFromSet(cfgset: ConfigSet, clearName: boolean) {
    const f = fieldsFromSet(cfgset, clearName);
    setName(f.name);
    setNotes(f.notes);
    setDurable(f.durable);
    setLemonade(f.lemonade);
    setComfyui(f.comfyui);
    setHipfire(f.hipfire);
    setPolicyOverrides(f.policyOverrides);
    setCatalogId(cfgset.durable?.activate_model_id ?? "");
    const loadedReserve = cfgset.ephemeral?.comfyui?.reserve_gb ?? 24;
    setReserveGb(loadedReserve);
    setReserveGbRaw(String(loadedReserve));   // both states, always together
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

  const draft = buildDraft({
    name, notes, durable, lemonade, comfyui, hipfire,
    policyOverrides,
  });
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

  // --- Placement (drag or click-to-place; both funnel through here) -------

  function placeModel(file: string) {
    setPlacedModel(file);
    setLemonade({ state: "loaded" });
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
    setPlacedModel(null);
    setLemonade({ state: "unloaded" });
    setDurable(null);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    // The lemonade/comfyui drop target isn't a form control, so the disabled
    // <fieldset> around the rest of the form doesn't block it natively —
    // guard explicitly so a drag can't mutate the draft while an overwrite
    // confirmation is pending (CRITICAL 1).
    if (overwriteSnapshot) return;
    const file = e.dataTransfer.getData("text/plain");
    // Malformed/unknown drops (empty payload, or a file not in the current
    // registry scan) are ignored silently rather than surfaced as an error
    // — a stray drag from elsewhere on the page shouldn't corrupt the draft.
    if (!file || !models.some((m) => m.file === file)) return;
    placeModel(file);
  }

  // --- Hipfire three-way (don't touch / running / parked) -----------------
  // Mirrors lemonade/comfyui's null-means-"don't touch" pattern exactly.

  function setHipfireChoice(choice: "none" | "running" | "parked") {
    if (choice === "none") {
      setHipfire(null);
      return;
    }
    setHipfire({ state: choice });
  }

  // --- ComfyUI three-way (don't touch / leave-reserve / free) -------------

  function setComfyChoice(choice: "none" | "leave" | "free") {
    if (choice === "none") {
      setComfyui(null);
      return;
    }
    setComfyui({ state: choice, reserve_gb: reserveGb });
  }

  // Whole GB only — fractional input causes an avoidable server 422.
  //
  // An unparseable or out-of-range edit updates the VISIBLE string but does
  // not touch the committed draft: the operator can clear the field, type,
  // and correct themselves without the value being rewritten under them, and
  // the draft keeps the last thing they actually meant. onBlur re-syncs an
  // abandoned edit so the field and the draft agree again.
  function updateReserveGb(raw: string) {
    setReserveGbRaw(raw);
    const trimmed = raw.trim();
    if (trimmed === "") return;              // mid-edit, not a value yet
    const n = Number(trimmed);
    if (!Number.isInteger(n) || n < 1) return;
    setReserveGb(n);
    if (comfyui) setComfyui({ ...comfyui, reserve_gb: n });
  }

  // --- Footprint budgets ----------------------------------------------------

  // Which physical GPU each column represents — derived from the world
  // snapshot's placement map (World.snapshot "placement"), not hardcoded;
  // GpuColumn derives the same values independently for its own layout.
  const hipfireGpu = world.placement.hipfire;
  const sharedGpu = world.placement.lemonade;

  const gpu0Total = gpus.find((g) => g.index === hipfireGpu)?.total ?? DEFAULT_GPU_BYTES;
  const gpu1Total = gpus.find((g) => g.index === sharedGpu)?.total ?? DEFAULT_GPU_BYTES;

  const placedFootprint = placedModel
    ? (models.find((m) => m.file === placedModel)?.footprint ?? 0)
    : 0;
  const comfyReserveBytes = comfyui?.state === "leave" ? comfyui.reserve_gb * 1e9 : 0;

  // null ("don't touch") shows the CURRENT live footprint (world state
  // already reports 0 when hipfire isn't running, see
  // app/state.py:HIPFIRE_FOOTPRINT-if-running-else-0) rather than a guessed
  // number — the meter gets a subdued style below so it reads as "this is
  // what's there now," not "this is what the draft will do."
  const gpu0Bytes =
    hipfire === null
      ? world.tenants.hipfire.footprint
      : hipfire.state === "running"
        ? HIPFIRE_FOOTPRINT_BYTES
        : 0;
  const gpu1Bytes = placedFootprint + comfyReserveBytes;

  const gpu0Pct = gpu0Total > 0 ? (gpu0Bytes / gpu0Total) * 100 : 0;
  const gpu1Pct = gpu1Total > 0 ? (gpu1Bytes / gpu1Total) * 100 : 0;

  const hipfireChoice: "none" | "running" | "parked" = hipfire === null ? "none" : hipfire.state;
  const comfyChoice: "none" | "leave" | "free" = comfyui === null ? "none" : comfyui.state;

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
            CRITICAL 1. The lemonade/comfyui drop target isn't a form
            control, so handleDrop also short-circuits explicitly while a
            snapshot is pending. */}
        <fieldset className="builder-fieldset" disabled={overwriteSnapshot !== null}>
          <div className="builder-side">
            <ModelLibrary models={models} onPlace={placeModel} targetGpu={sharedGpu} />
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
              <div className="builder-gpu-row">
                <div className="builder-gpu">
                  <h3>GPU {hipfireGpu}</h3>
                  {gpu0Pct > OVER_BUDGET_PCT && <Banner message={messages.overBudget()} />}
                  {/* Subdued only while hipfire is "don't touch": the bar is
                      then reporting the LIVE footprint, not a drafted one,
                      and the caption underneath says so in words. */}
                  <div className={hipfire === null ? "builder-meter-subdued" : undefined}>
                    <Meter capacity={{ used: gpu0Bytes, total: gpu0Total }} />
                  </div>
                  {hipfire === null && <p className="helper-text">current live state</p>}

                  <div className="tenant-card">
                    <div className="tenant-card-head">
                      <span className="tenant-name">hipfire</span>
                    </div>
                    <div className="tenant-actions">
                      <button
                        type="button"
                        className={hipfireChoice === "none" ? "primary" : undefined}
                        onClick={() => setHipfireChoice("none")}
                      >
                        don't touch
                      </button>
                      <button
                        type="button"
                        className={hipfireChoice === "running" ? "primary" : undefined}
                        onClick={() => setHipfireChoice("running")}
                      >
                        Running
                      </button>
                      <button
                        type="button"
                        className={hipfireChoice === "parked" ? "primary" : undefined}
                        onClick={() => setHipfireChoice("parked")}
                      >
                        Parked
                      </button>
                    </div>
                  </div>
                </div>

                <div
                  className="builder-gpu set-builder-drop"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                >
                  <h3>GPU {sharedGpu}</h3>
                  {gpu1Pct > OVER_BUDGET_PCT && <Banner message={messages.overBudget()} />}
                  <Meter capacity={{ used: gpu1Bytes, total: gpu1Total }} />

                  <div className="tenant-card">
                    <div className="tenant-card-head">
                      <span className="tenant-name">lemonade</span>
                    </div>
                    {placedModel ? (
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
                      // durable persists even though no placedModel could be
                      // derived from it (e.g. default_route_model without the
                      // "extra." prefix) — surfaced explicitly so it can
                      // never silently ride along on save. See IMPORTANT 4.
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
                        {lemonade?.state === "unloaded" && (
                          <p className="helper-text">currently set to unload</p>
                        )}
                      </>
                    )}
                  </div>

                  <div className="tenant-card">
                    <div className="tenant-card-head">
                      <span className="tenant-name">comfyui</span>
                    </div>
                    <div className="tenant-actions">
                      <button
                        type="button"
                        className={comfyChoice === "none" ? "primary" : undefined}
                        onClick={() => setComfyChoice("none")}
                      >
                        don't touch
                      </button>
                      <button
                        type="button"
                        className={comfyChoice === "leave" ? "primary" : undefined}
                        onClick={() => setComfyChoice("leave")}
                      >
                        leave (reserve)
                      </button>
                      <button
                        type="button"
                        className={comfyChoice === "free" ? "primary" : undefined}
                        onClick={() => setComfyChoice("free")}
                      >
                        free
                      </button>
                    </div>
                    {comfyChoice === "leave" && (
                      <label className="builder-field builder-field-inline">
                        reserve GB
                        <input
                          type="number"
                          min={1}
                          step={1}
                          value={reserveGbRaw}
                          onChange={(e) => updateReserveGb(e.target.value)}
                          onBlur={() => setReserveGbRaw(String(reserveGb))}
                        />
                      </label>
                    )}
                  </div>
                </div>
              </div>
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
