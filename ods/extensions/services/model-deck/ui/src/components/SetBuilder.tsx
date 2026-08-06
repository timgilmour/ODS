import { useEffect, useState, type DragEvent } from "react";
import {
  ApiError,
  bytesToGB,
  deleteSet,
  getSets,
  meterFillClass,
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
import {
  buildDraft,
  derivePlacedModel,
  draftEquals,
  EXTRA_PREFIX,
  fieldsFromSet,
} from "../model/setDraft";
import ApplyModal from "./ApplyModal";
import ModelLibrary from "./ModelLibrary";

interface SetBuilderProps {
  models: ModelFile[];
  gpus: Gpu[];
  world: World;
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

// Snapshot taken the instant a 409 fires on save: the exact draft + the
// slug it collided with. Frozen at that moment rather than re-derived from
// live state later, so Overwrite can never act on a draft edited (or a
// different name typed) after the confirm banner appeared — see CRITICAL 1
// in the review that produced this fix.
interface OverwriteSnapshot {
  draft: ConfigSet;
  slug: string;
}

/** Drag-and-drop set editor: the hipfire column (a toggle, not a drop
 * target) and the lemonade/comfyui column (a drop target) mirror
 * GpuColumn's tenant grouping, but render bespoke controls per tenant
 * rather than TenantCard's live-status view — this is a *draft* of desired
 * state, not a live status card. Column indices (which physical GPU each
 * one is) come from the world snapshot's placement map, not a hardcoded
 * layout — GpuColumn no longer hardcodes it either. */
export default function SetBuilder({ models, gpus, world, onModalOpenChange }: SetBuilderProps) {
  // Existing sets, for the load/duplicate/delete select.
  const [sets, setSets] = useState<ConfigSet[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState("");

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

  // Delete flow (two-click, no window.confirm).
  const [deleteArmed, setDeleteArmed] = useState(false);

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
    setPolicyOverrides(null);
    setSaveError(null);
    setOverwriteSnapshot(null);
    setDeleteArmed(false);
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
    setReserveGb(cfgset.ephemeral?.comfyui?.reserve_gb ?? 24);
    setPlacedModel(derivePlacedModel(cfgset));
    setSaveError(null);
    setOverwriteSnapshot(null);
    setDeleteArmed(false);
  }

  function handleLoad() {
    const cfgset = sets.find((s) => slugify(s.name) === selectedSlug);
    if (!cfgset) return;
    populateFromSet(cfgset, false);
    setSavedSlug(selectedSlug);
    setSavedSet(cfgset);
  }

  function handleDuplicate() {
    const cfgset = sets.find((s) => slugify(s.name) === selectedSlug);
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

  async function confirmDelete() {
    if (!savedSlug) return;
    try {
      await deleteSet(savedSlug);
      setDeleteArmed(false);
      setSavedSlug(null);
      setSavedSet(null);
      resetDraft();
      refreshSets();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
      setDeleteArmed(false);
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
  function updateReserveGb(value: number) {
    const clamped = Math.max(1, Math.round(value || 0));
    setReserveGb(clamped);
    if (comfyui) setComfyui({ ...comfyui, reserve_gb: clamped });
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
      {/* Rendered OUTSIDE the disabled fieldset below so Overwrite/Cancel
          stay clickable while every draft-mutating control is locked — the
          target name is read from the frozen snapshot, never the
          (now-disabled, but belt-and-suspenders-safe) live draft. */}
      {overwriteSnapshot && (
        <div className="banner-error builder-overwrite-banner">
          <span>Overwrite set '{overwriteSnapshot.slug}'?</span>
          <button className="primary" onClick={() => handleSave(true)} disabled={saving}>
            Overwrite
          </button>
          <button onClick={cancelOverwrite} disabled={saving}>
            Cancel
          </button>
        </div>
      )}

      <div className="builder-layout">
        {/* Disabled as a whole (native fieldset cascade covers every
            input/textarea/select/button inside, including ModelLibrary's
            Place buttons) whenever an overwrite confirmation is pending —
            see CRITICAL 1. The lemonade/comfyui drop target isn't a form control, so
            handleDrop also short-circuits explicitly while a snapshot is
            pending. */}
        <fieldset className="builder-fieldset" disabled={overwriteSnapshot !== null}>
          <ModelLibrary models={models} onPlace={placeModel} targetGpu={sharedGpu} />

          <div className="builder-main">
            <div className="panel">
              <h2>Set Builder</h2>

              <div className="builder-load-row">
                <select
                  aria-label="load an existing set"
                  value={selectedSlug}
                  onChange={(e) => setSelectedSlug(e.target.value)}
                >
                  <option value="">select a saved set…</option>
                  {sets.map((s) => (
                    <option key={s.name} value={slugify(s.name)}>
                      {s.name}
                    </option>
                  ))}
                </select>
                <button onClick={handleLoad} disabled={!selectedSlug}>
                  Load
                </button>
                <button onClick={handleDuplicate} disabled={!selectedSlug}>
                  Duplicate
                </button>
              </div>
              {listError && <div className="banner-error"><span>{listError}</span></div>}

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
                <div className="banner-error">
                  <span>{saveError}</span>
                  <button onClick={() => setSaveError(null)} aria-label="dismiss error">
                    ×
                  </button>
                </div>
              )}

              <div className="builder-actions">
                <button
                  className="primary"
                  onClick={() => handleSave(false)}
                  disabled={saving || !draft.name}
                >
                  {saving ? "Saving…" : "Save"}
                </button>

                {!deleteArmed ? (
                  <button
                    onClick={() => setDeleteArmed(true)}
                    disabled={!savedSlug}
                    title={!savedSlug ? "load or save a set first" : undefined}
                  >
                    Delete
                  </button>
                ) : (
                  <>
                    <button className="primary" onClick={confirmDelete}>
                      Really delete?
                    </button>
                    <button onClick={() => setDeleteArmed(false)}>Cancel</button>
                  </>
                )}

                <button
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
                  Preview steps
                </button>
              </div>
            </div>

            <div className="gpu-row">
              <div className="gpu-column">
                <h2>GPU {hipfireGpu}</h2>
                {gpu0Pct > 90 && (
                  <div className="banner-error">
                    <span>Over budget — loads may fail</span>
                  </div>
                )}
                <div className="gpu-meter">
                  <div className="meter-track">
                    <div
                      className={
                        hipfire === null
                          ? "meter-fill meter-fill-subdued"
                          : meterFillClass(gpu0Pct)
                      }
                      style={{ width: `${Math.min(gpu0Pct, 100)}%` }}
                    />
                  </div>
                  <div className="meter-label">
                    {bytesToGB(gpu0Bytes)} / {bytesToGB(gpu0Total)} GB ({gpu0Pct.toFixed(0)}%)
                    {hipfire === null && " · current live state"}
                  </div>
                </div>

                <div className="tenant-card">
                  <div className="tenant-card-head">
                    <span className="tenant-name">hipfire</span>
                  </div>
                  <div className="tenant-actions">
                    <button
                      className={hipfireChoice === "none" ? "primary" : undefined}
                      onClick={() => setHipfireChoice("none")}
                    >
                      don't touch
                    </button>
                    <button
                      className={hipfireChoice === "running" ? "primary" : undefined}
                      onClick={() => setHipfireChoice("running")}
                    >
                      Running
                    </button>
                    <button
                      className={hipfireChoice === "parked" ? "primary" : undefined}
                      onClick={() => setHipfireChoice("parked")}
                    >
                      Parked
                    </button>
                  </div>
                </div>
              </div>

              <div
                className="gpu-column set-builder-drop"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
              >
                <h2>GPU {sharedGpu}</h2>
                {gpu1Pct > 90 && (
                  <div className="banner-error">
                    <span>Over budget — loads may fail</span>
                  </div>
                )}
                <div className="gpu-meter">
                  <div className="meter-track">
                    <div
                      className={meterFillClass(gpu1Pct)}
                      style={{ width: `${Math.min(gpu1Pct, 100)}%` }}
                    />
                  </div>
                  <div className="meter-label">
                    {bytesToGB(gpu1Bytes)} / {bytesToGB(gpu1Total)} GB ({gpu1Pct.toFixed(0)}%)
                  </div>
                </div>

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
                        <button onClick={removeModel}>remove model</button>
                      </div>
                    </>
                  ) : durable ? (
                    // durable persists even though no placedModel could be
                    // derived from it (e.g. default_route_model without the
                    // "extra." prefix) — surfaced explicitly so it can
                    // never silently ride along on save. See IMPORTANT 4.
                    <div className="durable-chip">
                      <span>durable route: {durable.default_route_model}</span>
                      <button onClick={() => setDurable(null)} aria-label="clear durable route">
                        ✕
                      </button>
                    </div>
                  ) : (
                    <div className="dropzone-empty">
                      Drag a model here from the library, or use its Place button.
                      {lemonade?.state === "unloaded" && " (currently set to unload)"}
                    </div>
                  )}
                </div>

                <div className="tenant-card">
                  <div className="tenant-card-head">
                    <span className="tenant-name">comfyui</span>
                  </div>
                  <div className="tenant-actions">
                    <button
                      className={comfyChoice === "none" ? "primary" : undefined}
                      onClick={() => setComfyChoice("none")}
                    >
                      don't touch
                    </button>
                    <button
                      className={comfyChoice === "leave" ? "primary" : undefined}
                      onClick={() => setComfyChoice("leave")}
                    >
                      leave (reserve)
                    </button>
                    <button
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
                        value={reserveGb}
                        onChange={(e) => updateReserveGb(Number(e.target.value))}
                      />
                    </label>
                  )}
                </div>
              </div>
            </div>
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
