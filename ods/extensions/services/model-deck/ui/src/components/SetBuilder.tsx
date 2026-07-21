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
} from "../api";
import ApplyModal from "./ApplyModal";
import ModelLibrary from "./ModelLibrary";

interface SetBuilderProps {
  models: ModelFile[];
  gpus: Gpu[];
  token: string;
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

const EXTRA_PREFIX = "extra.";

function emptyEphemeral(
  lemonade: LemonadeEphemeral | null,
  comfyui: ComfyuiEphemeral | null,
  hipfire: HipfireEphemeral,
) {
  return { lemonade, comfyui, hipfire };
}

/** Drag-and-drop set editor: GPU 0 (hipfire toggle, not a drop target) and
 * GPU 1 (lemonade/comfyui, a drop target) mirror GpuColumn's fixed
 * engine->GPU placement (see GpuColumn.tsx:GPU_TENANTS) but render bespoke
 * controls per tenant rather than TenantCard's live-status view — this is
 * a *draft* of desired state, not a live status card. */
export default function SetBuilder({ models, gpus, token, onModalOpenChange }: SetBuilderProps) {
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
  const [hipfire, setHipfire] = useState<HipfireEphemeral>({ state: "running" });
  const [placedModel, setPlacedModel] = useState<string | null>(null);
  const [makeDefault, setMakeDefault] = useState(true);
  const [catalogId, setCatalogId] = useState("");
  const [reserveGb, setReserveGb] = useState(24);

  // Save flow.
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [confirmOverwrite, setConfirmOverwrite] = useState(false);
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
    setHipfire({ state: "running" });
    setPlacedModel(null);
    setMakeDefault(true);
    setCatalogId("");
    setReserveGb(24);
    setSaveError(null);
    setConfirmOverwrite(false);
    setDeleteArmed(false);
  }

  function populateFromSet(cfgset: ConfigSet, clearName: boolean) {
    setName(clearName ? "" : cfgset.name);
    setNotes(cfgset.notes);
    setDurable(cfgset.durable);
    setLemonade(cfgset.ephemeral?.lemonade ?? null);
    setComfyui(cfgset.ephemeral?.comfyui ?? null);
    setHipfire(cfgset.ephemeral?.hipfire ?? { state: "running" });
    setMakeDefault(cfgset.durable !== null);
    setCatalogId(cfgset.durable?.activate_model_id ?? "");
    setReserveGb(cfgset.ephemeral?.comfyui?.reserve_gb ?? 24);
    // Best-effort: only durable.default_route_model with the "extra."
    // prefix + an active "loaded" ephemeral intent can be traced back to a
    // library model file. A set that names some other litellm route, or
    // that was saved with "make default route" unchecked, can't be — the
    // model chip is left blank and the user re-drops to pin one down.
    const derived = cfgset.durable?.default_route_model.startsWith(EXTRA_PREFIX)
      ? cfgset.durable.default_route_model.slice(EXTRA_PREFIX.length)
      : null;
    setPlacedModel(cfgset.ephemeral?.lemonade?.state === "loaded" ? derived : null);
    setSaveError(null);
    setConfirmOverwrite(false);
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

  function buildDraft(): ConfigSet {
    return {
      name: name.trim(),
      notes,
      durable,
      ephemeral: emptyEphemeral(lemonade, comfyui, hipfire),
      policy_overrides: null,
    };
  }

  const draft = buildDraft();
  const isSavedUnchanged = savedSet !== null && JSON.stringify(draft) === JSON.stringify(savedSet);

  async function handleSave(overwrite: boolean) {
    if (!draft.name) return;
    setSaving(true);
    setSaveError(null);
    try {
      const { slug } = await saveSet(draft, overwrite);
      setSavedSlug(slug);
      setSavedSet(draft);
      setConfirmOverwrite(false);
      refreshSets();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setConfirmOverwrite(true);
      } else {
        setSaveError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setSaving(false);
    }
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
    setDurable(
      makeDefault
        ? { default_route_model: `${EXTRA_PREFIX}${file}`, activate_model_id: catalogId.trim() || null }
        : null,
    );
  }

  function toggleMakeDefault(checked: boolean) {
    setMakeDefault(checked);
    if (!placedModel) return;
    setDurable(
      checked
        ? {
            default_route_model: `${EXTRA_PREFIX}${placedModel}`,
            activate_model_id: catalogId.trim() || null,
          }
        : null,
    );
  }

  function updateCatalogId(value: string) {
    setCatalogId(value);
    if (placedModel && makeDefault) {
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
    const file = e.dataTransfer.getData("text/plain");
    // Malformed/unknown drops (empty payload, or a file not in the current
    // registry scan) are ignored silently rather than surfaced as an error
    // — a stray drag from elsewhere on the page shouldn't corrupt the draft.
    if (!file || !models.some((m) => m.file === file)) return;
    placeModel(file);
  }

  // --- ComfyUI three-way (don't touch / leave-reserve / free) -------------

  function setComfyChoice(choice: "none" | "leave" | "free") {
    if (choice === "none") {
      setComfyui(null);
      return;
    }
    setComfyui({ state: choice, reserve_gb: reserveGb });
  }

  function updateReserveGb(value: number) {
    const clamped = Math.max(0, value || 0);
    setReserveGb(clamped);
    if (comfyui) setComfyui({ ...comfyui, reserve_gb: clamped });
  }

  // --- Footprint budgets ----------------------------------------------------

  const gpu0Total = gpus.find((g) => g.index === 0)?.total ?? DEFAULT_GPU_BYTES;
  const gpu1Total = gpus.find((g) => g.index === 1)?.total ?? DEFAULT_GPU_BYTES;

  const placedFootprint = placedModel
    ? (models.find((m) => m.file === placedModel)?.footprint ?? 0)
    : 0;
  const comfyReserveBytes = comfyui?.state === "leave" ? comfyui.reserve_gb * 1e9 : 0;

  const gpu0Bytes = hipfire.state === "running" ? HIPFIRE_FOOTPRINT_BYTES : 0;
  const gpu1Bytes = placedFootprint + comfyReserveBytes;

  const gpu0Pct = gpu0Total > 0 ? (gpu0Bytes / gpu0Total) * 100 : 0;
  const gpu1Pct = gpu1Total > 0 ? (gpu1Bytes / gpu1Total) * 100 : 0;

  const comfyChoice: "none" | "leave" | "free" = comfyui === null ? "none" : comfyui.state;

  return (
    <div className="builder-layout">
      <ModelLibrary models={models} onPlace={placeModel} />

      <div className="builder-main">
        <div className="panel">
          <h2>Set Builder</h2>

          {!token && (
            <div className="banner-error">
              <span>read-only — set an admin token to save, delete, or preview sets</span>
            </div>
          )}

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

          {saveError && (
            <div className="banner-error">
              <span>{saveError}</span>
              <button onClick={() => setSaveError(null)} aria-label="dismiss error">
                ×
              </button>
            </div>
          )}

          {confirmOverwrite && (
            <div className="banner-error">
              <span>A set with this slug exists — overwrite?</span>
              <button onClick={() => handleSave(true)} disabled={saving}>
                Overwrite
              </button>
              <button onClick={() => setConfirmOverwrite(false)}>Cancel</button>
            </div>
          )}

          <div className="builder-actions">
            <button
              className="primary"
              onClick={() => handleSave(false)}
              disabled={!token || saving || !draft.name}
            >
              {saving ? "Saving…" : "Save"}
            </button>

            {!deleteArmed ? (
              <button
                onClick={() => setDeleteArmed(true)}
                disabled={!token || !savedSlug}
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
              disabled={!token || !savedSlug || !isSavedUnchanged}
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
            <h2>GPU 0</h2>
            {gpu0Pct > 90 && (
              <div className="banner-error">
                <span>Over budget — loads may fail</span>
              </div>
            )}
            <div className="gpu-meter">
              <div className="meter-track">
                <div
                  className={meterFillClass(gpu0Pct)}
                  style={{ width: `${Math.min(gpu0Pct, 100)}%` }}
                />
              </div>
              <div className="meter-label">
                {bytesToGB(gpu0Bytes)} / {bytesToGB(gpu0Total)} GB ({gpu0Pct.toFixed(0)}%)
              </div>
            </div>

            <div className="tenant-card">
              <div className="tenant-card-head">
                <span className="tenant-name">hipfire</span>
              </div>
              <div className="tenant-actions">
                <button
                  className={hipfire.state === "running" ? "primary" : undefined}
                  onClick={() => setHipfire({ state: "running" })}
                >
                  Running
                </button>
                <button
                  className={hipfire.state === "parked" ? "primary" : undefined}
                  onClick={() => setHipfire({ state: "parked" })}
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
            <h2>GPU 1</h2>
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
                  <label className="builder-checkbox">
                    <input
                      type="checkbox"
                      checked={makeDefault}
                      onChange={(e) => toggleMakeDefault(e.target.checked)}
                    />
                    make default route
                  </label>
                  {makeDefault && (
                    <input
                      type="text"
                      className="builder-catalog-id"
                      placeholder="catalog id (optional, enables durable revert)"
                      value={catalogId}
                      onChange={(e) => updateCatalogId(e.target.value)}
                    />
                  )}
                  <div className="tenant-actions">
                    <button onClick={removeModel}>remove model</button>
                  </div>
                </>
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
                    min={0}
                    value={reserveGb}
                    onChange={(e) => updateReserveGb(Number(e.target.value))}
                  />
                </label>
              )}
            </div>
          </div>
        </div>
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
    </div>
  );
}
