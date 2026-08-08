import { useEffect, useRef, useState } from "react";
import {
  getCatalog,
  getEffective,
  previewParse,
  previewRender,
  putSettings,
  type ArgValue,
  type Catalog,
  type EffectiveResponse,
  type SettingsKind,
  type SettingsWarning,
  type Widget,
} from "../api";
import { labels, messages } from "../model/messages";
import {
  bufferRemove,
  bufferSet,
  buildChips,
  displayValue,
  emptyBuffer,
  isDirty,
  LAYER_FOR_KIND,
  mergedArgsForPreview,
  parseValueText,
  POSITIONAL_KEY,
  scopeKeys,
  toPuts,
  type Buffer,
  type ChipView,
} from "../model/settingsView";
import Banner from "../ui/Banner";
import Modal from "../ui/Modal";
import ProvenanceDot from "../ui/ProvenanceDot";

/** Where the panel is pointed. `model === null` is the ENGINE-LEVEL entry
 * (the node card's "Engine settings" button): there is no checkpoint in
 * context, so the two model-scoped write kinds have no store key to write to
 * and their tabs are disabled.
 *
 * VERIFIED against the real route, not assumed (2026-08-07, via
 * tests/-harness TestClient over app.main.create_app): `GET
 * /api/settings/effective/{node}/{engine}/` — i.e. `getEffective(node,
 * engine, "")` — answers **200** with the ordinary `{resolved, argline,
 * warnings}` shape, carrying exactly the engine-scope view: the
 * `engine_defaults` derived layer and the `engines` store layer, with the
 * `models`/`engine_models` layers empty. That falls out of the route
 * definition rather than being incidental: `{model:path}` compiles to a
 * regex that matches the empty string, `_resolve` reads
 * `store.scope("models", "")` and `store.scope("engine_models",
 * "<node>/<engine>|")` (both simple dict lookups that miss and return `{}`),
 * `characteristics.entry("model/")` likewise misses and returns `{}`, and
 * `_facts_for` short-circuits to `{}` on a falsy model. So the engine-only
 * view needs no separate code path here. */
export interface SettingsTarget {
  node: string;
  engine: string;
  model: string | null;
}

/** app/settings_store.py:110's `KINDS`, in ladder order (least → most
 * specific). The segmented control renders them in exactly this order. */
const KIND_ORDER: SettingsKind[] = ["engines", "models", "engine_models"];

const PREVIEW_DEBOUNCE_MS = 300;

/**
 * The Settings panel: the five-layer resolution of one (node, engine, model)
 * triple, edited at one of three write scopes at a time.
 *
 * Nothing here writes until Save. Every edit lands in a `Buffer`
 * (model/settingsView.ts) keyed by write kind, so switching scopes never
 * discards a pending edit made at another one — Save ships them all, one PUT
 * per touched scope document.
 *
 * The remove ✕ and "Revert to inherited" are rendered ONLY when
 * `ChipView.setAtKind` is true. settingsView's `bufferRemove` is deliberately
 * a pure buffer operation with no `resolved` to judge legality against, so
 * this guard is the UI's job: buffering a remove for a key that is not set at
 * the selected scope would ship a PUT whose `remove` list names a key that
 * scope never had.
 */
export default function SettingsModal({
  target,
  onClose,
  onSaved,
}: {
  target: SettingsTarget;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { node, engine, model } = target;

  const [effective, setEffective] = useState<EffectiveResponse | null>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [kind, setKind] = useState<SettingsKind>(model ? "engine_models" : "engines");
  const [buffer, setBuffer] = useState<Buffer>(emptyBuffer);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [popover, setPopover] = useState<string | null>(null);
  const [argline, setArgline] = useState("");
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState("");
  const [importError, setImportError] = useState<string | null>(null);
  const [importWarnings, setImportWarnings] = useState<SettingsWarning[]>([]);

  // Escape must cancel an inline editor without buffering. Removing the
  // focused input from the DOM can fire blur on the way out in some engines,
  // and blur is what commits — so the cancel is recorded here first and the
  // blur handler defers to it.
  const cancelledEdit = useRef(false);

  const dirty = isDirty(buffer);

  useEffect(() => {
    let alive = true;
    setEffective(null);
    setLoadError(null);
    Promise.all([
      getEffective(node, engine, model ?? ""),
      // A pair that has never harvested returns 200 + JSON null — a
      // supported state, not an error (app/routers/settings.py:get_catalog).
      // Every chip then falls back to the "text" widget and no option has
      // choices. A transport failure is treated the same way: the panel is
      // still fully usable without a catalog, so it must not be blocked by
      // one.
      getCatalog(node, engine).catch(() => null),
    ]).then(
      ([eff, cat]) => {
        if (!alive) return;
        setEffective(eff);
        setCatalog(cat);
        setArgline(eff.argline);
      },
      (err) => {
        if (alive) setLoadError(err instanceof Error ? err.message : String(err));
      },
    );
    return () => {
      alive = false;
    };
  }, [node, engine, model]);

  // Live command preview. While the buffer is clean there is nothing to
  // preview: /effective's own `argline` IS the server's declared-only render
  // of this exact resolution, so re-asking for it would be one request to
  // learn what we already hold. Once dirty, the merged args go back to the
  // server 300ms after the last keystroke — the render direction of
  // POST /settings/preview, so the line on screen is the server's rendering
  // rules (shlex quoting, dash-shaped values in equals-form, positionals
  // first), never a second implementation of them in TypeScript.
  useEffect(() => {
    if (!effective) return;
    if (!dirty) {
      setArgline(effective.argline);
      setPreviewError(null);
      return;
    }
    let cancelled = false;
    const id = setTimeout(() => {
      previewRender(mergedArgsForPreview(effective.resolved, buffer), {
        node,
        engine,
        model: model ?? undefined,
      }).then(
        (res) => {
          if (cancelled) return;
          setArgline(res.argline);
          setPreviewError(null);
        },
        (err) => {
          // Deliberately keeps the last good argline on screen rather than
          // blanking it: a blank preview reads as "this command has no
          // flags", which is a far worse lie than a stale one the banner
          // above it labels as stale.
          if (cancelled) return;
          setPreviewError(err instanceof Error ? err.message : String(err));
        },
      );
    }, PREVIEW_DEBOUNCE_MS);
    // Also cancels the state update of an in-flight request, so a slow
    // response for an older buffer can never land on top of a newer one.
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [buffer, dirty, effective, node, engine, model]);

  const resolved = effective?.resolved ?? {};
  const { declared, applied } = buildChips(resolved, buffer, kind);

  // Warnings describe what the server currently HOLDS (validate_settings runs
  // over the saved resolution, app/routers/settings.py:get_effective), so a
  // key the operator just typed carries none until Save + refetch. That is
  // correct rather than a gap: nothing has validated it yet.
  const warningsFor = new Map<string, SettingsWarning[]>();
  for (const w of effective?.warnings ?? []) {
    const list = warningsFor.get(w.key);
    if (list) list.push(w);
    else warningsFor.set(w.key, [w]);
  }

  function widgetFor(name: string): Widget {
    // `_positional` is not a catalog option and never will be — it is a list
    // of tokens, so it gets the list editor explicitly.
    if (name === POSITIONAL_KEY) return "list";
    // app/harvest.py:widget_for produces exactly the five `Widget` values;
    // "text" is the honest fallback for a key with no harvested option.
    return catalog?.options[name]?.widget ?? "text";
  }

  function startEdit(chip: ChipView) {
    cancelledEdit.current = false;
    setPopover(null);
    setEditing(chip.name);
    setDraft(displayValue(chip.value));
  }

  function applyEdit(name: string, value: ArgValue) {
    setBuffer((b) => bufferSet(b, kind, name, value));
    setEditing(null);
  }

  /** Only ever called from a `setAtKind` chip — see the component docstring's
   * remove guard. */
  function removeAt(name: string) {
    setBuffer((b) => bufferRemove(b, kind, name));
    setEditing(null);
    setPopover(null);
  }

  async function handleSave() {
    if (!effective) return;
    setSaving(true);
    setSaveError(null);
    try {
      // Sequential, in toPuts order, aborting on the first failure. Each PUT
      // is its own scope document and the deck has no transaction across them
      // (app/routers/settings.py:put_settings writes one scope), so a failure
      // part-way through can leave EARLIER scopes already written. There is
      // deliberately no rollback: a compensating PUT is another write that
      // can fail exactly the same way, and it would be guessing at a prior
      // state this panel does not hold. The modal stays open with the error;
      // the refetch on close-and-reopen is what shows the truth.
      for (const put of toPuts(buffer, scopeKeys(node, engine, model))) {
        await putSettings(
          put.kind,
          put.key,
          "args",
          put.values,
          put.remove.length > 0 ? put.remove : undefined,
        );
      }
      onSaved();
      onClose();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  }

  async function applyImport() {
    setImportError(null);
    try {
      const res = await previewParse(importText, {
        node,
        engine,
        model: model ?? undefined,
      });
      setBuffer((b) => {
        let next = b;
        for (const [name, value] of Object.entries(res.parsed)) {
          next = bufferSet(next, kind, name, value);
        }
        return next;
      });
      setImportWarnings(res.warnings);
      setImportOpen(false);
      setImportText("");
    } catch (err) {
      setImportError(err instanceof Error ? err.message : String(err));
    }
  }

  function renderEditor(chip: ChipView) {
    const widget = widgetFor(chip.name);
    const option = catalog?.options[chip.name];

    if (widget === "toggle") {
      const checked = chip.value === true;
      return (
        <span className="settings-editor">
          <input
            type="checkbox"
            autoFocus
            aria-label={`--${chip.name}`}
            // Unchecking can only mean "stop declaring this flag": there is
            // no false-valued rendering of a bare flag (app/argline.py emits
            // the flag for `True` and `--flag False` for anything else), so a
            // declared `false` would ship a literal "False" argument. When
            // the key is not set at the selected scope there is nothing here
            // to remove either, and the title says which scope to switch to.
            title={chip.setAtKind ? undefined : labels.cannotUnsetHere}
            checked={checked}
            onChange={(e) => {
              if (e.target.checked) applyEdit(chip.name, true);
              else if (chip.setAtKind) removeAt(chip.name);
              else setEditing(null);
            }}
          />
        </span>
      );
    }

    if (widget === "select") {
      const choices = option?.choices ?? [];
      const current = displayValue(chip.value);
      // The current value always appears, even when the harvested choices no
      // longer contain it — a <select> that silently drops the live value
      // would re-save something the operator never picked.
      const options = choices.includes(current) ? choices : [current, ...choices];
      return (
        <span className="settings-editor">
          <select
            autoFocus
            aria-label={`--${chip.name}`}
            value={current}
            onChange={(e) => applyEdit(chip.name, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") setEditing(null);
            }}
          >
            {options.map((c) => (
              <option key={c} value={c}>
                {c === "" ? "—" : c}
              </option>
            ))}
          </select>
        </span>
      );
    }

    function commit() {
      const text = draft.trim();
      // An emptied field is NOT a removal: removal is the ✕ / Revert action,
      // and committing "" would ship `--flag ''` (shlex.quote of the empty
      // string). Cancelling the edit is the honest reading of "I deleted
      // what was in the box".
      if (text === "") {
        setEditing(null);
        return;
      }
      applyEdit(chip.name, widget === "list" ? parseValueText(chip.name, text) : text);
    }

    return (
      <span className="settings-editor">
        <input
          type={widget === "number" ? "number" : "text"}
          autoFocus
          className="settings-editor-input"
          aria-label={`--${chip.name}`}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              cancelledEdit.current = true;
              setEditing(null);
            }
          }}
          onBlur={() => {
            if (cancelledEdit.current) {
              cancelledEdit.current = false;
              return;
            }
            commit();
          }}
        />
      </span>
    );
  }

  function renderChip(chip: ChipView) {
    const warns = warningsFor.get(chip.name) ?? [];
    const positional = chip.name === POSITIONAL_KEY;
    // A pending set whose layer is outranked by a more specific DECLARED
    // layer still ships on Save but does not change the winning value —
    // buildChips reports that as `pendingSet` with the resolved (not
    // buffered) layer. Badging it is the only way that edit is visible.
    const shadowed = chip.pendingSet && chip.layer !== LAYER_FOR_KIND[kind];
    const classes = [
      "settings-chip",
      warns.length > 0 ? "settings-chip-warn" : null,
      chip.pendingRemove ? "settings-chip-removing" : null,
      chip.pendingSet ? "settings-chip-pending" : null,
    ]
      .filter(Boolean)
      .join(" ");

    return (
      <div className="settings-chip-wrap" key={`${chip.name}-${chip.layer}`}>
        <div className={classes}>
          <button
            type="button"
            className="settings-chip-dot"
            aria-label={labels.provenance}
            onClick={() => setPopover((p) => (p === chip.name ? null : chip.name))}
          >
            <ProvenanceDot layer={chip.layer} />
          </button>

          <span
            className="settings-chip-name"
            title={positional ? labels.positionalTitle : undefined}
          >
            {positional ? labels.positionalName : `--${chip.name}`}
          </span>

          {editing === chip.name ? (
            renderEditor(chip)
          ) : (
            <button
              type="button"
              className="settings-chip-value"
              title={labels.editOption}
              onClick={() => startEdit(chip)}
            >
              {displayValue(chip.value) || "·"}
            </button>
          )}

          {chip.pendingRemove && (
            <span className="settings-chip-badge">{labels.willBeRemovedBadge}</span>
          )}
          {chip.pendingSet && !chip.pendingRemove && (
            <span className="settings-chip-badge">
              {shadowed ? labels.shadowedBadge : labels.unsavedBadge}
            </span>
          )}

          {/* Guard: only a key actually set at the selected scope can be
              removed from it. See the component docstring. */}
          {chip.setAtKind && !chip.pendingRemove && (
            <button
              type="button"
              className="settings-chip-x"
              aria-label={labels.removeOverride}
              title={labels.removeOverrideTitle}
              onClick={() => removeAt(chip.name)}
            >
              ×
            </button>
          )}
        </div>

        {warns.map((w, i) => (
          <div className="settings-chip-warning" key={i}>
            {w.message}
          </div>
        ))}

        {popover === chip.name && (
          <div className="settings-popover" role="group" aria-label={labels.provenance}>
            <div className="settings-popover-row">
              <ProvenanceDot layer={chip.layer} />
              <span>{labels.layerName(chip.layer)}</span>
            </div>
            <div className="settings-popover-note">{labels.winningLayerOnly}</div>
            {chip.setAtKind && !chip.pendingRemove && (
              <button type="button" onClick={() => removeAt(chip.name)}>
                {labels.revertToInherited}
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  function renderApplied(chip: ChipView) {
    // Inert by construction: derived layers are never editable and never
    // shipped (app/routers/settings.py's declared-only rule), so this render
    // has no buttons at all rather than disabled ones.
    return (
      <div className="settings-chip settings-chip-applied" key={chip.name}>
        <ProvenanceDot layer={chip.layer} />
        <span className="settings-chip-name">{`--${chip.name}`}</span>
        <span className="settings-chip-value-static">{displayValue(chip.value) || "·"}</span>
      </div>
    );
  }

  return (
    <>
      <Modal
        title={labels.settingsTitle(engine, node)}
        subtitle={model ?? undefined}
        onClose={saving ? undefined : onClose}
        footer={
          <>
            <button type="button" onClick={onClose} disabled={saving}>
              {labels.cancel}
            </button>
            <button
              type="button"
              className="primary"
              onClick={handleSave}
              disabled={saving || !dirty}
            >
              {saving ? labels.saving : labels.save}
            </button>
          </>
        }
      >
        {loadError && <Banner message={messages.settingsLoadFailed(loadError)} />}
        {saveError && (
          <Banner
            message={messages.settingsSaveFailed(saveError)}
            onDismiss={() => setSaveError(null)}
          />
        )}
        {previewError && (
          <Banner
            message={messages.settingsPreviewFailed(previewError)}
            onDismiss={() => setPreviewError(null)}
          />
        )}

        {!effective && !loadError && <div className="settings-loading">{labels.loading}</div>}

        {effective && (
          <>
            <div className="settings-scopes" role="group" aria-label={labels.writeScope}>
              {KIND_ORDER.map((k) => {
                // Both model-scoped kinds need a model to build a store key
                // from (settingsView's scopeKeys returns null for them), so
                // they are unreachable from the engine-level entry.
                const unavailable = model === null && k !== "engines";
                return (
                  <button
                    key={k}
                    type="button"
                    className={k === kind ? "primary" : undefined}
                    aria-pressed={k === kind}
                    disabled={unavailable}
                    title={unavailable ? labels.needsModelContext : undefined}
                    onClick={() => {
                      setKind(k);
                      setEditing(null);
                      setPopover(null);
                    }}
                  >
                    {labels.kindName(k)}
                  </button>
                );
              })}
            </div>

            <div className="settings-section">
              <h4 className="settings-section-head">
                {labels.declaredSection}
                <span className="settings-section-hint">{labels.declaredSectionHint}</span>
              </h4>
              {declared.length === 0 ? (
                <div className="settings-empty">{labels.nothingDeclared}</div>
              ) : (
                <div className="settings-chips">{declared.map(renderChip)}</div>
              )}
            </div>

            <div className="settings-actions">
              <button type="button" disabled title={labels.addOptionTitle}>
                {labels.addOption}
              </button>
              <button type="button" onClick={() => setImportOpen(true)}>
                {labels.importArgline}
              </button>
            </div>

            <div className="settings-section">
              <h4 className="settings-section-head">
                {labels.appliedSection}
                <span className="settings-section-hint">{labels.appliedSectionHint}</span>
              </h4>
              {applied.length === 0 ? (
                <div className="settings-empty">{labels.nothingApplied}</div>
              ) : (
                <div className="settings-chips settings-chips-applied">
                  {applied.map(renderApplied)}
                </div>
              )}
            </div>

            {importWarnings.length > 0 && (
              <div className="settings-section">
                <h4 className="settings-section-head">{labels.warningsHeading}</h4>
                <ul className="settings-warnings">
                  {importWarnings.map((w, i) => (
                    <li key={i}>{w.message}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="settings-section">
              <h4 className="settings-section-head">{labels.commandPreview}</h4>
              <pre className="argline-preview">{argline}</pre>
              <div className="settings-note">{labels.appliesOnReload}</div>
            </div>
          </>
        )}
      </Modal>

      {importOpen && (
        <Modal
          title={labels.importArglineTitle}
          subtitle={labels.kindName(kind)}
          onClose={() => setImportOpen(false)}
          footer={
            <>
              <button type="button" onClick={() => setImportOpen(false)}>
                {labels.cancel}
              </button>
              <button
                type="button"
                className="primary"
                onClick={applyImport}
                disabled={importText.trim() === ""}
              >
                {labels.importApply}
              </button>
            </>
          }
        >
          {importError && (
            <Banner
              message={messages.settingsImportFailed(importError)}
              onDismiss={() => setImportError(null)}
            />
          )}
          <textarea
            className="settings-import"
            rows={5}
            autoFocus
            aria-label={labels.importArglineTitle}
            placeholder={labels.importArglinePlaceholder}
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
          />
          <div className="settings-note">{labels.importArglineHint}</div>
        </Modal>
      )}
    </>
  );
}
