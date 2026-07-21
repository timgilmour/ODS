import { useEffect, useState } from "react";
import {
  applySet,
  getSets,
  previewSet,
  slugify,
  PREVIOUS_SLUG,
  type ApplyReport,
  type ConfigSet,
  type PreviewResponse,
} from "../api";

interface SetStripProps {
  token: string;
  onModalOpenChange: (open: boolean) => void;
  onChanged: () => void;
}

type Phase = "preview" | "confirm" | "applying" | "result";

interface ModalState {
  phase: Phase;
  slug: string;
  cfgset: ConfigSet;
  preview: PreviewResponse | null;
  report: ApplyReport | null;
  error: string | null;
}

/** Horizontal row of saved config-set buttons. Click -> preview -> modal
 * with the plan + estimate -> Confirm -> apply -> report. Close refetches
 * everything (both this component's own set list and the parent's world
 * state, via onChanged). */
export default function SetStrip({ token, onModalOpenChange, onChanged }: SetStripProps) {
  const [sets, setSets] = useState<ConfigSet[]>([]);
  const [previous, setPrevious] = useState<ConfigSet | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [modal, setModal] = useState<ModalState | null>(null);

  function refreshList() {
    getSets()
      .then((r) => {
        setSets(r.sets);
        setPrevious(r.previous);
        setListError(null);
      })
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(refreshList, []);

  async function openPreview(slug: string, cfgset: ConfigSet) {
    setModal({ phase: "preview", slug, cfgset, preview: null, report: null, error: null });
    onModalOpenChange(true);
    try {
      const preview = await previewSet(slug);
      setModal((m) => (m ? { ...m, phase: "confirm", preview } : m));
    } catch (err) {
      setModal((m) =>
        m ? { ...m, error: err instanceof Error ? err.message : String(err) } : m,
      );
    }
  }

  async function confirmApply() {
    if (!modal) return;
    setModal({ ...modal, phase: "applying" });
    try {
      const report = await applySet(modal.slug);
      setModal((m) => (m ? { ...m, phase: "result", report } : m));
    } catch (err) {
      setModal((m) =>
        m
          ? { ...m, phase: "confirm", error: err instanceof Error ? err.message : String(err) }
          : m,
      );
    }
  }

  function closeModal() {
    setModal(null);
    onModalOpenChange(false);
    refreshList();
    onChanged();
  }

  return (
    <div className="panel">
      <h2>Config sets</h2>
      {listError && <div className="banner-error"><span>{listError}</span></div>}
      <div className="set-strip">
        {sets.map((s) => (
          <button
            key={s.name}
            onClick={() => openPreview(slugify(s.name), s)}
            disabled={!token}
            title={!token ? "admin token required" : undefined}
          >
            {s.name}
          </button>
        ))}
        {previous && (
          <button
            className="set-btn-previous"
            onClick={() => openPreview(PREVIOUS_SLUG, previous)}
            disabled={!token}
            title={!token ? "admin token required" : undefined}
          >
            {previous.name}
          </button>
        )}
        {sets.length === 0 && !previous && <span>no saved sets</span>}
      </div>

      {modal && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal-box">
            <h3>{modal.cfgset.name}</h3>
            {modal.cfgset.notes && <p className="modal-notes">{modal.cfgset.notes}</p>}

            {modal.phase === "preview" && !modal.error && <p>Loading preview…</p>}

            {modal.error && (
              <div className="banner-error">
                <span>{modal.error}</span>
                <button onClick={() => setModal((m) => (m ? { ...m, error: null } : m))}>
                  ×
                </button>
              </div>
            )}

            {modal.preview && (modal.phase === "confirm" || modal.phase === "applying") && (
              <>
                <div className="step-list">
                  {modal.preview.steps.length === 0 ? (
                    <div>(no changes — already matches this set)</div>
                  ) : (
                    modal.preview.steps.map((step, i) => (
                      <div key={i}>{JSON.stringify(step)}</div>
                    ))
                  )}
                </div>
                <p>~{modal.preview.estimate_s}s</p>
              </>
            )}

            {modal.phase === "result" && modal.report && (
              <div className="apply-report">
                <p>{modal.report.completed.length} step(s) completed.</p>
                {modal.report.failed && (
                  <p className="failed">
                    Failed at {JSON.stringify(modal.report.failed)}: {modal.report.error}
                  </p>
                )}
                {modal.report.warnings.length > 0 && (
                  <ul className="warnings">
                    {modal.report.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <div className="modal-actions">
              {modal.phase === "preview" && <button onClick={closeModal}>Cancel</button>}
              {modal.phase === "confirm" && (
                <>
                  <button onClick={closeModal}>Cancel</button>
                  <button className="primary" onClick={confirmApply}>
                    Confirm
                  </button>
                </>
              )}
              {modal.phase === "applying" && <button disabled>Applying…</button>}
              {modal.phase === "result" && (
                <button className="primary" onClick={closeModal}>
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
