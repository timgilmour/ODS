import { useEffect, useState } from "react";
import {
  ApiError,
  applySet,
  previewSet,
  type ApplyReport,
  type ConfigSet,
  type PreviewResponse,
} from "../api";

type Phase = "preview" | "confirm" | "applying" | "result";

interface ApplyModalProps {
  slug: string;
  cfgset: ConfigSet;
  onClose: () => void;
}

/** Preview -> confirm -> apply -> result modal for one config set. Shared by
 * SetStrip (Deck tab's saved-set buttons) and SetBuilder ("Preview steps",
 * only reachable once the draft is saved) so the two surfaces can never
 * drift on this flow.
 *
 * Mounting the component IS "open" — there's no internal open/closed state.
 * The parent decides when to mount/unmount and owns any onModalOpenChange
 * (poll-pause) bookkeeping around that; fetching the preview starts
 * immediately on mount. */
export default function ApplyModal({ slug, cfgset, onClose }: ApplyModalProps) {
  const [phase, setPhase] = useState<Phase>("preview");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [report, setReport] = useState<ApplyReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Armed when the apply 409s off the hipfire conversation-guard, so the
  // retry can offer force=true as one click instead of a curl command.
  const [offerForce, setOfferForce] = useState(false);

  useEffect(() => {
    let cancelled = false;
    previewSet(slug)
      .then((p) => {
        if (cancelled) return;
        setPreview(p);
        setPhase("confirm");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [slug]);

  async function confirmApply(force = false) {
    setPhase("applying");
    try {
      const r = await applySet(slug, force);
      setReport(r);
      setPhase("result");
      setOfferForce(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setOfferForce(err instanceof ApiError && err.status === 409);
      setPhase("confirm");
    }
  }

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-box">
        <h3>{cfgset.name}</h3>
        {cfgset.notes && <p className="modal-notes">{cfgset.notes}</p>}

        {phase === "preview" && !error && <p>Loading preview…</p>}

        {error && (
          <div className="banner-error">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="dismiss error">
              ×
            </button>
          </div>
        )}

        {preview && (phase === "confirm" || phase === "applying") && (
          <>
            <div className="step-list">
              {preview.steps.length === 0 ? (
                <div>(no changes — already matches this set)</div>
              ) : (
                preview.steps.map((step, i) => <div key={i}>{JSON.stringify(step)}</div>)
              )}
            </div>
            <p>~{preview.estimate_s}s</p>
          </>
        )}

        {phase === "result" && report && (
          <div className="apply-report">
            <p>{report.completed.length} step(s) completed.</p>
            {report.failed && (
              <p className="failed">
                Failed at {JSON.stringify(report.failed)}: {report.error}
              </p>
            )}
            {report.warnings.length > 0 && (
              <ul className="warnings">
                {report.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="modal-actions">
          {phase === "preview" && <button onClick={onClose}>Cancel</button>}
          {phase === "confirm" && (
            <>
              <button onClick={onClose}>Cancel</button>
              {offerForce && (
                <button
                  onClick={() => confirmApply(true)}
                  title="override the hipfire conversation-guard — the live/recent conversation will lose its cache and its next turn will re-read the whole history"
                >
                  Force apply
                </button>
              )}
              <button className="primary" onClick={() => confirmApply()}>
                Confirm
              </button>
            </>
          )}
          {phase === "applying" && <button disabled>Applying…</button>}
          {phase === "result" && (
            <button className="primary" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
