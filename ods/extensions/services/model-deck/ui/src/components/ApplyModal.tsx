import { useEffect, useState } from "react";
import {
  ApiError,
  applySet,
  previewSet,
  type ApplyReport,
  type ConfigSet,
  type PreviewResponse,
} from "../api";
import { previewRows, reportRows } from "../model/applySteps";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Modal from "../ui/Modal";
import StepList from "../ui/StepList";

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
  // Two error slots, not one: the preview fetch (a read) and confirmApply's
  // catch (a mutation refusal) render under different Message factories —
  // stateRefreshFailed vs guardRefused — so which one fired has to survive
  // past the catch block. Only one is ever non-null at a time.
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [applyError, setApplyError] = useState<string | null>(null);
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
        setPreviewError(err instanceof Error ? err.message : String(err));
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
      setApplyError(err instanceof Error ? err.message : String(err));
      setOfferForce(err instanceof ApiError && err.status === 409);
      setPhase("confirm");
    }
  }

  return (
    <Modal
      title={cfgset.name}
      subtitle={cfgset.notes || undefined}
      onClose={phase === "applying" ? undefined : onClose}
      footer={
        <>
          {phase === "preview" && (
            <button type="button" onClick={onClose}>{labels.cancel}</button>
          )}
          {phase === "confirm" && (
            <>
              <button type="button" onClick={onClose}>{labels.cancel}</button>
              {offerForce && (
                <button
                  type="button"
                  onClick={() => confirmApply(true)}
                  title={labels.forceApplyTitle}
                >
                  {labels.forceApply}
                </button>
              )}
              <button type="button" className="primary" onClick={() => confirmApply()}>
                {labels.confirm}
              </button>
            </>
          )}
          {phase === "applying" && (
            <button type="button" disabled>{labels.applying}</button>
          )}
          {phase === "result" && (
            <button type="button" className="primary" onClick={onClose}>
              {labels.close}
            </button>
          )}
        </>
      }
    >
      {phase === "preview" && !previewError && <p>{labels.loadingPreview}</p>}

      {previewError && (
        <Banner
          message={messages.stateRefreshFailed(previewError)}
          onDismiss={() => setPreviewError(null)}
        />
      )}

      {applyError && (
        <Banner
          message={messages.guardRefused(applyError)}
          onDismiss={() => setApplyError(null)}
        />
      )}

      {preview && (phase === "confirm" || phase === "applying") && (
        <>
          {preview.steps.length === 0 ? (
            <p>{labels.noChanges}</p>
          ) : (
            <StepList items={previewRows(preview.steps)} />
          )}
          <p className="helper-text">{labels.estimate(preview.estimate_s)}</p>
        </>
      )}

      {phase === "result" && report && (
        <>
          <StepList items={reportRows(report)} />
          <p className="helper-text">{labels.stepsCompleted(report.completed.length)}</p>
          {report.error && (
            <Banner message={messages.guardRefused(report.error)} />
          )}
          {report.warnings.length > 0 && (
            <ul className="ui-warnings">
              {report.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </Modal>
  );
}
