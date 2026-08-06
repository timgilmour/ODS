import type { ReactNode } from "react";
import { labels } from "../model/messages";

/** The modal shell: overlay · header (title/subtitle/×) · body · footer.
 * Replaces the ad-hoc .modal-overlay/.modal-box conventions so every modal
 * carries the same chrome. Mounting IS "open" (ApplyModal's idiom): there is
 * no internal open state, the parent mounts/unmounts and owns any
 * poll-pause bookkeeping. `onClose` is optional because some phases must
 * not be abandonable (mid-apply); when absent, no × renders. */
export default function Modal({
  title,
  subtitle,
  onClose,
  footer,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  onClose?: () => void;
  footer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="ui-modal-overlay" role="dialog" aria-modal="true">
      <div className="ui-modal">
        <div className="ui-modal-head">
          <div className="ui-modal-titles">
            <h3 className="ui-modal-title">{title}</h3>
            {subtitle && <div className="ui-modal-subtitle">{subtitle}</div>}
          </div>
          {onClose && (
            <button type="button" aria-label={labels.close} onClick={onClose}>
              ×
            </button>
          )}
        </div>
        <div className="ui-modal-body">{children}</div>
        {footer && <div className="ui-modal-foot">{footer}</div>}
      </div>
    </div>
  );
}
