import { useEffect, useState } from "react";
import { messages } from "../model/messages";

/** Outlined, never filled — "available but deliberate" is the exact
 * affordance a force override wants. The second click is what actually
 * fires; the first only arms it.
 *
 * `type="button"` is not decoration on a shared primitive: a <button> with
 * no type defaults to type="submit", so the first screen that nests one of
 * these inside a <form> would submit that form on every click. Banner was
 * given it when the argument was first made; ArmedButton, Toolbar and
 * ModelChip were missed, and this is the one where a spurious submit would
 * ride alongside a destructive override. */
export default function ArmedButton({
  label,
  onConfirm,
  disabled = false,
  resetToken = 0,
}: {
  label: string;
  onConfirm: () => void;
  disabled?: boolean;
  resetToken?: number;
}) {
  const [armed, setArmed] = useState(false);

  // Disarm whenever the resetToken changes (on every new refusal)
  useEffect(() => setArmed(false), [resetToken]);

  return (
    <div className="armed-wrap">
      <button
        type="button"
        className="armed-button"
        disabled={disabled}
        aria-label={armed ? `${label} — ${messages.forceConfirm().title}` : label}
        onClick={() => {
          if (armed) {
            setArmed(false);
            onConfirm();
          } else {
            setArmed(true);
          }
        }}
      >
        ⚠ {label}
      </button>
      {armed && <div className="armed-hint" role="status">{messages.forceConfirm().title}</div>}
    </div>
  );
}
