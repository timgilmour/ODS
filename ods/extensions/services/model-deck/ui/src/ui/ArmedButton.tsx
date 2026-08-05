import { useState } from "react";
import { messages } from "../model/messages";

/** Outlined, never filled — "available but deliberate" is the exact
 * affordance a force override wants. The second click is what actually
 * fires; the first only arms it. */
export default function ArmedButton({
  label,
  onConfirm,
  disabled = false,
}: {
  label: string;
  onConfirm: () => void;
  disabled?: boolean;
}) {
  const [armed, setArmed] = useState(false);

  return (
    <div className="armed-wrap">
      <button
        className="armed-button"
        disabled={disabled}
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
      {armed && <div className="armed-hint">{messages.forceConfirm().title}</div>}
    </div>
  );
}
