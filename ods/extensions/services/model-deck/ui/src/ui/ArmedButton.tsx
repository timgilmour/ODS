import { messages } from "../model/messages";

/** Outlined, never filled — "available but deliberate" is the exact
 * affordance a force override wants. The second click is what actually
 * fires; the first only arms it.
 *
 * CONTROLLED, and that is the safety property, not a style preference. This
 * component holds no `armed` state and has no disarming effect: the owner
 * decides, by comparing the refusal it armed against with the refusal
 * currently on screen (model/armed.ts). While `armed` lived here, a retry of
 * the same guarded action updated this instance instead of remounting it and
 * a stale arming survived into a refusal the operator never clicked — one
 * click from a force override. There is now no state here to go stale.
 *
 * `type="button"` is not decoration on a shared primitive: a <button> with
 * no type defaults to type="submit", so the first screen that nests one of
 * these inside a <form> would submit that form on every click. */
export default function ArmedButton({
  label,
  armed,
  onArm,
  onConfirm,
  disabled = false,
}: {
  label: string;
  /** Whether this button is armed. Derived by the owner; never local. */
  armed: boolean;
  /** First click: the operator wants this available, not fired. */
  onArm: () => void;
  /** Second click, on the same refusal: fire it. */
  onConfirm: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="armed-wrap">
      <button
        type="button"
        className="armed-button"
        disabled={disabled}
        // The accessible name changes with the state, so a screen reader
        // that re-reads the control after the first click hears that its
        // meaning changed rather than the same two words twice.
        aria-label={armed ? `${label} — ${messages.forceConfirm().title}` : label}
        onClick={armed ? onConfirm : onArm}
      >
        ⚠ {label}
      </button>
      {armed && <div className="armed-hint" role="status">{messages.forceConfirm().title}</div>}
    </div>
  );
}
