import { truncateMiddle } from "../api";
import type { StepItem } from "../model/applySteps";

/** Vertical step list (Activate modal). Marker colour carries state; the
 * label always carries it too (never colour alone). */
export default function StepList({ items }: { items: StepItem[] }) {
  return (
    <ol className="ui-steplist">
      {items.map((it) => (
        <li key={it.key} className={`ui-step ui-step-${it.state}`}>
          <span className="ui-step-marker" aria-hidden="true" />
          <span className="ui-step-label">{it.label}</span>
          {it.detail && (
            <span className="ui-step-detail" title={it.detail}>
              {truncateMiddle(it.detail, 40)}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
