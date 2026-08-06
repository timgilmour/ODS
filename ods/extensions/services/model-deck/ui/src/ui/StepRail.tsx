import type { RailStop } from "../model/moveRail";

/** Horizontal progress rail (Move modal): dot + label per stop, a
 * connecting line between stops. */
export default function StepRail({ stops }: { stops: RailStop[] }) {
  return (
    <div className="ui-steprail">
      {stops.map((s) => (
        <div key={s.label} className={`ui-steprail-stop ui-steprail-${s.state}`}>
          <span className="ui-steprail-dot" aria-hidden="true" />
          <span className="ui-steprail-label">{s.label}</span>
        </div>
      ))}
    </div>
  );
}
