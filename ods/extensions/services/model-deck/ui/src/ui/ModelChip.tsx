import { bytesToGB, truncateMiddle } from "../api";
import type { Placement } from "../model/nodes";
import StatePill from "./StatePill";

const ICON: Record<Placement["kind"], string> = {
  model: "▣",
  engine: "▤",
  external: "▨",
};

/** A placement, as it appears on the board.
 *
 * The identity is truncated for layout only — `title` always carries the
 * verbatim checkpoint name, because that name IS the model's identity. */
export default function ModelChip({
  placement,
  onClick,
}: {
  placement: Placement;
  onClick?: () => void;
}) {
  const body = (
    <>
      <span aria-hidden="true">{ICON[placement.kind]}</span>
      <span className="ui-chip-name" title={placement.name}>
        {truncateMiddle(placement.name, 34)}
      </span>
      {placement.bytes != null && (
        <span className="ui-chip-size">{bytesToGB(placement.bytes)} GB</span>
      )}
      <StatePill status={placement.status} stale={placement.stale} />
    </>
  );

  if (!onClick) return <div className="ui-chip">{body}</div>;
  return (
    <button className="ui-chip" onClick={onClick}>
      {body}
    </button>
  );
}
