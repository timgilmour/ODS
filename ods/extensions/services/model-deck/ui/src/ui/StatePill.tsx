import type { LifecycleStatus } from "../api";
import { messages } from "../model/messages";

/** One visual language for all ten lifecycle statuses.
 *
 * Colour follows decision 5: green healthy, blue neutral/in-progress, amber
 * wants a decision, red failed. `drifted` is blue and NOT amber — its own
 * copy says nothing is applied yet, so nothing is wrong. */
const TONE: Record<LifecycleStatus, string> = {
  serving: "good",
  idle: "off",
  warming: "busy",
  drifted: "busy",
  parked: "off",
  unmanaged: "off",
  unexpected: "warn",
  down: "bad",
  unreachable: "bad",
  quarantined: "bad",
};

export default function StatePill({
  status,
  stale = false,
}: {
  status: LifecycleStatus;
  stale?: boolean;
}) {
  // A stale reading is grey whatever it says: it is a memory, not a fact.
  const tone = stale ? "off" : TONE[status];
  return (
    <span className={`ui-pill ui-pill-${tone}`}>
      {stale ? messages.lastKnown().title : status}
    </span>
  );
}
