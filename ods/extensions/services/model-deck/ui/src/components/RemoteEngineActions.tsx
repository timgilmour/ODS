import { useState } from "react";
import { postEngineVerb } from "../api";
import { labels, messages, stateTone } from "../model/messages";
import type { RemoteEngineControl } from "../model/nodes";
import Banner from "../ui/Banner";

/** Controls for one DECLARED REMOTE engine — the off-box counterpart of
 * `PlacementActions`, and the only component that calls the Task 8 verb
 * route (`POST /api/nodes/{node_id}/engines/{resource}/{verb}`,
 * app/routers/serving.py:247-357).
 *
 * It decides nothing. WHICH verbs exist, and which of them would be a no-op
 * right now, are computed in `model/engineVerbs.ts` from the kind's own
 * `human_verbs` and carried here on `control` (`model/nodes.ts`'s
 * `RemoteEngineControl`) — so a kind gaining a verb, or an unreachable node
 * withdrawing all of them, is a model change with a unit test, never an edit
 * here. This file renders buttons, holds the in-flight flag, and shows what
 * the backend said.
 *
 * Nothing local is read: no `world`, no `models`, no `coldGgufs`. That is the
 * whole point of the seam (`model/nodes.ts`'s header) — App.tsx drills the
 * LOCAL box's snapshot down to every card, and this card belongs to another
 * machine.
 *
 * The verb is asynchronous end to end: the route answers 202 (accepted), the
 * node's agent queues the request for its host-side swap-helper, and a cold
 * sglang-omni start takes ~4 minutes (GF4). So a resolved click means
 * "asked", and the answer arrives as the next poll's lifecycle status —
 * `warming` while the boot is in flight, which is why nothing here claims an
 * outcome.
 */
export default function RemoteEngineActions({
  control,
  onRefresh,
}: {
  control: RemoteEngineControl;
  onRefresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(verb: string) {
    // PlacementActions' runAction shape, minus the two guard offers that
    // have no counterpart here (there is no force parameter on this route,
    // and no cold-model pull): optimistic-disable, surface the response's
    // own detail, refetch either way.
    setBusy(true);
    try {
      await postEngineVerb(control.nodeId, control.resource, verb);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

  return (
    <div className="tenant-actions">
      {/* The ENGINE's own word, beside the chip's lifecycle status: "busy"
          is the one thing that says a render is in flight RIGHT NOW, and
          unloading would end it. Rendered verbatim — the unavailable-busy
          -indicator case is already folded into it backend-side (design §4). */}
      <span className={`ui-pill ui-pill-${stateTone(control.state)}`}
            title={labels.engineStateTitle}>
        {control.state}
      </span>

      {error && (
        <Banner message={messages.guardRefused(error)} onDismiss={() => setError(null)} />
      )}

      {control.verbs.map((v) => (
        <button
          key={v.verb}
          type="button"
          disabled={busy || v.disabled}
          onClick={() => run(v.verb)}
        >
          {labels.engineVerb(v.verb)}
        </button>
      ))}
    </div>
  );
}
