import { useState } from "react";
import { ApiError, sparkSwap, type SparkStatus } from "../api";
import { messages } from "../model/messages";
import Banner from "../ui/Banner";

/** Spark's profile picker. Swap can 409 two ways: the busy guard (in-flight
 * requests — force helps) and an already-running swap (force will not help;
 * the message says so). The litellm default-route guard also 409s and is
 * deliberately not force-offerable — the backend ignores force for it. */
export default function SparkSwap({
  spark,
  onChanged,
}: {
  spark: SparkStatus;
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offerForce, setOfferForce] = useState(false);

  async function doSwap(force = false) {
    if (!selected) return;
    setBusy(true);
    try {
      await sparkSwap(selected, force);
      setError(null);
      setOfferForce(false);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      // Only the busy guard is force-retryable; a mid-swap 409 or the
      // litellm guard won't yield to force, so don't offer it for those.
      setOfferForce(err instanceof ApiError && err.status === 409 && msg.includes("in-flight"));
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  return (
    <div className="tenant-actions">
      {/* Names the group, like every tenant control row on the board. The
          node card above says which node; this says what on it these verbs
          drive — which stops being obvious the moment a node has more than
          one slot. */}
      <span className="tenant-name">spark</span>

      {error && (
        <Banner message={messages.guardRefused(error)} onDismiss={() => setError(null)} />
      )}
      {/* Gated on `error` too: in the card this moved from, Force swap was a
          CHILD of the refusal banner, so dismissing the refusal took the
          override with it. Left as independent siblings they come apart —
          dismiss the banner and a bare "Force swap" sits there with nothing
          on screen saying what it overrides. */}
      {error && offerForce && (
        <button disabled={busy} onClick={() => doSwap(true)}>
          Force swap
        </button>
      )}
      <select value={selected} onChange={(e) => setSelected(e.target.value)} disabled={busy}>
        <option value="">swap to…</option>
        {spark.profiles.map((p) => (
          <option key={p.name} value={p.name}>
            {p.engine !== "vllm" ? `${p.name} (${p.engine})` : p.name}
          </option>
        ))}
      </select>
      <button
        disabled={busy || !selected || spark.swap_status?.state === "swapping"}
        onClick={() => doSwap(false)}
      >
        Swap
      </button>
    </div>
  );
}
