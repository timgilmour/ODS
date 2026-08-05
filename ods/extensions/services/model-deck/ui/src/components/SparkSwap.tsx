import { useState } from "react";
import { ApiError, sparkSwap, type SparkStatus } from "../api";
import { messages, labels } from "../model/messages";
import Banner from "../ui/Banner";
import ArmedButton from "../ui/ArmedButton";

/** Spark's profile picker. Swap can 409 three ways, and exactly one of them
 * gets a Force button:
 *
 * - **busy guard** (in-flight requests). Force works, and the override is
 *   offered inline — retrying later is the only alternative.
 * - **a previous swap still booting.** Force works here too: the
 *   boot-window guard in app/engines/spark.py:swap() is inside
 *   `if not endpoint_ok and not force`, so force skips it outright, and the
 *   guard's own message says "use force to interrupt it". The button is
 *   withheld anyway, deliberately — interrupting a boot discards the 5-15
 *   minutes of weight load and FlashInfer autotune already spent, so an
 *   operator who really means it is made to go to the API for it. This is
 *   friction on a working override, not a missing capability; do not "fix"
 *   it by arming the button.
 * - **the litellm default-route guard.** This is the one force genuinely
 *   cannot help with: it runs before the force check and says so itself. */
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
  // Token to disarm ArmedButton on every new refusal, even if the error
  // message stays the same (e.g. retrying a guard that still 409s)
  const [refusalSeq, setRefusalSeq] = useState(0);

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
      // Only the busy guard ARMS Force. A mid-swap 409 would yield to force
      // as well, but interrupting a boot throws its autotune away, so that
      // override deliberately lives at the API only; the litellm guard is
      // the one that force cannot help with at all. See the docstring.
      setOfferForce(err instanceof ApiError && err.status === 409 && msg.includes("in-flight"));
      // Bump refusal sequence to disarm any armed Force button on retry
      setRefusalSeq((n) => n + 1);
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
        <ArmedButton
          label={labels.forceSwap}
          disabled={busy}
          resetToken={refusalSeq}
          onConfirm={() => doSwap(true)}
        />
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
