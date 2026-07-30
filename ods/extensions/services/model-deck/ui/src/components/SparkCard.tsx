import { useEffect, useState } from "react";
import {
  ApiError,
  getSparkStatus,
  sparkSwap,
  type SparkStatus,
} from "../api";

/** Remote Spark node card — single serving slot, profile picker, swap.
 *
 * Self-fetching (like EventLog): spark status lives on /api/spark/status,
 * not in the /api/state world snapshot, because the Spark is deliberately
 * outside VRAM arbitration. Renders nothing at all when the backend says
 * 503 (spark not configured on this deployment).
 *
 * Swap can 409 two ways: the busy guard (in-flight requests — force is
 * offered, mirroring hipfire's force-park) and an already-running swap
 * (force won't help; the banner just says wait). The litellm default-route
 * guard also 409s and is deliberately not force-offerable (the backend
 * ignores force for it anyway). First boot of a profile autotunes for up
 * to ~15 min — the card says so while the endpoint is down post-swap.
 */
export default function SparkCard({
  refreshTrigger,
  onChanged,
}: {
  refreshTrigger: number;
  onChanged: () => void;
}) {
  const [status, setStatus] = useState<SparkStatus | null>(null);
  const [configured, setConfigured] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offerForce, setOfferForce] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getSparkStatus()
      .then((s) => {
        if (cancelled) return;
        setConfigured(s !== null);
        setStatus(s);
        setFetchError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setFetchError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [refreshTrigger]);

  if (!configured) return null;

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
      setOfferForce(
        err instanceof ApiError &&
          err.status === 409 &&
          msg.includes("in-flight"),
      );
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  const serving = status?.serving;
  const swapStatus = status?.swap_status ?? null;
  const chip = serving?.endpoint_ok ? "running" : "loading";
  const bootingHint =
    swapStatus &&
    swapStatus.state !== "error" &&
    serving &&
    !serving.endpoint_ok;

  return (
    <div className="tenant-card">
      <div className="tenant-card-head">
        <span className="tenant-name">spark</span>
        <span className={`chip chip-${chip}`}>
          {serving?.endpoint_ok ? "serving" : "endpoint down"}
        </span>
      </div>

      <div className="tenant-meta">
        <div>
          model: <strong>{serving?.model ?? "—"}</strong>
        </div>
        {swapStatus && (
          <div>
            last swap: {swapStatus.profile} — {swapStatus.state}
            {swapStatus.state === "error" && ` (${swapStatus.message})`}
          </div>
        )}
        {bootingHint && (
          <div>vLLM is coming up — a profile's first boot can autotune for ~15 min</div>
        )}
        {fetchError && <div className="banner-error">{fetchError}</div>}
      </div>

      {error && (
        <div className="banner-error">
          {error}
          {offerForce && (
            <button disabled={busy} onClick={() => doSwap(true)}>
              Force swap
            </button>
          )}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="tenant-actions">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={busy}
        >
          <option value="">swap to…</option>
          {(status?.profiles ?? []).map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <button
          disabled={busy || !selected || swapStatus?.state === "swapping"}
          onClick={() => doSwap(false)}
        >
          Swap
        </button>
      </div>
    </div>
  );
}
