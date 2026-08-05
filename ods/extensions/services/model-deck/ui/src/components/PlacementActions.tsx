import { useEffect, useState } from "react";
import {
  ApiError,
  bytesToGB,
  postAction,
  truncateMiddle,
  type ModelFile,
  type StorageUnit,
  type TenantName,
  type World,
} from "../api";
import { messages, labels } from "../model/messages";
import Banner from "../ui/Banner";
import ArmedButton from "../ui/ArmedButton";

/** Tenant states are the engine's OWN vocabulary, not LifecycleStatus, so
 * StatePill (typed to the latter) cannot render them — but they take the
 * same four tones, so the two can never disagree about what green means. */
const STATE_TONE: Record<string, string> = {
  loaded: "good",
  running: "good",
  loading: "busy",
  busy: "busy",
  unloaded: "off",
  parked: "off",
  idle: "off",
  unknown: "bad",
};

/** Controls for one tenant on one resource, plus the guard banners its
 * actions can raise. Keyed by tenant rather than by placement, because an
 * UNLOADED tenant still needs its Load control and has no placement.
 *
 * Every action optimistic-disables while in flight, surfaces the response's
 * `detail`, and refetches either way. Two guards get inline offers rather
 * than a dead end: hipfire's park 409 (a chat is in flight or was recently
 * active) arms Force park, and a lemonade load against a cold model 409s
 * with `pull=true`, which arms a "Pull + load" confirm. */
export default function PlacementActions({
  tenant,
  world,
  models,
  hasPlacement,
  coldGgufs,
  onRefresh,
}: {
  tenant: TenantName;
  world: World;
  models: ModelFile[];
  /** Whether this tenant already has a chip on the resource above. When it
   * does not — a parked hipfire, an unloaded lemonade — this control row is
   * the only thing on the panel naming the tenant, so the state has to come
   * with the name instead of being implied by which buttons are enabled. */
  hasPlacement: boolean;
  /** Resident-but-cold GGUFs (App.tsx's coldGgufs) — surfaced as a separate
   * optgroup in the Load dropdown; empty array (never undefined) so these
   * controls render the same whether storage is configured or not. */
  coldGgufs: StorageUnit[];
  onRefresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offerForcePark, setOfferForcePark] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  // Pull-through idiom (Force-park's armed-confirm, applied to a cold
  // lemonade load): a 409 whose detail contains "pull=true" arms this
  // banner instead of the plain error one; confirming retries with
  // ?pull=true. pullingModel tracks the in-flight pull-then-load by bare
  // model name until a later poll reports it loaded (see the effect below).
  const [pullOffer, setPullOffer] = useState<{ model: string; sizeBytes: number } | null>(null);
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  // Token to disarm ArmedButton on every new refusal, even if the error
  // message stays the same (e.g. retrying a guard that still 409s)
  const [refusalSeq, setRefusalSeq] = useState(0);

  async function runAction(
    action: () => Promise<unknown>,
    opts?: {
      parkGuard?: boolean;
      pullGuard?: { model: string; sizeBytes: number };
      // Set by the plain Load/Unload handlers only — NOT by the "Pull +
      // load" confirm's own retry, which sets pullingModel from inside its
      // action and must not have this wipe it out immediately after. Any
      // *other* successful lemonade action means whatever pullingModel was
      // tracking is stale (superseded by a fresh load, or the tenant was
      // unloaded out from under it), so it's cleared here rather than left
      // for the pulling banner to reason about.
      clearPulling?: boolean;
    },
  ) {
    setBusy(true);
    try {
      await action();
      setError(null);
      setOfferForcePark(false);
      setPullOffer(null);
      if (opts?.clearPulling) setPullingModel(null);
    } catch (err) {
      const isPullGuard =
        Boolean(opts?.pullGuard) &&
        err instanceof ApiError &&
        err.status === 409 &&
        err.message.includes("pull=true");
      setPullOffer(isPullGuard ? (opts!.pullGuard as { model: string; sizeBytes: number }) : null);
      setError(isPullGuard ? null : err instanceof Error ? err.message : String(err));
      setOfferForcePark(Boolean(opts?.parkGuard) && err instanceof ApiError && err.status === 409);
      // Bump refusal sequence to disarm any armed Force button on retry
      setRefusalSeq((n) => n + 1);
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

  // lemonade reports its loaded model with an internal "extra." namespace
  // prefix (e.g. "extra.foo.gguf"; see LemonadeClient.status() docstring),
  // so endsWith() matches without knowing the exact prefix string.
  const lemonade = world.tenants.lemonade;

  // ComfyUI holds VRAM until it is idle AND its queue has drained; Free
  // refuses otherwise, so the button says so up front.
  const comfyuiBlocked =
    world.tenants.comfyui.state === "busy" || (world.tenants.comfyui.queue ?? 0) > 0;

  // One-shot consumption of the pull-tracking token: the moment a poll sees
  // the pulled model actually loaded, clear it, so a LATER unload (or
  // loading something else) can never resurrect the chip from a stale
  // value. The banner below is a dumb `!= null` check; clearing is entirely
  // this effect's (plus runAction's clearPulling, plus the banner's own
  // dismiss button) job.
  useEffect(() => {
    if (
      pullingModel != null &&
      lemonade.state === "loaded" &&
      lemonade.model?.endsWith(pullingModel)
    ) {
      setPullingModel(null);
    }
  }, [lemonade.state, lemonade.model, pullingModel]);

  const state = world.tenants[tenant].state;

  return (
    <div className="tenant-actions">
      {/* Names the group. On a GPU hosting two tenants these would otherwise
          be two anonymous rows of buttons; and when the tenant has no chip
          above (parked hipfire, unloaded lemonade) this is the only thing
          saying which tenant it is and what it is doing. */}
      <span className="tenant-name">{tenant}</span>
      {!hasPlacement && (
        <span className={`ui-pill ui-pill-${STATE_TONE[state] ?? "off"}`}>{state}</span>
      )}

      {error && (
        <Banner
          message={messages.guardRefused(error)}
          onDismiss={() => setError(null)}
        />
      )}

      {/* Gated on `error` as well as the offer: in the card this moved from,
          Force park was a CHILD of the refusal banner, so dismissing the
          refusal took the override with it. As siblings they would come
          apart — dismiss the banner and a bare "Force park" button sits
          there with nothing left on screen saying what it overrides. */}
      {error && offerForcePark && (
        <ArmedButton
          label={labels.forcePark}
          disabled={busy}
          resetToken={refusalSeq}
          onConfirm={() => runAction(() => postAction("/tenants/hipfire/park?force=true"))}
        />
      )}

      {pullOffer && (
        <Banner
          message={messages.modelIsCold(bytesToGB(pullOffer.sizeBytes))}
          onAction={() =>
            runAction(async () => {
              const res = (await postAction("/tenants/lemonade/load?pull=true", {
                model: pullOffer.model,
              })) as { status?: string };
              if (res.status === "pulling") setPullingModel(pullOffer.model);
            })
          }
          onDismiss={() => setPullOffer(null)}
        />
      )}

      {pullingModel != null && (
        <Banner
          message={messages.pullingFromCold()}
          // Covers the failed/cancelled-job case: on_success never runs
          // server-side, so no poll result would ever clear the effect
          // above — this is the only way to unstick it then.
          onDismiss={() => setPullingModel(null)}
        />
      )}

      {tenant === "lemonade" && (
        <>
          <select
            aria-label="model to load"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={busy || (models.length === 0 && coldGgufs.length === 0)}
          >
            <option value="">
              {models.length === 0 && coldGgufs.length === 0 ? "no models found" : "select a model…"}
            </option>
            {models.map((m) => (
              <option key={m.file} value={m.file}>
                {m.file}
              </option>
            ))}
            {coldGgufs.length > 0 && (
              <optgroup label="❄ cold">
                {coldGgufs.map((u) => (
                  <option key={u.id} value={u.name}>
                    {`❄ ${truncateMiddle(u.name)} (${bytesToGB(u.size)} GB)`}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          <button
            disabled={busy || !selectedModel}
            onClick={() => {
              const coldUnit = coldGgufs.find((u) => u.name === selectedModel);
              runAction(() => postAction("/tenants/lemonade/load", { model: selectedModel }), {
                pullGuard: coldUnit
                  ? { model: selectedModel, sizeBytes: coldUnit.size }
                  : undefined,
                clearPulling: true,
              });
            }}
          >
            Load
          </button>
          <button
            disabled={busy || lemonade.state !== "loaded"}
            onClick={() =>
              runAction(() => postAction("/tenants/lemonade/unload", {}), { clearPulling: true })
            }
          >
            Unload
          </button>
        </>
      )}

      {tenant === "comfyui" && (
        <button
          disabled={busy || comfyuiBlocked}
          // Tooltip tracks the SAME condition as `disabled`: a non-empty
          // queue disables Free while the tenant still reads "idle", so
          // narrowing this to state === "busy" would leave that case a
          // greyed-out button with no explanation.
          title={comfyuiBlocked ? "ComfyUI is busy or has a non-empty queue" : undefined}
          onClick={() => runAction(() => postAction("/tenants/comfyui/free"))}
        >
          Free
        </button>
      )}

      {tenant === "hipfire" && (
        <>
          <button
            disabled={busy || world.tenants.hipfire.state === "parked"}
            onClick={() => runAction(() => postAction("/tenants/hipfire/park"), { parkGuard: true })}
          >
            Park
          </button>
          <button
            disabled={busy || world.tenants.hipfire.state === "running"}
            onClick={() => runAction(() => postAction("/tenants/hipfire/resume"))}
          >
            Resume
          </button>
        </>
      )}
    </div>
  );
}
