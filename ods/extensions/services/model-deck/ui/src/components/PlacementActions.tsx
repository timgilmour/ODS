import { useEffect, useState } from "react";
import {
  ApiError,
  bytesToGB,
  postAction,
  truncateMiddle,
  type ModelFile,
  type StorageUnit,
  type World,
} from "../api";
import { isArmedFor } from "../model/armed";
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

/** Controls for one declared LOCAL RESOURCE, plus the guard banners its
 * actions can raise. Keyed by resource (E1: no longer by a fixed tenant
 * name) because an UNLOADED lemonade-kind resource still needs its Load
 * control and has no placement.
 *
 * Branches on `world.tenants[resource].engine` (the resource's declared
 * KIND), never on `resource` itself — `KNOWN_KINDS`
 * (app/engine_kinds.py:90-94) is the closed backend enum this mirrors, the
 * same posture `nodes.ts`'s `tenantPlacement` and `setDraft.ts`'s
 * `KIND_DRAFT_SPEC` take. Every verb dispatches through Task 7's generic
 * `POST /api/tenants/{resource}/{verb}` route, so a resource can be named
 * anything and still hit the right kind's handler.
 *
 * Every action optimistic-disables while in flight, surfaces the response's
 * `detail`, and refetches either way. Two guards get inline offers rather
 * than a dead end: a park 409 (a chat is in flight or was recently active)
 * arms Force park, and a lemonade-kind load against a cold model 409s with
 * `pull=true`, which arms a "Pull + load" confirm. */
export default function PlacementActions({
  resource,
  world,
  models,
  hasPlacement,
  coldGgufs,
  onRefresh,
}: {
  resource: string;
  world: World;
  models: ModelFile[];
  /** Whether this resource already has a chip on the card above. When it
   * does not — a parked hipfire-kind resource, an unloaded lemonade-kind
   * one — this control row is the only thing on the panel naming the
   * resource, so the state has to come with the name instead of being
   * implied by which buttons are enabled. */
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
  // lemonade-kind load): a 409 whose detail contains "pull=true" arms this
  // banner instead of the plain error one; confirming retries with
  // ?pull=true. pullingModel tracks the in-flight pull-then-load by bare
  // model name until a later poll reports it loaded (see the effect below).
  const [pullOffer, setPullOffer] = useState<{ model: string; sizeBytes: number } | null>(null);
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  // Identity of the refusal currently on screen. It increments on EVERY
  // refusal, including a retry that produces the same message, which is what
  // makes it usable as an identity rather than the message text.
  const [refusalSeq, setRefusalSeq] = useState(0);
  // Which refusal the operator armed Force against — null if none. `armed`
  // is then a pure comparison of two numbers (model/armed.ts), so a stale
  // arming cannot survive into a refusal nobody clicked.
  const [armedForSeq, setArmedForSeq] = useState<number | null>(null);

  async function runAction(
    action: () => Promise<unknown>,
    opts?: {
      parkGuard?: boolean;
      pullGuard?: { model: string; sizeBytes: number };
      // Set by the plain Load/Unload handlers only — NOT by the "Pull +
      // load" confirm's own retry, which sets pullingModel from inside its
      // action and must not have this wipe it out immediately after. Any
      // *other* successful lemonade action means whatever pullingModel was
      // tracking is stale (superseded by a fresh load, or the resource was
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

  const tenant = world.tenants[resource];
  const kind = tenant.engine;

  // ComfyUI-kind holds VRAM until it is idle AND its queue has drained;
  // Free refuses otherwise, so the button says so up front.
  const comfyuiBlocked = tenant.state === "busy" || (tenant.queue ?? 0) > 0;

  // One-shot consumption of the pull-tracking token: the moment a poll sees
  // the pulled model actually loaded, clear it, so a LATER unload (or
  // loading something else) can never resurrect the chip from a stale
  // value. The banner below is a dumb `!= null` check; clearing is entirely
  // this effect's (plus runAction's clearPulling, plus the banner's own
  // dismiss button) job. Only ever armed for a lemonade-kind resource (the
  // Load dropdown below is the only thing that sets it), so this is a
  // harmless no-op for any other kind.
  useEffect(() => {
    if (
      pullingModel != null &&
      tenant.state === "loaded" &&
      tenant.model?.endsWith(pullingModel)
    ) {
      setPullingModel(null);
    }
  }, [tenant.state, tenant.model, pullingModel]);

  return (
    <div className="tenant-actions">
      {/* Names the resource. When it has no chip above (parked hipfire-kind,
          unloaded lemonade-kind) this is the only thing saying which
          resource it is and what it is doing. */}
      <span className="tenant-name">{resource}</span>
      {!hasPlacement && (
        <span className={`ui-pill ui-pill-${STATE_TONE[tenant.state] ?? "off"}`}>{tenant.state}</span>
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
          armed={isArmedFor(armedForSeq, refusalSeq)}
          onArm={() => setArmedForSeq(refusalSeq)}
          onConfirm={() => runAction(() => postAction(`/tenants/${resource}/park?force=true`))}
        />
      )}

      {pullOffer && (
        <Banner
          message={messages.modelIsCold(bytesToGB(pullOffer.sizeBytes))}
          onAction={() =>
            runAction(async () => {
              const res = (await postAction(`/tenants/${resource}/load?pull=true`, {
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

      {kind === "lemonade" && (
        <>
          <select
            aria-label={labels.modelToLoad}
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={busy || (models.length === 0 && coldGgufs.length === 0)}
          >
            <option value="">
              {models.length === 0 && coldGgufs.length === 0
                ? labels.noModels
                : labels.selectModel}
            </option>
            {models.map((m) => (
              <option key={m.file} value={m.file}>
                {m.file}
              </option>
            ))}
            {coldGgufs.length > 0 && (
              <optgroup label={labels.coldGroup}>
                {coldGgufs.map((u) => (
                  <option key={u.id} value={u.name}>
                    {labels.coldOption(truncateMiddle(u.name), bytesToGB(u.size))}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          <button
            disabled={busy || !selectedModel}
            onClick={() => {
              const coldUnit = coldGgufs.find((u) => u.name === selectedModel);
              runAction(() => postAction(`/tenants/${resource}/load`, { model: selectedModel }), {
                pullGuard: coldUnit
                  ? { model: selectedModel, sizeBytes: coldUnit.size }
                  : undefined,
                clearPulling: true,
              });
            }}
          >
            {labels.load}
          </button>
          <button
            disabled={busy || tenant.state !== "loaded"}
            onClick={() =>
              runAction(() => postAction(`/tenants/${resource}/unload`, {}), { clearPulling: true })
            }
          >
            {labels.unload}
          </button>
        </>
      )}

      {kind === "comfyui" && (
        <button
          disabled={busy || comfyuiBlocked}
          // Tooltip tracks the SAME condition as `disabled`: a non-empty
          // queue disables Free while the resource still reads "idle", so
          // narrowing this to state === "busy" would leave that case a
          // greyed-out button with no explanation.
          title={comfyuiBlocked ? labels.comfyuiBlockedTitle : undefined}
          onClick={() => runAction(() => postAction(`/tenants/${resource}/free`))}
        >
          {labels.free}
        </button>
      )}

      {kind === "hipfire" && (
        <>
          <button
            disabled={busy || tenant.state === "parked"}
            onClick={() => runAction(() => postAction(`/tenants/${resource}/park`), { parkGuard: true })}
          >
            {labels.park}
          </button>
          <button
            disabled={busy || tenant.state === "running"}
            onClick={() => runAction(() => postAction(`/tenants/${resource}/resume`))}
          >
            {labels.resume}
          </button>
        </>
      )}
    </div>
  );
}
