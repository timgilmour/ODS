import { useEffect, useState } from "react";
import {
  ApiError,
  bytesToGB,
  postAction,
  truncateMiddle,
  type ComfyuiTenant,
  type HipfireTenant,
  type LemonadeTenant,
  type ModelFile,
  type StorageUnit,
  type TenantPolicy,
} from "../api";

type TenantCardProps =
  | {
      name: "lemonade";
      data: LemonadeTenant;
      policy: TenantPolicy;
      models: ModelFile[];
      /** Resident-but-cold GGUFs (App.tsx's coldGgufs) — surfaced as a
       * separate optgroup in the Load dropdown; empty array (never
       * undefined) so this card renders the same whether storage is
       * configured or not. */
      coldGgufs: StorageUnit[];
      onRefresh: () => void;
    }
  | {
      name: "comfyui";
      data: ComfyuiTenant;
      policy: TenantPolicy;
      onRefresh: () => void;
    }
  | {
      name: "hipfire";
      data: HipfireTenant;
      policy: TenantPolicy;
      onRefresh: () => void;
    };

/** One tenant's live status + its control buttons (no auth — the admin
 * gate was removed 2026-07-22; every control is always available).
 * Every action optimistic-disables while in flight, surfaces the response's
 * `detail` in a dismissible banner on failure, and refetches state either way.
 *
 * hipfire's park can 409 off the conversation-guard (a chat is in flight
 * or was active within the activity window); that specific failure also
 * arms a "Force park" button in the banner so the override is one click,
 * not a curl command. */
export default function TenantCard(props: TenantCardProps) {
  const { data, policy, onRefresh } = props;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offerForcePark, setOfferForcePark] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  // Pull-through idiom (Force-park's armed-confirm, applied to a cold
  // lemonade load): a 409 whose detail contains "pull=true" arms this
  // banner instead of the plain error one; confirming retries with
  // ?pull=true. pullingModel tracks the in-flight pull-then-load by bare
  // model name until a later poll reports it loaded (see isPulling below).
  const [pullOffer, setPullOffer] = useState<{ model: string; sizeBytes: number } | null>(null);
  const [pullingModel, setPullingModel] = useState<string | null>(null);

  async function runAction(
    action: () => Promise<unknown>,
    opts?: {
      parkGuard?: boolean;
      pullGuard?: { model: string; sizeBytes: number };
      // Set by the plain Load/Unload handlers only — NOT by the "Pull +
      // load" confirm button's own retry, which sets pullingModel itself
      // from inside its action and must not have this wipe it out again
      // immediately after. Any *other* successful lemonade action means
      // whatever pullingModel was tracking is stale (superseded by a fresh
      // load, or the tenant was unloaded out from under it), so it's
      // cleared here rather than left for isPulling to reason about.
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
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

  // lemonade's "loaded" model comes back with lemonade's internal "extra."
  // namespace prefix (e.g. "extra.foo.gguf"; see LemonadeClient.status()
  // docstring) — endsWith() matches regardless of whether that prefix is
  // present, so this doesn't need to know the exact prefix string.
  const lemonadeState = props.name === "lemonade" ? props.data.state : undefined;
  const lemonadeModel = props.name === "lemonade" ? props.data.model : undefined;

  // One-shot consumption of the pull-tracking token: the moment a poll
  // observes the pulled model actually loaded, clear it here so a LATER
  // unload (or loading something else) can never make the chip resurrect
  // from a stale pullingModel — isPulling below is a dumb `!= null` check;
  // clearing is entirely this effect's (plus runAction's clearPulling,
  // plus the chip's own dismiss button) job.
  useEffect(() => {
    if (pullingModel != null && lemonadeState === "loaded" && lemonadeModel?.endsWith(pullingModel)) {
      setPullingModel(null);
    }
  }, [lemonadeState, lemonadeModel, pullingModel]);

  const isPulling = props.name === "lemonade" && pullingModel != null;

  return (
    <div className="tenant-card">
      <div className="tenant-card-head">
        <span className="tenant-name">{props.name}</span>
        <span className={`chip chip-${data.state}`}>{data.state}</span>
        {props.name === "hipfire" && (props.data.queue_depth ?? 0) > 0 && (
          <span
            className="chip chip-busy"
            title="a conversation turn is being served right now — park/apply will refuse without force"
          >
            in use
          </span>
        )}
        {isPulling && (
          <>
            <span className="chip chip-busy" title="pulling from cold storage to hot, then loading">
              pulling ❄ → 🔥
            </span>
            {/* Covers the failed/cancelled-job case: on_success never runs
                server-side, so no poll result would ever clear the effect
                above — this is the only way to unstick the chip then. */}
            <button onClick={() => setPullingModel(null)} aria-label="dismiss pulling status">
              ×
            </button>
          </>
        )}
      </div>

      <div className="tenant-meta">
        {props.name !== "comfyui" && props.data.model && (
          <span title={props.data.model}>{truncateMiddle(props.data.model)}</span>
        )}
        {props.name === "comfyui" && <span>queue {props.data.queue ?? "—"}</span>}
        {props.name !== "comfyui" && props.data.footprint != null && (
          <span>{bytesToGB(props.data.footprint)} GB</span>
        )}
        {props.name !== "hipfire" && props.data.idle_s != null && (
          <span>idle {Math.round(props.data.idle_s)} s</span>
        )}
        {policy.pinned && (
          <span className="badge-pinned" title="pinned (exempt from eviction)">
            📌
          </span>
        )}
        <span title="eviction priority">P{policy.priority}</span>
      </div>

      {error && (
        <div className="banner-error">
          <span>{error}</span>
          {offerForcePark && (
            <button
              onClick={() =>
                runAction(() => postAction("/tenants/hipfire/park?force=true"))
              }
              disabled={busy}
            >
              Force park
            </button>
          )}
          <button onClick={() => setError(null)} aria-label="dismiss error">
            ×
          </button>
        </div>
      )}

      {pullOffer && (
        <div className="banner-error">
          <span>
            model is cold — pull {bytesToGB(pullOffer.sizeBytes)} GB to hot storage then load?
          </span>
          <button
            onClick={() =>
              runAction(async () => {
                const res = (await postAction("/tenants/lemonade/load?pull=true", {
                  model: pullOffer.model,
                })) as { status?: string };
                if (res.status === "pulling") {
                  setPullingModel(pullOffer.model);
                }
              })
            }
            disabled={busy}
          >
            Pull + load
          </button>
          <button onClick={() => setPullOffer(null)} aria-label="dismiss">
            ×
          </button>
        </div>
      )}

      <div className="tenant-actions">
        {props.name === "lemonade" && (
            <LemonadeActions
              data={props.data}
              models={props.models}
              coldGgufs={props.coldGgufs}
              busy={busy}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              onLoad={() => {
                const coldUnit = props.coldGgufs.find((u) => u.name === selectedModel);
                runAction(
                  () => postAction("/tenants/lemonade/load", { model: selectedModel }),
                  {
                    pullGuard: coldUnit ? { model: selectedModel, sizeBytes: coldUnit.size } : undefined,
                    clearPulling: true,
                  },
                );
              }}
              onUnload={() =>
                runAction(() => postAction("/tenants/lemonade/unload", {}), { clearPulling: true })
              }
            />
          )}
          {props.name === "comfyui" && (
            <ComfyuiActions
              data={props.data}
              busy={busy}
              onFree={() => runAction(() => postAction("/tenants/comfyui/free"))}
            />
          )}
          {props.name === "hipfire" && (
            <HipfireActions
              data={props.data}
              busy={busy}
              onPark={() =>
                runAction(() => postAction("/tenants/hipfire/park"), { parkGuard: true })
              }
              onResume={() => runAction(() => postAction("/tenants/hipfire/resume"))}
            />
          )}
        </div>
    </div>
  );
}

function LemonadeActions({
  data,
  models,
  coldGgufs,
  busy,
  selectedModel,
  onSelectModel,
  onLoad,
  onUnload,
}: {
  data: LemonadeTenant;
  models: ModelFile[];
  coldGgufs: StorageUnit[];
  busy: boolean;
  selectedModel: string;
  onSelectModel: (file: string) => void;
  onLoad: () => void;
  onUnload: () => void;
}) {
  const empty = models.length === 0 && coldGgufs.length === 0;
  return (
    <>
      <select
        aria-label="model to load"
        value={selectedModel}
        onChange={(e) => onSelectModel(e.target.value)}
        disabled={busy || empty}
      >
        <option value="">{empty ? "no models found" : "select a model…"}</option>
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
      <button onClick={onLoad} disabled={busy || !selectedModel}>
        Load
      </button>
      <button onClick={onUnload} disabled={busy || data.state !== "loaded"}>
        Unload
      </button>
    </>
  );
}

function ComfyuiActions({
  data,
  busy,
  onFree,
}: {
  data: ComfyuiTenant;
  busy: boolean;
  onFree: () => void;
}) {
  const blocked = data.state === "busy" || (data.queue ?? 0) > 0;
  return (
    <button
      onClick={onFree}
      disabled={busy || blocked}
      title={blocked ? "ComfyUI is busy or has a non-empty queue" : undefined}
    >
      Free
    </button>
  );
}

function HipfireActions({
  data,
  busy,
  onPark,
  onResume,
}: {
  data: HipfireTenant;
  busy: boolean;
  onPark: () => void;
  onResume: () => void;
}) {
  return (
    <>
      <button onClick={onPark} disabled={busy || data.state === "parked"}>
        Park
      </button>
      <button onClick={onResume} disabled={busy || data.state === "running"}>
        Resume
      </button>
    </>
  );
}
