import { useState } from "react";
import {
  ApiError,
  bytesToGB,
  postAction,
  truncateMiddle,
  type ComfyuiTenant,
  type HipfireTenant,
  type LemonadeTenant,
  type ModelFile,
  type TenantPolicy,
} from "../api";

type TenantCardProps =
  | {
      name: "lemonade";
      data: LemonadeTenant;
      policy: TenantPolicy;
      models: ModelFile[];
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

  async function runAction(action: () => Promise<unknown>, opts?: { parkGuard?: boolean }) {
    setBusy(true);
    try {
      await action();
      setError(null);
      setOfferForcePark(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setOfferForcePark(Boolean(opts?.parkGuard) && err instanceof ApiError && err.status === 409);
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

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

      <div className="tenant-actions">
        {props.name === "lemonade" && (
            <LemonadeActions
              data={props.data}
              models={props.models}
              busy={busy}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              onLoad={() =>
                runAction(() => postAction("/tenants/lemonade/load", { model: selectedModel }))
              }
              onUnload={() => runAction(() => postAction("/tenants/lemonade/unload", {}))}
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
  busy,
  selectedModel,
  onSelectModel,
  onLoad,
  onUnload,
}: {
  data: LemonadeTenant;
  models: ModelFile[];
  busy: boolean;
  selectedModel: string;
  onSelectModel: (file: string) => void;
  onLoad: () => void;
  onUnload: () => void;
}) {
  return (
    <>
      <select
        aria-label="model to load"
        value={selectedModel}
        onChange={(e) => onSelectModel(e.target.value)}
        disabled={busy || models.length === 0}
      >
        <option value="">
          {models.length === 0 ? "no models found" : "select a model…"}
        </option>
        {models.map((m) => (
          <option key={m.file} value={m.file}>
            {m.file}
          </option>
        ))}
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
