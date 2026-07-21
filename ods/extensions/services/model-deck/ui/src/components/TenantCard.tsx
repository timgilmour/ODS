import { useState } from "react";
import {
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
      token: string;
      models: ModelFile[];
      onRefresh: () => void;
    }
  | {
      name: "comfyui";
      data: ComfyuiTenant;
      policy: TenantPolicy;
      token: string;
      onRefresh: () => void;
    }
  | {
      name: "hipfire";
      data: HipfireTenant;
      policy: TenantPolicy;
      token: string;
      onRefresh: () => void;
    };

/** One tenant's live status + (when a token is set) its control buttons.
 * Every action optimistic-disables while in flight, surfaces the response's
 * `detail` in a dismissible banner on failure, and refetches state either way. */
export default function TenantCard(props: TenantCardProps) {
  const { data, policy, token, onRefresh } = props;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState("");

  async function runAction(action: () => Promise<unknown>) {
    setBusy(true);
    try {
      await action();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
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
          <button onClick={() => setError(null)} aria-label="dismiss error">
            ×
          </button>
        </div>
      )}

      {token && (
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
              onPark={() => runAction(() => postAction("/tenants/hipfire/park"))}
              onResume={() => runAction(() => postAction("/tenants/hipfire/resume"))}
            />
          )}
        </div>
      )}
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
