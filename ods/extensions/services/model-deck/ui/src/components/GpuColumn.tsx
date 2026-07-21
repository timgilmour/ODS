import { bytesToGB, meterFillClass, type Gpu, type ModelFile, type PolicyMap, type TenantName, type World } from "../api";
import TenantCard from "./TenantCard";

// Mirrors the backend's fixed engine->GPU placement (compose GPU device
// assignment / app/settings.py): hipfire is pinned to GPU 0; lemonade and
// comfyui share GPU 1. Not derived from any API field — the world snapshot
// carries no per-tenant GPU attribution (see app/state.py's externals-only
// GPU->pid mapping). SetBuilder.tsx hardcodes the same fixed placement for
// its (differently-shaped, drag/drop) GPU 0/1 columns rather than importing
// this — kept un-exported to avoid a needless coupling between the two
// components' very different rendering.
const GPU_TENANTS: Record<number, TenantName[]> = {
  0: ["hipfire"],
  1: ["lemonade", "comfyui"],
};

interface GpuColumnProps {
  gpu: Gpu;
  world: World;
  policy: PolicyMap;
  models: ModelFile[];
  token: string;
  onRefresh: () => void;
}

export default function GpuColumn({ gpu, world, policy, models, token, onRefresh }: GpuColumnProps) {
  const pct = gpu.total > 0 ? (gpu.used / gpu.total) * 100 : 0;
  const tenants = GPU_TENANTS[gpu.index] ?? [];
  const externals = world.externals.filter((e) => e.gpu === gpu.index);

  return (
    <div className="gpu-column">
      <h2>GPU {gpu.index}</h2>

      <div className="gpu-meter">
        <div className="meter-track">
          <div
            className={meterFillClass(pct)}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
        <div className="meter-label">
          {bytesToGB(gpu.used)} / {bytesToGB(gpu.total)} GB ({pct.toFixed(0)}%)
        </div>
      </div>

      {tenants.includes("hipfire") && (
        <TenantCard
          name="hipfire"
          data={world.tenants.hipfire}
          policy={policy.hipfire}
          token={token}
          onRefresh={onRefresh}
        />
      )}
      {tenants.includes("lemonade") && (
        <TenantCard
          name="lemonade"
          data={world.tenants.lemonade}
          policy={policy.lemonade}
          token={token}
          models={models}
          onRefresh={onRefresh}
        />
      )}
      {tenants.includes("comfyui") && (
        <TenantCard
          name="comfyui"
          data={world.tenants.comfyui}
          policy={policy.comfyui}
          token={token}
          onRefresh={onRefresh}
        />
      )}

      {externals.map((e) => (
        <div key={e.pid} className="external-row">
          external pid {e.pid} — {bytesToGB(e.bytes)} GB
        </div>
      ))}
    </div>
  );
}
