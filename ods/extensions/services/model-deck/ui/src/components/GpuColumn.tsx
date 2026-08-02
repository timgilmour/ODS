import { bytesToGB, meterFillClass, type Gpu, type ModelFile, type PolicyMap, type StorageUnit, type TenantName, type World } from "../api";
import TenantCard from "./TenantCard";

// Fixed display order; membership comes from the backend's placement map
// (World.snapshot "placement"), so the layout is data-driven per host.
const TENANT_ORDER: TenantName[] = ["hipfire", "lemonade", "comfyui"];

interface GpuColumnProps {
  gpu: Gpu;
  world: World;
  policy: PolicyMap;
  models: ModelFile[];
  /** Cold (non-hot-lemonade) resident GGUFs — only meaningful to the
   * lemonade card's Load dropdown, but threaded through here since GpuColumn
   * is the only place that knows which tenant column is lemonade. */
  coldGgufs: StorageUnit[];
  onRefresh: () => void;
}

export default function GpuColumn({ gpu, world, policy, models, coldGgufs, onRefresh }: GpuColumnProps) {
  const pct = gpu.total > 0 ? (gpu.used / gpu.total) * 100 : 0;
  const tenants = TENANT_ORDER.filter((t) => world.placement[t] === gpu.index);
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
          onRefresh={onRefresh}
        />
      )}
      {tenants.includes("lemonade") && (
        <TenantCard
          name="lemonade"
          data={world.tenants.lemonade}
          policy={policy.lemonade}
          models={models}
          coldGgufs={coldGgufs}
          onRefresh={onRefresh}
        />
      )}
      {tenants.includes("comfyui") && (
        <TenantCard
          name="comfyui"
          data={world.tenants.comfyui}
          policy={policy.comfyui}
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
