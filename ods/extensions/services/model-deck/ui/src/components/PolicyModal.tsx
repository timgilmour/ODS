import { useState } from "react";
import {
  putPolicy,
  putStoragePolicy,
  updateLocation,
  type PolicyMap,
  type StorageState,
  type TenantName,
} from "../api";
import { labels, messages } from "../model/messages";
import Banner from "../ui/Banner";
import Modal from "../ui/Modal";

interface PolicyModalProps {
  policy: PolicyMap;
  /** null when storage is down/unconfigured — the whole Storage section is
   * omitted rather than shown disabled, matching StorageView's own
   * loading/absent handling. */
  storageState: StorageState | null;
  onClose: () => void;
  onSaved: () => void;
}

// Priorities assigned top-to-bottom on Save, by final row order — reordering
// via Up/Down never lets a rank gap or duplicate happen, since it's always
// re-derived from row position rather than edited directly.
const RANKS = [100, 50, 40];

/** Tenant policy table: reorder (Up/Down, not drag — simpler and
 * keyboard-accessible) assigns descending priority; pinned + idle_ttl are
 * edited in place per row. Save PUTs the full three-tenant mapping and
 * closes; Cancel discards every local edit (nothing is written until
 * Save). */
export default function PolicyModal({ policy, storageState, onClose, onSaved }: PolicyModalProps) {
  const [order, setOrder] = useState<TenantName[]>(
    () =>
      (Object.keys(policy) as TenantName[]).sort(
        (a, b) => policy[b].priority - policy[a].priority,
      ),
  );
  const [pinned, setPinned] = useState<Record<TenantName, boolean>>(() => {
    const out = {} as Record<TenantName, boolean>;
    for (const t of order) out[t] = policy[t].pinned;
    return out;
  });
  const [idleTtl, setIdleTtl] = useState<Record<TenantName, number>>(() => {
    const out = {} as Record<TenantName, number>;
    for (const t of order) out[t] = policy[t].idle_ttl;
    return out;
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Storage section — omitted entirely when storageState is null (see
  // JSX below), but hooks still run unconditionally per rules-of-hooks;
  // these all just default to empty when there's nothing to edit.
  // props are stable while this modal is open: App.tsx pauses its poll
  // for the modal's whole lifetime (see App.tsx's `if (modalOpen ||
  // policyModalOpen) return`), so diffing local edits against
  // storageState directly on Save (no separate "initial" snapshot) is safe.
  const hotLocations = storageState?.locations.filter((l) => l.role === "hot") ?? [];
  const coldLocations = storageState?.locations.filter((l) => l.role === "cold") ?? [];
  const [autoTiering, setAutoTiering] = useState<boolean>(() => storageState?.policy.auto ?? false);
  const [watermarkInputs, setWatermarkInputs] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const loc of hotLocations) out[loc.name] = loc.watermark_gb != null ? String(loc.watermark_gb) : "";
    return out;
  });
  const [archiveInputs, setArchiveInputs] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const loc of hotLocations) out[loc.name] = loc.archive_to ?? "";
    return out;
  });

  function moveUp(i: number) {
    if (i === 0) return;
    setOrder((o) => {
      const next = [...o];
      [next[i - 1], next[i]] = [next[i], next[i - 1]];
      return next;
    });
  }

  function moveDown(i: number) {
    if (i === order.length - 1) return;
    setOrder((o) => {
      const next = [...o];
      [next[i], next[i + 1]] = [next[i + 1], next[i]];
      return next;
    });
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    const payload = {} as PolicyMap;
    order.forEach((tenant, i) => {
      payload[tenant] = { priority: RANKS[i], pinned: pinned[tenant], idle_ttl: idleTtl[tenant] };
    });
    try {
      await putPolicy(payload);

      if (storageState) {
        if (autoTiering !== storageState.policy.auto) {
          await putStoragePolicy({ auto: autoTiering });
        }
        const parseWatermark = (raw: string | undefined): number | null => {
          const trimmed = raw?.trim();
          if (!trimmed) return null;
          const n = Number(trimmed);
          return Number.isFinite(n) ? n : null;
        };
        const changedRows = hotLocations.filter((loc) => {
          const nextWatermark = parseWatermark(watermarkInputs[loc.name]);
          const nextArchive = archiveInputs[loc.name] || null;
          return nextWatermark !== loc.watermark_gb || nextArchive !== loc.archive_to;
        });
        await Promise.all(
          changedRows.map((loc) =>
            updateLocation(loc.name, {
              watermark_gb: parseWatermark(watermarkInputs[loc.name]),
              archive_to: archiveInputs[loc.name] || null,
            }),
          ),
        );
      }

      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      title={labels.policyTitle}
      onClose={onClose}
      footer={
        <>
          <button type="button" onClick={onClose} disabled={saving}>
            {labels.cancel}
          </button>
          <button type="button" className="primary" onClick={handleSave} disabled={saving}>
            {saving ? labels.saving : labels.save}
          </button>
        </>
      }
    >
      {error && (
        <Banner
          message={messages.guardRefused(error)}
          onDismiss={() => setError(null)}
        />
      )}

      <table className="policy-table">
        <thead>
          <tr>
            <th>rank</th>
            <th>tenant</th>
            <th>pinned</th>
            <th>idle TTL (s)</th>
            <th>reorder</th>
          </tr>
        </thead>
        <tbody>
          {order.map((tenant, i) => (
            <tr key={tenant}>
              <td>P{RANKS[i]}</td>
              <td className="tenant-name">{tenant}</td>
              <td>
                <input
                  type="checkbox"
                  aria-label={`${tenant} pinned`}
                  checked={pinned[tenant]}
                  onChange={(e) =>
                    setPinned((p) => ({ ...p, [tenant]: e.target.checked }))
                  }
                />
              </td>
              <td>
                <input
                  type="number"
                  min={0}
                  step={1}
                  aria-label={`${tenant} idle ttl seconds`}
                  value={idleTtl[tenant]}
                  onChange={(e) =>
                    setIdleTtl((p) => ({
                      ...p,
                      // Whole seconds only — fractional input causes an
                      // avoidable server 422.
                      [tenant]: Math.max(0, Math.round(Number(e.target.value) || 0)),
                    }))
                  }
                />
              </td>
              <td className="policy-reorder">
                <button
                  type="button"
                  aria-label={`move ${tenant} up`}
                  onClick={() => moveUp(i)}
                  disabled={i === 0}
                >
                  ↑
                </button>
                <button
                  type="button"
                  aria-label={`move ${tenant} down`}
                  onClick={() => moveDown(i)}
                  disabled={i === order.length - 1}
                >
                  ↓
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {storageState && (
        <div className="policy-storage-section">
          <h4>{labels.storageSection}</h4>
          <label>
            <input
              type="checkbox"
              checked={autoTiering}
              onChange={(e) => setAutoTiering(e.target.checked)}
            />{" "}
            {labels.autoTiering}
          </label>

          {hotLocations.length > 0 && (
            <table className="policy-table">
              <thead>
                <tr>
                  <th>hot location</th>
                  <th>watermark (GB)</th>
                  <th>archive to</th>
                </tr>
              </thead>
              <tbody>
                {hotLocations.map((loc) => (
                  <tr key={loc.name}>
                    <td className="tenant-name">{loc.name}</td>
                    <td>
                      <input
                        type="number"
                        min={0}
                        step={1}
                        placeholder="disabled"
                        aria-label={`${loc.name} watermark GB`}
                        value={watermarkInputs[loc.name] ?? ""}
                        onChange={(e) =>
                          setWatermarkInputs((w) => ({ ...w, [loc.name]: e.target.value }))
                        }
                      />
                    </td>
                    <td>
                      <select
                        aria-label={`${loc.name} archive to`}
                        value={archiveInputs[loc.name] ?? ""}
                        onChange={(e) =>
                          setArchiveInputs((a) => ({ ...a, [loc.name]: e.target.value }))
                        }
                      >
                        <option value="">—</option>
                        {coldLocations.map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </Modal>
  );
}
