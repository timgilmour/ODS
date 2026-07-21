import { useState } from "react";
import { putPolicy, type PolicyMap, type TenantName } from "../api";

interface PolicyModalProps {
  policy: PolicyMap;
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
export default function PolicyModal({ policy, onClose, onSaved }: PolicyModalProps) {
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
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-box">
        <h3>Tenant policy</h3>

        {error && (
          <div className="banner-error">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="dismiss error">
              ×
            </button>
          </div>
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
                    aria-label={`move ${tenant} up`}
                    onClick={() => moveUp(i)}
                    disabled={i === 0}
                  >
                    ↑
                  </button>
                  <button
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

        <div className="modal-actions">
          <button onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button className="primary" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
