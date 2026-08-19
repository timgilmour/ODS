import { useState } from "react";
import {
  putPolicy,
  putStoragePolicy,
  updateLocation,
  type EngineKindDef,
  type PolicyMap,
  type StorageState,
} from "../api";
import { demandFor, idleReleaseFor } from "../model/engineForm";
import { labels, messages } from "../model/messages";
import { parseWatermark } from "../model/watermark";
import Banner from "../ui/Banner";
import Modal from "../ui/Modal";

interface PolicyModalProps {
  policy: PolicyMap;
  /** null when storage is down/unconfigured — the whole Storage section is
   * omitted rather than shown disabled, matching StorageView's own
   * loading/absent handling. */
  storageState: StorageState | null;
  /** The kinds catalog App already fetched (App.tsx's `engineKinds`) — null
   * when that fetch failed, which renders as "unknown" rather than a
   * guessed consequence. */
  kinds: EngineKindDef[] | null;
  /** resource -> declared kind (model/engineForm.ts's resourceKindMap). */
  resourceKinds: Record<string, string>;
  onClose: () => void;
  onSaved: () => void;
}

// Priorities assigned top-to-bottom on Save, by final row order — reordering
// via Up/Down never lets a rank gap or duplicate happen, since it's always
// re-derived from row position rather than edited directly. E1: any number
// of declared resources now (not a fixed three), so this is a FORMULA, not
// a fixed list — still strictly descending, still gapped by 10 so a later
// manual policy.json edit between two ranks has room.
function rankFor(rowIndex: number): number {
  return 100 - rowIndex * 10;
}

/** Resource policy table: reorder (Up/Down, not drag — simpler and
 * keyboard-accessible) assigns descending priority; pinned + idle_ttl are
 * edited in place per row. Save PUTs the full resource->policy mapping and
 * closes; Cancel discards every local edit (nothing is written until
 * Save). */
export default function PolicyModal({ policy, storageState, kinds, resourceKinds, onClose, onSaved }: PolicyModalProps) {
  const [order, setOrder] = useState<string[]>(
    () => Object.keys(policy).sort((a, b) => policy[b].priority - policy[a].priority),
  );
  const [pinned, setPinned] = useState<Record<string, boolean>>(() => {
    const out: Record<string, boolean> = {};
    for (const t of order) out[t] = policy[t].pinned;
    return out;
  });
  const [idleTtl, setIdleTtl] = useState<Record<string, number>>(() => {
    const out: Record<string, number> = {};
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
      payload[tenant] = { priority: rankFor(i), pinned: pinned[tenant], idle_ttl: idleTtl[tenant] };
    });
    // REFUSE BEFORE ANY WRITE [max-review #15]. An unparseable watermark used
    // to coerce to null, and `watermark_gb: null` is a legal value meaning "no
    // watermark on this drive" — so a typo like "50 GB" silently DISABLED
    // auto-archiving and reported success. Checked ahead of putPolicy so a
    // refused save writes NOTHING: from the operator's view the modal either
    // applies or it doesn't, never half.
    const invalidWatermarks = hotLocations.filter(
      (loc) => parseWatermark(watermarkInputs[loc.name]) === "invalid",
    );
    if (invalidWatermarks.length > 0) {
      setError(messages.invalidWatermark(invalidWatermarks.map((l) => l.name)).body ?? "");
      setSaving(false);
      return;
    }

    try {
      await putPolicy(payload);

      if (storageState) {
        if (autoTiering !== storageState.policy.auto) {
          await putStoragePolicy({ auto: autoTiering });
        }
        const changedRows = hotLocations.filter((loc) => {
          const nextWatermark = parseWatermark(watermarkInputs[loc.name]);
          const nextArchive = archiveInputs[loc.name] || null;
          return nextWatermark !== loc.watermark_gb || nextArchive !== loc.archive_to;
        });
        await Promise.all(
          changedRows.map((loc) =>
            updateLocation(loc.name, {
              // The "invalid" arm is unreachable past the gate above; the
              // cast keeps that fact local rather than widening
              // updateLocation's contract to accept a sentinel.
              watermark_gb: parseWatermark(watermarkInputs[loc.name]) as number | null,
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
            <th>idle TTL (s, 0 = off)</th>
            <th>reorder</th>
          </tr>
        </thead>
        <tbody>
          {order.map((tenant, i) => (
            <tr key={tenant}>
              <td>P{rankFor(i)}</td>
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
                <div className="field-hint">
                  {messages.ttlConsequence(
                    idleTtl[tenant],
                    demandFor(kinds, resourceKinds[tenant] ?? ""),
                    idleReleaseFor(kinds, resourceKinds[tenant] ?? ""),
                  )}
                </div>
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
