import { useState } from "react";
import { getFacts, sparkReload, type SettingsDrift } from "../api";
import { partitionDrift } from "../model/driftView";
import { humanizeAge, labels, messages } from "../model/messages";
import { SPARK_SLOT_KEY, type Placement } from "../model/nodes";
import { settingsIdentityFor } from "../model/settingsView";
import Banner from "../ui/Banner";
import type { SettingsTarget } from "./SettingsModal";

/**
 * The board's settings-drift notice: declared settings for this placement
 * were written more recently than it last (re)launched
 * (app/routers/__init__.py's `_settings_drift`, keyed off intent's
 * `updated_ts` — see its own docstring for why that baseline and not
 * `last_healthy_ts`). BLUE treatment deliberately, not the model detail
 * drawer's amber facts-drift table: build-design decision 5 makes this a
 * decision pending, not a disagreement the deck is flagging as wrong —
 * nothing here is broken, an operator simply has not reloaded to pick up
 * what they already declared.
 *
 * Rendered by ResourcePanel directly beneath the chip whose
 * `placement.settingsDrift` is set.
 */
export default function DriftCard({
  placement,
  drift,
  nodeId,
  settingsEngine,
  stale,
  onOpenSettings,
  onRefresh,
}: {
  placement: Placement;
  drift: SettingsDrift;
  nodeId: string;
  /** The node's configurable engine (App's catalog probe: presence of a
   * harvested option catalog IS the configurability signal), or `null` when
   * it has none — the same gate ModelDetailDrawer's Settings button uses. */
  settingsEngine: string | null;
  /** Owning node unreachable: the card still shows (declared settings really
   * did change since the last launch — that fact does not go stale), but no
   * verb can currently reach anything, matching every other control on this
   * panel (ResourcePanel's `{!stale && ...}` guard). */
  stale: boolean;
  onOpenSettings: (target: SettingsTarget) => void;
  onRefresh: () => void;
}) {
  const [reloadBusy, setReloadBusy] = useState(false);
  const [reloadError, setReloadError] = useState<string | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);

  // Rows AND names, never one or the other: a report can carry both at once
  // (see partitionDrift), and the header's count is over `changed`, so a name
  // dropped here is a key the card claims to be reporting and then does not
  // show.
  const { rows, legacy } = partitionDrift(drift.changed, drift.entries);
  const age = humanizeAge(drift.since);
  // Spark is a single-slot node whose lifecycle key is fixed backend-side
  // (app/observe.py SPARK_SLOT_KEY, mirrored in nodes.ts) and the one
  // placement with a reload verb (app/routers/spark.py:97) — every other
  // engine has no such action wired today.
  const isSparkSlot = placement.id === SPARK_SLOT_KEY;

  async function handleReload() {
    setReloadBusy(true);
    try {
      await sparkReload();
      setReloadError(null);
    } catch (err) {
      setReloadError(err instanceof Error ? err.message : String(err));
    } finally {
      setReloadBusy(false);
      onRefresh();
    }
  }

  /** Facts hold the profile->identity translation Settings needs
   * (settingsIdentityFor) and App does not keep them loaded globally, so
   * this fetches once, on click, rather than polling them the way the
   * drawer does — no standing cost for a card that is usually not even on
   * screen. On failure the banner shows and Settings never opens: opening it
   * on the untranslated placement name is the exact D11 defect
   * settingsIdentityFor's docstring names, and a silent wrong target is
   * worse than a refusal. */
  async function handleReview() {
    if (!settingsEngine) return;
    setReviewBusy(true);
    try {
      const facts = await getFacts();
      setReviewError(null);
      onOpenSettings({
        node: nodeId,
        engine: settingsEngine,
        model: settingsIdentityFor(facts, nodeId, settingsEngine, placement.name),
      });
    } catch (err) {
      setReviewError(err instanceof Error ? err.message : String(err));
    } finally {
      setReviewBusy(false);
    }
  }

  return (
    <div className="drift-card">
      <div className="drift-card-head">
        <span className="drift-card-title">{labels.settingsDrift}</span>
        <span className="drift-card-count">{labels.keysChanged(drift.changed.length)}</span>
        {age && <span className="drift-card-age">{age}</span>}
      </div>

      {reloadError && (
        <Banner
          message={messages.reloadFailed(reloadError)}
          onDismiss={() => setReloadError(null)}
        />
      )}
      {reviewError && (
        <Banner
          message={messages.factsLoadFailed(reviewError)}
          onDismiss={() => setReviewError(null)}
        />
      )}

      {rows.length > 0 && (
        <div className="drift-rows">
          {rows.map((row) => (
            <div className="drift-row" key={row.key}>
              <span className="drift-key">{row.displayKey}</span>
              <span className="drift-old">{row.oldText ?? labels.addedValue}</span>
              <span className="drift-arrow" aria-hidden="true">
                →
              </span>
              {row.newText === null ? (
                <span className="drift-value-removed">{labels.removedValue}</span>
              ) : (
                <span className="drift-new">{row.newText}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Legacy (pre-journal) path: app/routers/__init__.py's
          `_settings_drift` reports every current key of a touched namespace
          with no per-key history to diff — names only, never a fabricated
          old/new pair the server never sent. Beneath the rows rather than
          instead of them: the choice is made per namespace backend-side, so
          one report can hold both. */}
      {legacy.length > 0 && (
        <ul className="drift-legacy-list">
          {legacy.map((name) => (
            <li key={name.key}>{name.displayKey}</li>
          ))}
        </ul>
      )}

      <div className="settings-note">{labels.notAppliedUntilReload}</div>

      {!stale && (isSparkSlot || settingsEngine) && (
        <div className="drift-card-actions">
          {isSparkSlot && (
            <button
              type="button"
              disabled={reloadBusy}
              title={labels.reloadTitle}
              onClick={handleReload}
            >
              {reloadBusy ? labels.reloading : labels.reloadToApply}
            </button>
          )}
          {settingsEngine && (
            <button type="button" disabled={reviewBusy} onClick={handleReview}>
              {reviewBusy ? labels.openingSettings : labels.reviewInSettings}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
