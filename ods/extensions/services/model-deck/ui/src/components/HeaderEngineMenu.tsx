import { useState } from "react";
import { postEngineVerb } from "../api";
import { loadVerbFor } from "../model/engineVerbs";
import { labels, messages } from "../model/messages";
import type { RemoteEngineControl } from "../model/nodes";
import Banner from "../ui/Banner";

/**
 * Node-header "Load engine…" menu — the load path for a DECLARED REMOTE
 * engine once the 2026-08-18 ruling stopped rendering an engine with nothing
 * loaded anywhere on the board at all. Those engines live in
 * `DeckNode.hiddenEngines` (`model/nodes.ts`); this is the only surface that
 * can still reach one, because `RemoteEngineActions` only ever renders
 * beside a CHIP, and a hidden engine has none.
 *
 * ⚠ INTERIM SURFACE. Once Set Builder carries engine assignment (E2), a
 * declared-but-unloaded engine gets picked from there, the way a local
 * resource's Load dropdown already works — at which point this component
 * and the `hiddenEngines` plumbing that feeds it should come out entirely.
 * Design doc: ~/notes/designs/2026-08-18-model-deck-board-gpu-cards-design.md
 * ruling 4.
 *
 * `loadVerbFor` (model/engineVerbs.ts) decides WHICH entries get a button —
 * a kind with no load verb, or an entry whose catalog hasn't landed, offers
 * none — so this component only renders branches, never derives them.
 *
 * The verb is asynchronous end to end (see `RemoteEngineActions`'s doc): a
 * resolved click means "asked", not "loaded" — the outcome arrives as the
 * next poll's lifecycle `warming`, so nothing here claims more than that.
 */
export default function HeaderEngineMenu({
  hiddenEngines,
  onRefresh,
}: {
  hiddenEngines: RemoteEngineControl[] | undefined;
  onRefresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // flatMap rather than map+filter: `loadVerbFor` returning null (kind has
  // no load verb, or the catalog hasn't landed) drops the entry instead of
  // needing a separate type-narrowing filter step.
  const entries = (hiddenEngines ?? []).flatMap((control) => {
    const verb = loadVerbFor(control);
    return verb ? [{ control, verb }] : [];
  });

  if (entries.length === 0) return null;

  async function run(control: RemoteEngineControl) {
    // RemoteEngineActions' runAction shape: optimistic-disable, surface the
    // response's own detail, refetch either way.
    setBusy(true);
    try {
      await postEngineVerb(control.nodeId, control.resource, "load");
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      onRefresh();
    }
  }

  return (
    <div className="node-header-engine-menu">
      {error && (
        <Banner message={messages.guardRefused(error)} onDismiss={() => setError(null)} />
      )}
      {entries.map(({ control, verb }) => (
        <button
          key={control.resource}
          type="button"
          title={labels.loadEngineTitle(control.resource)}
          disabled={busy || verb.disabled}
          onClick={() => run(control)}
        >
          {labels.loadEngine(control.resource)}
        </button>
      ))}
    </div>
  );
}
