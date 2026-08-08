import type { Layer } from "../api";
import { labels } from "../model/messages";

/** The 6px provenance dot: which of the ladder's five layers a value's
 * winning entry came from (`Layer` in api.ts mirrors app/ladder.py:48's
 * LAYERS tuple).
 *
 * Colour is token-only and lives in deck.css as `.prov-<layer>`. The two
 * DERIVED layers deliberately share one grey: the distinction an operator
 * acts on is declared-vs-derived (only declared values are editable, and
 * only they are ever shipped as flags — app/routers/settings.py's
 * `_declared_only` rule), and WHICH derived layer it was is spelled out in
 * the title rather than encoded as a second colour nobody could decode.
 *
 * Colour is never the only carrier: the title/aria-label names the layer,
 * and the popover behind the dot repeats it as text.
 *
 * `label` overrides that name for the ONE surface that reuses this dot for
 * something that is not a ladder layer: the model detail drawer's facts
 * table, whose entries have two ORIGINS (derived/declared,
 * app/facts.py:resolve_facts) rather than five layers. It borrows the two
 * colours whose meaning already matches — dim grey for "the machine read
 * this", bright for "a human declared it" — and must not borrow the layer
 * sentence with them, because "harvested from the engine" is simply not
 * where a fact comes from. Absent, the name is the layer's, exactly as
 * before. */
export default function ProvenanceDot({ layer, label }: { layer: Layer; label?: string }) {
  const name = label ?? labels.layerName(layer);
  return (
    <span
      className={`prov-dot prov-${layer}`}
      role="img"
      aria-label={name}
      title={name}
    />
  );
}
