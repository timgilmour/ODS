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
 * and the popover behind the dot repeats it as text. */
export default function ProvenanceDot({ layer }: { layer: Layer }) {
  const name = labels.layerName(layer);
  return (
    <span
      className={`prov-dot prov-${layer}`}
      role="img"
      aria-label={name}
      title={name}
    />
  );
}
