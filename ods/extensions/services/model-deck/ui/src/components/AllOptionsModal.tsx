import { useMemo, useState } from "react";
import { postHarvest, type Catalog, type Widget } from "../api";
import { filterCatalog, type CatalogRow } from "../model/catalogFilter";
import { humanizeAge, labels, messages, type Message } from "../model/messages";
import Banner from "../ui/Banner";
import Modal from "../ui/Modal";

// The five widget categories app/harvest.py:widget_for ever produces,
// mirrored as `Widget` in api.ts — every filter chip this row offers.
const WIDGETS: Widget[] = ["toggle", "list", "select", "number", "text"];

/**
 * The searchable harvested-option catalog (Settings panel's "+ Add option").
 * `catalog` may be `null` — a pair that has never harvested is a supported
 * state (app/routers/settings.py:get_catalog) — in which case there is
 * nothing to search yet and Refresh is the only useful control.
 *
 * `onAdd` is a single callback for BOTH the "+" on an unset row and the "✓"
 * on an already-set one: this component only knows `isSet` (from
 * `setNames`, computed by the caller), not what the parent should do about
 * it. SettingsModal's handler is what decides "buffer a fresh value" vs.
 * "just open the existing chip's editor" — see its `handleAddOption`.
 *
 * Search + filters are plain markup, not `ui/Toolbar`: this row needs an
 * autofocused input and an Enter handler that adds the first visible row,
 * neither of which Toolbar's props expose, and extending Toolbar.tsx for
 * one caller was out of scope here. It reuses Toolbar's own CSS classes
 * (`ui-toolbar`, `ui-toolbar-search`, `ui-filter-chip`, `ui-filter-active`)
 * so it looks identical anyway.
 */
export default function AllOptionsModal({
  node,
  engine,
  catalog,
  setNames,
  onAdd,
  onClose,
  onRefreshed,
}: {
  node: string;
  engine: string;
  catalog: Catalog | null;
  setNames: Set<string>;
  onAdd: (name: string) => void;
  onClose: () => void;
  onRefreshed: () => void;
}) {
  const [query, setQuery] = useState("");
  const [widgets, setWidgets] = useState<Widget[]>([]);
  const [setOnly, setSetOnly] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshNote, setRefreshNote] = useState<Message | null>(null);
  const [refreshError, setRefreshError] = useState<Message | null>(null);

  const rows: CatalogRow[] = useMemo(() => {
    if (!catalog) return [];
    return filterCatalog(catalog, { query, widgets, setOnly, setNames });
  }, [catalog, query, widgets, setOnly, setNames]);

  const optionCount = catalog ? Object.keys(catalog.options).length : 0;

  function toggleWidget(w: Widget) {
    setWidgets((cur) => (cur.includes(w) ? cur.filter((x) => x !== w) : [...cur, w]));
  }

  async function handleRefresh() {
    setRefreshing(true);
    setRefreshNote(null);
    setRefreshError(null);
    try {
      const { outcome } = await postHarvest(node, engine);
      if (outcome === "harvested") {
        // Parent refetches via getCatalog and passes the new `catalog` prop
        // down — this component holds no catalog state of its own.
        onRefreshed();
      } else if (outcome === "current") {
        setRefreshNote(messages.catalogCurrent());
      } else {
        setRefreshError(messages.harvestFailed(outcome));
      }
    } catch (err) {
      // ApiError (and any other transport failure) walks the same banner
      // path as a "failed"/"empty" outcome — see messages.harvestFailed.
      setRefreshError(
        messages.harvestFailed(err instanceof Error ? err.message : String(err)),
      );
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <Modal
      title={labels.allOptions}
      subtitle={`${engine} on ${node}`}
      onClose={onClose}
      footer={
        <button type="button" onClick={onClose}>
          {labels.close}
        </button>
      }
    >
      {/* Escape closes the modal from anywhere inside it, not just the
          search input — a filter-chip button can hold focus just as easily
          after a click. */}
      <div
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
        }}
      >
        {refreshError && (
          <Banner message={refreshError} onDismiss={() => setRefreshError(null)} />
        )}
        {refreshNote && (
          <Banner message={refreshNote} onDismiss={() => setRefreshNote(null)} />
        )}

        <div className="catalog-provenance">
          <span>
            {labels.catalogAge(humanizeAge(catalog?.harvested_ts ?? null))} ·{" "}
            {labels.optionCount(optionCount)}
          </span>
          <button type="button" onClick={handleRefresh} disabled={refreshing}>
            {refreshing ? labels.refreshing : labels.refresh}
          </button>
        </div>

        <div className="ui-toolbar">
          <input
            autoFocus
            className="ui-toolbar-search"
            value={query}
            placeholder={labels.searchOptions}
            aria-label={labels.searchOptions}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              // The mockup's keyboard-focused-row affordance, simplified:
              // Enter always targets the first visible row rather than
              // tracking a separately-moved selection cursor.
              if (e.key === "Enter") {
                e.preventDefault();
                if (rows.length > 0) onAdd(rows[0].name);
              }
            }}
          />
          {WIDGETS.map((w) => (
            <button
              key={w}
              type="button"
              className={`ui-filter-chip ${widgets.includes(w) ? "ui-filter-active" : ""}`}
              aria-pressed={widgets.includes(w)}
              onClick={() => toggleWidget(w)}
            >
              {labels.widgetName(w)}
            </button>
          ))}
          <button
            type="button"
            className={`ui-filter-chip ${setOnly ? "ui-filter-active" : ""}`}
            aria-pressed={setOnly}
            onClick={() => setSetOnly((v) => !v)}
          >
            {labels.setOnly}
          </button>
        </div>

        {!catalog ? (
          <div className="catalog-empty">{labels.catalogNeverHarvested}</div>
        ) : rows.length === 0 ? (
          <div className="catalog-empty">{labels.noCatalogMatches}</div>
        ) : (
          <div className="catalog-rows">
            {rows.map((row) => (
              <CatalogRowView key={row.name} row={row} onAdd={onAdd} />
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}

function CatalogRowView({ row, onAdd }: { row: CatalogRow; onAdd: (name: string) => void }) {
  const { name, flag, option, isSet } = row;
  // "None" is app/harvest.py:81's spelling of "the engine has no default"
  // (a repr'd Python `None`), not a value — render nothing for it.
  const defaultText = option.default === "None" ? null : option.default;
  const label = isSet ? labels.jumpToOptionLabel(flag) : labels.addOptionRowLabel(flag);

  return (
    <div className={`catalog-row ${isSet ? "catalog-row-set" : ""}`}>
      <button
        type="button"
        className={`catalog-row-add ${isSet ? "catalog-row-add-set" : ""}`}
        aria-label={label}
        title={label}
        onClick={() => onAdd(name)}
      >
        {isSet ? "✓" : "+"}
      </button>
      <div className="catalog-row-main">
        <div className="catalog-row-head">
          <span className="catalog-row-flag">{flag}</span>
          {option.aliases.length > 0 && (
            <span className="catalog-row-aliases">{option.aliases.join(" ")}</span>
          )}
          <span className="catalog-row-badge">{option.type ?? option.widget}</span>
          {defaultText !== null && <span className="catalog-row-default">= {defaultText}</span>}
        </div>
        {option.help && (
          <div className="catalog-row-help" title={option.help}>
            {option.help}
          </div>
        )}
      </div>
    </div>
  );
}
