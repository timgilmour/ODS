/**
 * Search and filter catalog options for the All-options modal.
 *
 * Pure: no React, no fetch, no side effects. Everything is a function of
 * its inputs. Mirrors app/harvest.py's parse_probe_output for canonical
 * option identity.
 */

import type { Catalog, CatalogOption, Widget } from "../api";

/**
 * A filtered and sorted catalog row, ready to render.
 */
export interface CatalogRow {
  /** Option name (canonical key from catalog.options). */
  name: string;
  /** CLI flag form: `--${name}`. */
  flag: string;
  /** The full option from the catalog. */
  option: CatalogOption;
  /** True when this option name is in the provided setNames. */
  isSet: boolean;
}

/**
 * Filter and sort catalog options by query, widget type, and set membership.
 *
 * @param catalog - The option catalog (from GET /api/settings/catalog/{node}/{engine})
 * @param opts - Filter options
 * @param opts.query - Search query: matches name, aliases (dash-insensitive), and help text, case-insensitive substring. Empty/whitespace matches all. Leading dashes are stripped from the query upfront so users can type displayed flag forms (e.g., "--host") and match options named "host". This also affects help-text matching (a query of "--foo" searches help for "foo"), which is accepted since help prose rarely contains flag spellings and finding the option beats matching its mention.
 * @param opts.widgets - Widget type whitelist. Empty array = no filter. Otherwise only options whose widget is in this list.
 * @param opts.setOnly - When true, include only options whose name is in setNames. When false, include all (but still badge isSet).
 * @param opts.setNames - Set of option names to badge as isSet. Used both for filtering (when setOnly=true) and badging (always).
 * @returns Rows sorted by name (localeCompare).
 */
export function filterCatalog(
  catalog: Catalog,
  opts: {
    query: string;
    widgets: Widget[];
    setOnly: boolean;
    setNames: Set<string>;
  },
): CatalogRow[] {
  const { query, widgets, setOnly, setNames } = opts;

  // Normalize query: trim, lowercase, and strip leading dashes. Dashes are stripped
  // upfront so users can type the displayed flag form (e.g., "--host" or "--ctx-len")
  // and match options without rescuing aliases.
  const normalizedQuery = query.trim().toLowerCase().replace(/^-+/, "");

  // Filter by query match: name, aliases, or help text.
  const queryMatches = (name: string, option: CatalogOption): boolean => {
    // Empty query matches all.
    if (normalizedQuery === "") return true;

    // Match against name (case-insensitive substring).
    if (name.toLowerCase().includes(normalizedQuery)) return true;

    // Match against help text (case-insensitive substring).
    if (option.help.toLowerCase().includes(normalizedQuery)) return true;

    // Match against aliases (dash-insensitive). Strip leading dashes from
    // aliases before comparing against the already-normalized query.
    for (const alias of option.aliases) {
      const aliasWithoutDashes = alias.replace(/^-+/, "");
      if (aliasWithoutDashes.includes(normalizedQuery)) return true;
    }

    return false;
  };

  // Filter by widget type. Empty array = no filter.
  const widgetMatches = (option: CatalogOption): boolean => {
    if (widgets.length === 0) return true;
    return widgets.includes(option.widget);
  };

  // Build rows for all matching options, then apply setOnly filter.
  const rows: CatalogRow[] = Object.entries(catalog.options)
    .filter(([name, option]) => queryMatches(name, option) && widgetMatches(option))
    .map(([name, option]) => ({
      name,
      flag: `--${name}`,
      option,
      isSet: setNames.has(name),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  // When setOnly is true, keep only rows in setNames.
  if (setOnly) {
    return rows.filter((r) => setNames.has(r.name));
  }

  return rows;
}
