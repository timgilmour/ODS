/**
 * Drift row shaping — the settings-drift card's one piece of pure logic.
 * Pure: no React, no fetch, no clock, same rule as model/nodes.ts and
 * model/settingsView.ts.
 *
 * `SettingsDriftEntry.old`/`.new` are `unknown` by contract (api.ts): a
 * changed key can live in the `args` namespace (`ArgValue` — string,
 * string[], boolean) or in `env`/`container` (arbitrary JSON — see
 * app/settings_store.py's `CONTAINER_ALLOWLIST`), so nothing here may assume
 * a shape beyond "JSON-serializable or a string".
 */

import type { SettingsDriftEntry } from "../api";

const ARGS_PREFIX = "args:";

/** Every changed key is namespace-qualified — `"args:max-model-len"`, never
 * a bare `"max-model-len"` (the fold app/routers/__init__.py:239 performs,
 * so two same-named keys in different namespaces never dedupe into one).
 * `args` is the overwhelming majority of settings keys, so its prefix is
 * stripped for DISPLAY ONLY — `env:`/`container:` stay visible verbatim,
 * because for those two the namespace IS the disambiguation an operator
 * needs. */
export function displayKeyFor(key: string): string {
  return key.startsWith(ARGS_PREFIX) ? key.slice(ARGS_PREFIX.length) : key;
}

/** An old/new value as text. A string renders verbatim — `"131072"` reads as
 * itself, not `'"131072"'` — because that is the one shape with its own
 * established "how an operator reads this" convention on this board
 * (model/settingsView.ts's `displayValue`). Everything else (numbers,
 * booleans, arrays, nested container mappings) has no such convention here,
 * so it is JSON.stringify'd compactly rather than guessed at. Never called
 * with `null` — the card's added/removed sentinel is handled by `driftRows`
 * before this runs. */
export function driftValueText(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export interface DriftRow {
  /** The qualified key, verbatim — stable as a React list key. */
  key: string;
  /** `key` with the `args:` namespace stripped for display. */
  displayKey: string;
  /** `null` means there was no old value — settings_store.py:252's
   * `before.get(name)` for a key that did not exist before this change (the
   * ADDED case). Never the string `"null"`. */
  oldText: string | null;
  /** `null` means the key was removed — settings_store.py:252's
   * `ns.get(name)` after a `remove` (the REMOVED case). Never the string
   * `"null"`. */
  newText: string | null;
}

/** Shapes the exact old->new entries the journal path produces
 * (app/routers/__init__.py's `_settings_drift`) into rows the card renders.
 * Never fabricates a row for a key the caller did not supply — the legacy
 * (pre-journal) path has no entries at all, and rendering it is the card's
 * job, not this function's. */
export function driftRows(entries: SettingsDriftEntry[]): DriftRow[] {
  return entries.map((e) => ({
    key: e.key,
    displayKey: displayKeyFor(e.key),
    oldText: e.old === null ? null : driftValueText(e.old),
    newText: e.new === null ? null : driftValueText(e.new),
  }));
}

/** A changed key the report named but carries no old→new entry for: the
 * legacy (pre-journal) path's names-only output. `key` is the qualified key
 * verbatim (stable React key); `displayKey` is what an operator reads. */
export interface DriftLegacyKey {
  key: string;
  displayKey: string;
}

export interface DriftPartition {
  rows: DriftRow[];
  legacy: DriftLegacyKey[];
}

/** Splits one drift report into the two things the card can actually show.
 *
 * A report is NOT one path or the other: `_settings_drift`
 * (app/routers/__init__.py) walks up to three scopes × three namespaces and
 * chooses per namespace — journal present means exact old→new `entries`,
 * journal absent means C1's every-current-key names, appended to `changed`
 * with no entry at all. A placement whose engine scope has a journal and
 * whose model scope predates it (every namespace stamped before Task 1's
 * journal existed — i.e. live ds4 on its first post-deploy settings edit)
 * produces exactly that MIXED payload. Rendering rows-or-list exclusively
 * then hid the journal-less names entirely while the header, which counts
 * `changed`, still said "3 keys changed" over one visible row.
 *
 * Matching is on the DISPLAY key, `displayKeyFor` on both sides, not on the
 * raw string: both lists are namespace-qualified by the same backend fold, so
 * this agrees with a raw comparison for every payload the server actually
 * emits, and an unqualified key on one side only (a hand-written fixture, an
 * older report) still pairs with its row instead of being printed twice. */
export function partitionDrift(
  changed: string[],
  entries: SettingsDriftEntry[],
): DriftPartition {
  const rows = driftRows(entries);
  const hasRow = new Set(rows.map((r) => r.displayKey));
  const legacy = changed
    .map((key) => ({ key, displayKey: displayKeyFor(key) }))
    .filter((k) => !hasRow.has(k.displayKey));
  return { rows, legacy };
}
