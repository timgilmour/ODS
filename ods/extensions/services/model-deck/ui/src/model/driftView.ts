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
