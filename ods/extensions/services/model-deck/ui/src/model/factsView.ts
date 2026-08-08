/**
 * Facts, as the model detail drawer reads them.
 *
 * Pure: no React, no fetch, no clock — same rule as model/nodes.ts and
 * model/settingsView.ts. Everything here exists because `FactEntry.value` is
 * `unknown` BY CONTRACT (app/facts.py:resolve_facts copies whatever the
 * characteristics cache holds — strings, numbers, lists, nested maps), so
 * every render of one is a coercion, and a coercion inlined in a component is
 * a coercion no test can reach.
 */

import type { FactEntry, FactsDriftItem } from "../api";

/** A fact value as an operator reads it.
 *
 * Objects and arrays are JSON-rendered compactly rather than stringified by
 * JS's default rules: `String({a: 1})` is "[object Object]", and
 * `String([1,2])` is "1,2" — both of which look like a value rather than
 * like structure. `null` prints as "null" because a fact that IS null is a
 * fact, and printing it as an empty cell would read as "not recorded".
 */
export function factValueText(value: unknown): string {
  if (typeof value === "string") return value;
  // Only ever reachable for a `shadowed_value` that is absent — the caller
  // checks that first — but a fact map arriving without a key must not
  // render "undefined".
  if (value === undefined) return "";
  if (value === null) return "null";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** A declared TEXT field (`label`, `notes`, `engine_preference`) as an
 * editable draft. app/declared.py:32-38 validates all three as `str` on
 * write, so a non-string here can only be a DERIVED value shadowed under the
 * same name — which the form must not adopt as if a human had typed it. "" is
 * then the honest starting draft: nothing declared yet. */
export function declaredText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** The declared `tags` list. Validated `list[str]` on write
 * (app/declared.py:32-38); anything else came from somewhere else and is not
 * a tag list, so it renders as no tags rather than as garbage chips. Filters
 * per-element too: one non-string entry must not discard the rest. */
export function declaredTags(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((t): t is string => typeof t === "string");
}

/** The fact rows of one key, name-sorted. Sorted because the map arrives in
 * insertion order (derived first, then declared — app/facts.py:resolve_facts)
 * and a table whose row order shifts when a human declares something is a
 * table nobody can scan twice. */
export function factRows(
  entry: Record<string, FactEntry> | undefined,
): Array<[string, FactEntry]> {
  if (!entry) return [];
  return Object.entries(entry).sort(([a], [b]) => a.localeCompare(b));
}

/** Drift items split into "belongs to a fact row on screen" and "does not".
 *
 * ⚠ The second half is not a defensive afterthought — TODAY it is the normal
 * path. A drift item's `field` is the RUNTIME field name
 * (app/facts.py:detect_drift appends `runtime_field`), and DRIFT_RULES
 * (app/facts.py:31-36) maps each of those onto a DIFFERENTLY NAMED fact:
 * `quantization`→`quant_method`, `max_model_len`→`max_position_embeddings`,
 * `max_input_tokens`→`max_model_len_live`. So none of the three rules the
 * deck ships today names a fact row, and a UI that only amber-flagged
 * matching rows would render drift NOWHERE while the backend was reporting
 * it — the vocabulary bug this codebase has now paid for four times.
 *
 * The mapping is deliberately NOT mirrored here: a second copy of
 * DRIFT_RULES in TypeScript is exactly what goes stale silently. Unmatched
 * items are rendered on their own, naming their runtime field verbatim, so
 * every item the backend reports reaches the screen either way.
 */
export function partitionDrift(
  items: FactsDriftItem[] | undefined,
  factNames: string[],
): { byName: Map<string, FactsDriftItem>; unmatched: FactsDriftItem[] } {
  const names = new Set(factNames);
  const byName = new Map<string, FactsDriftItem>();
  const unmatched: FactsDriftItem[] = [];

  for (const item of items ?? []) {
    // One row carries one drift line, so the FIRST item for a matching fact
    // takes the row and any further item for that same field falls through to
    // the list below rather than being dropped: two rules can name one field,
    // and a silently discarded disagreement is the failure mode this whole
    // function exists to prevent.
    if (names.has(item.field) && !byName.has(item.field)) {
      byName.set(item.field, item);
      continue;
    }
    unmatched.push(item);
  }

  return { byName, unmatched };
}
