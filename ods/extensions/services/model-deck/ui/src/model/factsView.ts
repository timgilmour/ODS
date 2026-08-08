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

// ---------------------------------------------------------------------------
// Tags — the one declared field whose edits compose
// ---------------------------------------------------------------------------

/** The list to PUT when `raw` is added, or `null` when there is nothing to
 * write. A blank entry and a tag already on the model are both non-edits, and
 * the difference matters at the call site: a duplicate must NOT blank the
 * input, or the operator is left guessing where their word went. Trims,
 * because a tag is an identifier and " fast" and "fast" are one role named
 * twice. */
export function tagsWith(tags: string[], raw: string): string[] | null {
  const tag = raw.trim();
  if (tag === "" || tags.includes(tag)) return null;
  return [...tags, tag];
}

/** Whether the server's own view has caught up with the last tag list this
 * screen successfully wrote.
 *
 * A tags PUT replaces the WHOLE array (app/declared.py merges per FIELD, not
 * per element), so in the window between the write and the refetch that
 * reflects it, a second edit built on the pre-write array silently undoes the
 * first — remove A then B resurrects A, two quick adds drop the first. That
 * window is a full App refresh wide (three fetches, one of them to a remote
 * node), not an instant. So the drawer keeps the array it actually sent, and
 * further edits build on THAT until this returns true.
 *
 * Deliberately compares VALUES rather than counting refresh ticks: a tick
 * only says a fetch happened, not that it started after the PUT landed, and
 * clearing one fetch too early reopens exactly the window this closes. The
 * cost is the opposite bias — if another client edits the tags meanwhile,
 * this screen keeps showing its own last write until it writes again or is
 * reopened, which is the safer of the two ways to be wrong.
 */
export function tagsSettled(written: string[] | null, value: unknown): boolean {
  if (written === null) return false;
  const server = declaredTags(value);
  return server.length === written.length && server.every((t, i) => t === written[i]);
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
 *
 * ⚠ The row match is a NAME COLLISION, not a resolved relationship: a fact
 * that happened to be called `max_model_len` would capture the
 * `max_model_len` rule's item and print its numbers — which are the LIVE
 * length against `max_position_embeddings`, not against that fact — beneath
 * the wrong row. No such fact exists today. If one is ever derived, the fix
 * is for the drift payload to name the fact it compared against, not for
 * this file to learn the rules table.
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
