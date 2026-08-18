/** Pure helpers for `--capture`'s R17 union-by-default behavior (review fix
 * round 1, IMPORTANT finding). A single live snapshot only ever shows the
 * vocabulary a box happens to be exercising AT THAT INSTANT — a healthy box
 * never shows `status: "unreachable"`, a quiet event tail can easily miss
 * `apply-vetoed`. `run.mjs`'s old `--capture` behavior was a bare overwrite
 * of the committed `vocabulary.json` with exactly that snapshot, which meant
 * a routine re-capture from a healthy/quiet box could silently drop a
 * genuinely legitimate token or shape key the moment it wasn't reproduced —
 * undoing hand-verified coverage with no warning. A comment saying "hand-
 * verify before committing" is not a guard against that.
 *
 * `unionVocabulary` merges two `{shape, tokens}` vocabularies key-by-key,
 * array-by-array, so a plain re-capture can only ever GROW the committed
 * fixture, never shrink it. `shrinkDelta` reports what one vocabulary drops
 * relative to another, independent of how the candidate was produced — used
 * both to prove `unionVocabulary` never shrinks (see the test file) and to
 * print what an explicit `--allow-shrink` overwrite is about to drop, so
 * even a deliberate shrink is a visible delta, not a silent one. */

function unionArrays(a, b) {
  return [...new Set([...(a ?? []), ...(b ?? [])])].sort();
}

function unionMap(a, b) {
  const out = {};
  for (const key of new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})])) {
    out[key] = unionArrays(a?.[key], b?.[key]);
  }
  return out;
}

/** `{shape, tokens}` x `{shape, tokens}` -> `{shape, tokens}`, unioned
 * section-by-section, array-by-array. Neither input is mutated; every array
 * in the result is a fresh, sorted, deduplicated copy. */
export function unionVocabulary(existing, observed) {
  return {
    shape: unionMap(existing?.shape, observed?.shape),
    tokens: unionMap(existing?.tokens, observed?.tokens),
  };
}

/** Every key/token `existing` carried that `next` does not — the exact
 * thing a bare overwrite can do silently. Returns one human-readable string
 * per (section, key) that lost at least one entry, e.g. `tokens["status"]:
 * unreachable, down`. An empty array means `next` is a superset of
 * `existing` at every key `existing` declares (a NEW key/family in `next`
 * that `existing` never had is not a shrink — this only ever reports
 * entries `existing` had and `next` dropped). */
export function shrinkDelta(existing, next) {
  const lost = [];
  for (const section of ["shape", "tokens"]) {
    const existingSection = existing?.[section] ?? {};
    const nextSection = next?.[section] ?? {};
    for (const key of Object.keys(existingSection)) {
      const before = new Set(existingSection[key] ?? []);
      const after = new Set(nextSection[key] ?? []);
      const dropped = [...before].filter((t) => !after.has(t));
      if (dropped.length) lost.push(`${section}["${key}"]: ${dropped.join(", ")}`);
    }
  }
  return lost;
}
