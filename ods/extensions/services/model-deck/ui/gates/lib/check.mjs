/** Result accumulator shared by every gate script.
 *
 * Deliberately strict on two things a browser gate gets wrong easily:
 * duplicate names (a FAIL hiding behind a same-named PASS in the report)
 * and non-boolean `ok` (an ElementHandle is truthy, null is falsy, so
 * `check("x", await page.$(sel))` would "work" until the selector matched
 * something unexpected). Both throw rather than coerce. */
export function createResults() {
  const rows = [];
  const seen = new Set();
  return {
    check(name, ok, detail = "") {
      if (typeof ok !== "boolean") {
        throw new TypeError(`check("${name}"): ok must be a boolean, got ${typeof ok}`);
      }
      if (seen.has(name)) throw new Error(`duplicate check name: ${name}`);
      seen.add(name);
      rows.push({ name, ok, detail });
      console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
      return ok;
    },
    rows: () => rows.slice(),
    failed: () => rows.filter((r) => !r.ok).length,
  };
}

/** R15 (controller ruling): a gate that throws mid-run must still hand
 * `run.mjs` the checks that already printed PASS/FAIL, not lose them. A
 * bare throw out of a gate's `run()` propagates past every `results.check()`
 * call already made — `results` is local to that function and would
 * otherwise vanish with the exception, degrading `run.mjs`'s report into one
 * synthetic "gate threw" row with none of the real checks behind it (see
 * `run.mjs`'s own comment on `err.partialRows`).
 *
 * `reportingRun(fn)` is the one place this attach-before-rethrow happens: it
 * owns `createResults()`, hands the accumulator to `fn`, and on any throw
 * from `fn` stamps the rows gathered so far onto the error as `partialRows`
 * before letting it propagate — the same contract every gate needs, written
 * once instead of copied per gate file. A gate's `run()` becomes:
 *
 *     export async function run() {
 *       return reportingRun(async (results) => {
 *         ... results.check(...) calls, may throw ...
 *       });
 *     }
 *
 * Any setup that must run before a `results` accumulator even makes sense
 * (e.g. fidelity.gate.mjs's `--deck-url` precondition) stays outside this
 * call, exactly as it did before this helper existed — a throw there was
 * never eligible for `partialRows` (there were no rows yet) and still isn't. */
export async function reportingRun(fn) {
  const results = createResults();
  try {
    await fn(results);
  } catch (err) {
    err.partialRows = results.rows();
    throw err;
  }
  return results;
}
