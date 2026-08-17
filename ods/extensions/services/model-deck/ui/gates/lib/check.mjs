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
