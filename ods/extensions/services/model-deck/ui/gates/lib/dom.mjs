/** Browser-side source for a genuine disabled check.
 *
 * Kept as a STRING built by a pure function so it can be unit-tested here;
 * page.evaluate would otherwise put it permanently out of reach of every
 * test in the repo — the same "logic no test can reach" shape this harness
 * exists to fix. */
export function disabledExpression(selector) {
  return `(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    return el.matches(':disabled');
  })()`;
}

export async function isTrulyDisabled(page, selector) {
  return await page.evaluate(disabledExpression(selector));
}

/** Text of every node matching `selector`, trimmed. */
export async function textsOf(page, selector) {
  return await page.$$eval(selector, (els) => els.map((e) => e.textContent.trim()));
}

/** No vacuous negations, and no silent wrong-element matches: a selector
 * that matches zero elements can make a positive check pass vacuously
 * (nothing was found, so nothing failed), and one that matches more than
 * one can quietly assert against the wrong element. `assertUnique` throws
 * loudly in both cases instead.
 *
 * This was duplicated verbatim (save for the gate-name prefix in the thrown
 * message) in `e1-engines.gate.mjs` and `e1-board.gate.mjs` — a third gate
 * would have copied it a third time, and this README used to tell a future
 * author to "call `assertUnique`" as though it were already importable from
 * here. `makeAssertUnique(gateName)` is the hoisted, single copy: it returns
 * a 3-arg `assertUnique(page, selector, what)` closed over the calling
 * gate's own name, so every existing call site keeps its exact shape. */
export function makeAssertUnique(gateName) {
  return async function assertUnique(page, selector, what) {
    const n = await page.locator(selector).count();
    if (n !== 1) {
      throw new Error(
        `${gateName} gate: expected exactly 1 ${what} (selector ${selector}), found ${n}. ` +
          `A selector that matches zero elements can make a check pass vacuously; one ` +
          `that matches more than expected can assert against the wrong element.`,
      );
    }
  };
}
