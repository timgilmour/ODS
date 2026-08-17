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
