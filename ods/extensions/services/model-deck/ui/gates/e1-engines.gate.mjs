import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createResults } from "./lib/check.mjs";
import { launch } from "./lib/browser.mjs";
import { isTrulyDisabled, textsOf } from "./lib/dom.mjs";
import { startStub } from "./stub-server.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const name = "e1-engines";

/** E1 items 1-3: the engines editor on the Nodes screen renders the seeded
 * triple, its Add-engine kind picker is API-sourced and filtered, and Save
 * genuinely refuses on an empty form.
 *
 * Selector table, read out of ui/src/components/NodesView.tsx at bfa75274
 * (labels/messages confirmed in ui/src/model/messages.ts):
 *
 * | What                  | Selector / text                                            |
 * |-----------------------|-------------------------------------------------------------|
 * | Nodes tab             | `.view-tabs button:has-text("Nodes")`                        |
 * | Local node rail row   | `.nodes-row:has-text("autarch")`                             |
 * | Engines section       | `.engines-section`                                           |
 * | Declared rows         | `.engines-list .engine-row`                                  |
 * | Row fields            | `.engine-row-resource`, `.engine-row-kind`, `.engine-row-gpu`|
 * | Pinned badge          | `.engine-row .ui-pill` (text `📌`)                           |
 * | Add button            | text `+ Add engine` (exact)                                  |
 * | Form                  | `.engine-form`                                               |
 * | Kind select           | `.engine-form label:has-text("Kind") select`                 |
 * | GPU select's label    | `.engine-form label:has-text("GPU")`                         |
 * | Save                  | `.engine-form-actions .primary` (text `Save`)                |
 * | Allowlist banner      | `.engine-form .ui-banner` (hasText "park allowlist")         |
 *
 * R4 (controller ruling): `:has-text()` substring-matches, and the engine
 * form has several `<label>` elements. "Kind" and "GPU" are each confirmed,
 * by reading every label text the form can render (Resource name, Kind, each
 * kind's own connection field label(s), GPU, Pinned, Priority, Idle TTL
 * [ui/src/model/messages.ts:857-866,873]), to be a substring of no OTHER
 * label's text — but `assertUnique` below still asserts the match count
 * rather than trusting that reading, so a future label addition that DOES
 * collide fails loudly here instead of silently widening the selector's
 * match set. */

async function assertUnique(page, selector, what) {
  const n = await page.locator(selector).count();
  if (n !== 1) {
    throw new Error(
      `e1-engines gate: expected exactly 1 ${what} (selector ${selector}), found ${n}. ` +
        `A selector that matches zero elements can make a check pass vacuously; one ` +
        `that matches more than expected can assert against the wrong element.`,
    );
  }
}

export async function run() {
  const results = createResults();
  const scenario = JSON.parse(
    await readFile(join(HERE, "fixtures/e1-seeded-triple/scenario.json"), "utf8"),
  );
  const stub = await startStub({ scenario, distDir: join(HERE, "../dist"), port: 0 });
  const { browser, page, consoleErrors } = await launch();
  try {
    await page.goto(stub.url, { waitUntil: "networkidle" });

    await assertUnique(page, '.view-tabs button:has-text("Nodes")', "Nodes tab button");
    await page.click('.view-tabs button:has-text("Nodes")');

    await assertUnique(page, '.nodes-row:has-text("autarch")', "local node rail row");
    await page.click('.nodes-row:has-text("autarch")');

    // Waits for an actual declared-engine row, not just the section shell:
    // EnginesSection renders `.engines-section` immediately in a loading
    // state, before its own GET /api/engine-kinds + GET /api/nodes fetches
    // resolve, so waiting on the section alone would race the fetch and
    // read an empty list.
    await page.waitForSelector(".engines-list .engine-row");

    // Item 1 — seeded triple renders with live-policy pinned badges.
    const rows = await textsOf(page, ".engines-list .engine-row .engine-row-resource");
    results.check(
      "item1: seeded triple renders",
      JSON.stringify(rows.slice().sort()) === JSON.stringify(["comfyui", "hipfire", "lemonade"]),
      rows.join(","),
    );
    const gpus = await textsOf(page, ".engines-list .engine-row .engine-row-gpu");
    results.check(
      "item1: hipfire on GPU 0, lemonade and comfyui on GPU 1",
      JSON.stringify(gpus) === JSON.stringify(["GPU 0", "GPU 1", "GPU 1"]),
      gpus.join(","),
    );
    const pins = await page.locator(".engines-list .engine-row .ui-pill").count();
    results.check(
      "item1: exactly one pinned badge (hipfire, per live policy)",
      pins === 1,
      String(pins),
    );

    // Item 2 — Add flow: kinds are API-sourced and filtered to local_capable,
    // GPUs are real, banner is present.
    //
    // R9 (controller ruling): /api/engine-kinds now serves FOUR kinds
    // (comfyui, hipfire, lemonade, sglang-omni — the Task 5 fixture carries
    // all four). The local form shows three only because sglang-omni has
    // local_capable: false and kindsFor() (ui/src/model/engineForm.ts)
    // filters on exactly that field. A check that only counts to three
    // cannot tell a working filter from a deleted one, so both halves are
    // asserted as distinct, separately-named checks.
    await page.click('text="+ Add engine"');
    await page.waitForSelector(".engine-form");

    await assertUnique(page, '.engine-form label:has-text("Kind")', "Kind label");
    await assertUnique(page, '.engine-form label:has-text("GPU")', "GPU label");

    const kinds = await textsOf(page, '.engine-form label:has-text("Kind") select option');
    results.check(
      "item2: kind picker offers the three local-capable kinds",
      JSON.stringify(kinds.slice().sort()) === JSON.stringify(["comfyui", "hipfire", "lemonade"]),
      kinds.join(","),
    );
    results.check(
      "item2: kind picker filters out sglang-omni (not local_capable)",
      !kinds.includes("sglang-omni"),
      kinds.join(","),
    );
    const banner = await page
      .locator(".engine-form .ui-banner", { hasText: "park allowlist" })
      .count();
    results.check("item2: park-allowlist note is visible in Add mode", banner === 1, String(banner));

    // Item 3 — Save disabled until required fields are filled. THE
    // :disabled trap: isTrulyDisabled uses el.matches(':disabled'), which is
    // false for a button inside a disabled <fieldset> that has no `disabled`
    // attribute of its own — this form has no such fieldset, but the check
    // still goes through the ancestor-safe helper rather than a raw
    // getAttribute, matching Task 2's contract.
    await assertUnique(page, ".engine-form-actions .primary", "Save button");
    results.check(
      "item3: Save is genuinely disabled on an empty form",
      await isTrulyDisabled(page, ".engine-form-actions .primary"),
    );

    // Deliberately no blanket "no console errors" check here (unlike
    // smoke.gate.mjs): this fixture's `sparky` node-agent is control:"swap",
    // so App.tsx's swap-probe effect fires GET /api/nodes/sparky/serving/status
    // and GET /api/settings/catalog/sparky/vllm on load — routes this
    // scenario does not script, because items 1-3 (Nodes tab, local node's
    // Engines editor) have nothing to do with sparky's swap-probe UI. Those
    // 599s are real but orthogonal to this gate's scope; asserting a global
    // console.errors.length === 0 here would fail for a reason unrelated to
    // anything items 1-3 claim to check, in a fixture a later task did not
    // author to be console-clean beyond what E1 itself needs. consoleErrors
    // stays wired through launch() so a future gate item CAN use it.
    void consoleErrors;
  } finally {
    await browser.close();
    await stub.stop();
  }
  return results;
}
