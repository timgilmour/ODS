import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { errors as playwrightErrors } from "playwright-core";
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
 * | Cancel                | `.engine-form-actions button` (text `Cancel`, exact)         |
 * | Allowlist banner      | `.engine-form .ui-banner` (hasText "park allowlist")         |
 * | Policy button         | `text="Policy"` (exact — App.tsx header)                     |
 * | Policy table rows     | `.policy-table .tenant-name`                                 |
 * | Row Edit button       | `.engine-row:has-text(RESOURCE) button:has-text("Edit")`     |
 * | Row Forget button     | `.engine-row:has-text(RESOURCE) button:has-text("Forget")`   |
 * | Armed-confirm caption | `.engine-row .engine-caption` (rendered only while armed)    |
 *
 * R4 (controller ruling): `:has-text()` substring-matches, and the engine
 * form has several `<label>` elements. "Kind" and "GPU" are each confirmed,
 * by reading every label text the form can render (Resource name, Kind, each
 * kind's own connection field label(s), GPU, Pinned, Priority, Idle TTL
 * [ui/src/model/messages.ts:857-866,873]), to be a substring of no OTHER
 * label's text — but `assertUnique` below still asserts the match count
 * rather than trusting that reading, so a future label addition that DOES
 * collide fails loudly here instead of silently widening the selector's
 * match set. Items 4-7 (Task 7) add two exact-text() selectors: `text="Policy"`
 * (labels.policy on the App.tsx header button, and labels.policyTitle on the
 * PolicyModal's own <h3> once open — asserted unique BEFORE the modal opens,
 * when only the button exists) and `text="Cancel"` (labels.cancel, shared by
 * EngineFormPanel's Cancel and PolicyModal's footer Cancel — exact matching
 * means neither collides with the unrelated "Cancel move" label, and only one
 * of the two Cancel buttons is ever mounted at the point this gate clicks it). */

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

    // Item 4 — Add the dummy. Two DISTINCT claims, asserted separately
    // (R12, controller ruling):
    //
    //   (a) DISPATCH — the POST actually carried what the operator typed.
    //       Read straight from stub.requests(), never from anything the UI
    //       re-renders. gpu_index is the load-bearing field: GPU 0 is the
    //       picker's first <option>, so a form that silently dropped the
    //       gpu_index selection would still coincidentally submit 0 — the
    //       fixture's gguf-test is GPU 1 for exactly this reason.
    //
    //   (b) RENDER — the new row appears in the declared-engines list. This
    //       is deliberately NOT read off a board card / `/api/state`: that
    //       route is polled every 3000ms (App.tsx POLL_MS) independent of
    //       any click, so a scripted state transition on it would advance on
    //       a TIMER and could show "gguf-test" whether or not the POST ever
    //       fired — a check that passes when the feature is broken. The
    //       engines list instead comes from GET /api/nodes
    //       (listNodeRegistry, via EnginesSection's reload()), which this
    //       app calls from exactly two places: on mount, and from
    //       afterMutate() right after a successful add/edit/forget — never
    //       from the poll. Its scripted 3-entry transition (triple ->
    //       triple+gguf-test -> triple, fixtures/e1-seeded-triple/
    //       scenario.json) therefore only advances in lockstep with this
    //       gate's own clicks, so it is a check that can actually fail when
    //       the add flow is broken. The board-card half of item 4's original
    //       "appears as a board card" wording is NOT asserted here — see
    //       this task's report for why.
    await page.fill('.engine-form label:has-text("Resource name") input', "gguf-test");
    await page.selectOption('.engine-form label:has-text("Kind") select', "lemonade");
    // lemonade's connection fields (GET /api/engine-kinds: url, metrics_url,
    // container, all required) render as labels DERIVED from the field key
    // (engineFieldLabel, model/messages.ts) — "url", "metrics url",
    // "container" — with " *" appended when required. "url" is a SUBSTRING
    // of "metrics url"'s own label text, so a plain :has-text("url") would
    // match both and Playwright's strict mode would refuse to fill either
    // (R4): excluded with :not(:has-text("metrics")).
    await page.fill(
      '.engine-form label:has-text("url"):not(:has-text("metrics")) input',
      "http://llama-server-test:8080",
    );
    await page.fill(
      '.engine-form label:has-text("metrics url") input',
      "http://llama-server-test:8001/metrics",
    );
    await page.fill(
      '.engine-form label:has-text("container") input',
      "ods-llama-server-test",
    );
    await page.selectOption('.engine-form label:has-text("GPU") select', "1");
    await page.click(".engine-form-actions .primary");
    await page.waitForTimeout(300);
    const posted = stub.requests().find((r) => r.method === "POST");
    results.check(
      "item4: Save dispatched POST /api/nodes/local/engines with the typed values",
      posted?.path === "/api/nodes/local/engines" &&
        posted?.body?.resource === "gguf-test" &&
        posted?.body?.kind === "lemonade" &&
        posted?.body?.gpu_index === 1,
      JSON.stringify(posted ?? null),
    );
    const after = await textsOf(page, ".engines-list .engine-row .engine-row-resource");
    results.check("item4: the new engine renders as a row", after.includes("gguf-test"));

    await assertUnique(page, '.engine-row:has-text("gguf-test")', "the new gguf-test row");

    // Item 5 — PolicyModal renders every row of its policy map. PolicyModal
    // takes its `policy` prop straight off /api/state's own `policy` field
    // (App.tsx: `<PolicyModal policy={state.policy} .../>`) — it issues no
    // GET of its own (putPolicy is a PUT only). /api/state IS the
    // ambient-polled route the item-4(b) comment above rules out for
    // proving CAUSALITY (App.tsx's POLL_MS=3000 timer advances it whether
    // or not any click happened) — R12 (controller ruling): a polled route
    // may carry only a CONSTANT, never a transition a check leans on. So
    // the fixture's GET /api/state is a single repeat:true entry, 4 policy
    // keys (gguf-test included) from the very first fetch — not a
    // 3-then-4 script. This check therefore does NOT claim the Add caused
    // the 4th row (that causal claim is item 4's job, proven above via
    // GET /api/nodes, which only advances on mount/afterMutate); it only
    // asserts PolicyModal's own row-fidelity against a constant policy
    // map it is handed — exactly what this item's RED mutation (dropping a
    // row PolicyModal was given) targets.
    await assertUnique(page, 'text="Policy"', "Policy button");
    await page.click('text="Policy"');
    const policyRows = await page.locator(".policy-table .tenant-name").count();
    results.check(
      "item5: PolicyModal renders every policy row it is given (4, not the seeded 3)",
      policyRows === 4,
      String(policyRows),
    );
    // Not Escape: Modal.tsx (the shared shell) wires no keydown handler —
    // only AllOptionsModal/SettingsModal/ModelDetailDrawer do that
    // individually — so PolicyModal has no Escape-to-close. Its footer
    // Cancel button is the real close path (discards local edits, never
    // calls putPolicy). At this point in the flow it is the ONLY "Cancel"
    // on screen: item 4's Add form already closed via afterMutate.
    await page.click('text="Cancel"');
    await page.waitForSelector(".policy-table", { state: "detached" });

    // Item 6 — Edit: resource locked, kind still editable (deliberately, in
    // BOTH modes — NodesView.tsx's EngineFormPanel doc comment: a resource
    // re-declared under a different kind is explicitly accommodated).
    //
    // isTrulyDisabled (dom.mjs) runs its selector through the BROWSER's own
    // `document.querySelector`, not Playwright's selector engine — so a
    // Playwright-only pseudo-class like `:has-text()` throws inside
    // page.evaluate ("not a valid selector") rather than failing the check.
    // Resource name and Kind are always the FIRST and SECOND <label> in
    // EngineFormPanel's JSX, before the kind-varying connection-field
    // block, so `label:nth-of-type(n)` — plain CSS, browser-native — is both
    // valid here and stable across every kind.
    //
    // disabledExpression (dom.mjs) returns false for a MISSING element, not
    // an error — so on the "kind stays editable" check (negated: expects
    // isTrulyDisabled to be false), a wrong/stale selector would ALSO
    // return false and the negation would read as PASS. assertUnique closes
    // that: a selector matching zero (or more than one) element fails
    // loudly here, before either isTrulyDisabled call ever runs.
    await page.click('.engine-row:has-text("gguf-test") button:has-text("Edit")');
    await page.waitForSelector(".engine-form");
    await assertUnique(page, ".engine-form label:nth-of-type(1) input", "Resource name input (edit mode)");
    await assertUnique(page, ".engine-form label:nth-of-type(2) select", "Kind select (edit mode)");
    results.check(
      "item6: resource is locked in Edit mode",
      await isTrulyDisabled(page, ".engine-form label:nth-of-type(1) input"),
    );
    results.check(
      "item6: kind stays editable in Edit mode",
      !(await isTrulyDisabled(page, ".engine-form label:nth-of-type(2) select")),
    );

    // Item 7 — Forget is armed, states what it does, and isolates its
    // arming (does not arm any other row's Forget). Uses the shared armed
    // machinery (ArmedButton + model/armed.ts): the FIRST click on a row's
    // Forget arms it; the SECOND, on the same button, confirms it. The
    // rendered label text does not change between the two states (only
    // aria-label does), so the same selector re-clicked is correct, not
    // accidental.
    await page.click('text="Cancel"');
    await assertUnique(page, '.engine-row:has-text("gguf-test") button:has-text("Forget")', "gguf-test Forget button");
    await page.click('.engine-row:has-text("gguf-test") button:has-text("Forget")');
    // Bounded wait, not the 30s locator default: a mutation that fires
    // doForget on the first click (no arm step at all) would otherwise hang
    // here for 30s before failing loudly instead of failing fast. Only a
    // TimeoutError (the caption never showed up) is swallowed into "" —
    // matching this check's own "absent counts as not-armed" contract;
    // anything else propagates.
    let confirmCopy = "";
    try {
      confirmCopy = await page
        .locator('.engine-row:has-text("gguf-test") .engine-caption')
        .innerText({ timeout: 3000 });
    } catch (err) {
      if (!(err instanceof playwrightErrors.TimeoutError)) throw err;
      confirmCopy = "";
    }
    results.check(
      "item7: armed copy states the engine keeps running",
      confirmCopy.includes("a running engine keeps running"),
      confirmCopy,
    );
    // assertUnique on the hipfire row itself, not just the caption count:
    // otherArmed === 0 is trivially true for a selector that matches zero
    // elements FOR THE WRONG REASON (stale resource name, typo) as much as
    // for the right one (row exists, genuinely unarmed) — R4/finding 2.
    await assertUnique(page, '.engine-row:has-text("hipfire")', "the hipfire row");
    const otherArmed = await page
      .locator('.engine-row:has-text("hipfire") .engine-caption')
      .count();
    results.check("item7: arming one row does not arm another", otherArmed === 0, String(otherArmed));
    // Bounded for the same reason as the innerText() read above: a mutation
    // that fires doForget on the FIRST click already removes the row (via
    // afterMutate's reload()) before this second click, which would
    // otherwise hang 30s waiting for a target that is never coming back.
    try {
      await page.click(
        '.engine-row:has-text("gguf-test") button:has-text("Forget")',
        { timeout: 3000 },
      );
    } catch (err) {
      if (!(err instanceof playwrightErrors.TimeoutError)) throw err;
    }
    await page.waitForTimeout(300);
    const deleted = stub.requests().find((r) => r.method === "DELETE");
    results.check(
      "item7: confirm dispatched DELETE for the right resource",
      deleted?.path === "/api/nodes/local/engines/gguf-test",
      JSON.stringify(deleted ?? null),
    );

    // R14 (controller ruling): restore the blanket console-errors check
    // Task 6 dropped. Task 6's reason was real (sparky's swap-probe effect
    // fired two unscripted routes and 599'd) but was a FIXTURE gap, not a
    // reason to drop the assertion — spec §7 makes console errors an
    // assertion so a runtime failure cannot pass silently, and every later
    // gate built on this fixture would otherwise inherit the blind spot.
    // Closed here instead: fixtures/e1-seeded-triple/scenario.json now
    // scripts GET /api/nodes/sparky/serving/status (repeat:true — polled
    // every refreshState, same cadence as /api/state) and GET
    // /api/settings/catalog/sparky/vllm (single entry — the catalog probe
    // effect is keyed on the swap-node id list, which is "sparky" from the
    // very first /api/state response and never changes, so it fires
    // exactly once). The console is now genuinely clean, so the blanket
    // check applies with no named allowlist needed.
    // Named distinctly from smoke.gate.mjs's own "no console errors" check:
    // run.mjs's cross-gate duplicate-name guard (Task 4/6) refuses two gates
    // sharing a check name — caught on first run of the full suite, exactly
    // the FAIL-hiding-behind-another-gate's-PASS failure mode it exists for.
    results.check(
      "e1-engines: no console errors",
      consoleErrors.length === 0,
      consoleErrors.join(" | "),
    );
  } finally {
    await browser.close();
    await stub.stop();
  }
  return results;
}
