import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { errors as playwrightErrors } from "playwright-core";
import { reportingRun } from "./lib/check.mjs";
import { launch } from "./lib/browser.mjs";
import { isTrulyDisabled, makeAssertUnique, textsOf } from "./lib/dom.mjs";
import { startStub } from "./stub-server.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const name = "e1-instances";

/** INST I1 (Task 12) — the free-GPU-assignment engine-instance surface: the
 * board's "+ add engine here" entry point on an unmanaged card of an
 * instances-capable node, the create-instance form it opens (kind picker,
 * GPU checkbox picker, per-kind env fields, genuine Save-refusal), the
 * dispatched create/move/remove verbs, and the two controller-added checks
 * (10-11): a DECLARED (non-instance) engine's own GPU-picker edit path, and
 * the node form's three-way control `<select>`.
 *
 * Fixture: `fixtures/inst-partial/scenario.json` — local node "nimbus",
 * GPUs 2/3/4, ONE declared resource `gguf-a` (lemonade) claiming [2,3], so
 * gpu4 is the fixture's lone unmanaged card (see the fixture's own `_note`
 * for the full causal-`/api/nodes` script this gate's clicks walk through).
 *
 * Selector table, read out of ui/src/components/ResourcePanel.tsx,
 * ui/src/components/NodeCard.tsx, ui/src/components/NodesView.tsx and
 * ui/src/model/nodes.ts at the commit that added INST I1's Task 10/11 UI
 * (labels/messages confirmed in ui/src/model/messages.ts):
 *
 * | What                              | Selector / text                                          |
 * |------------------------------------|-----------------------------------------------------------|
 * | GPU card (deck/board view)         | `.resource-panel` (one per GPU, title `GPU <index>`)      |
 * | Unmanaged marker                   | `.resource-panel .unmanaged-tag` (text "unmanaged")       |
 * | Board's create-instance entry point| `.resource-panel .add-engine-here` (text                  |
 * |                                     | "+ add engine here" — ONLY on an unmanaged card of an     |
 * |                                     | instances-capable node, ResourcePanel.tsx's own gate)     |
 * | Nodes tab                          | `.view-tabs button:has-text("Nodes")`                     |
 * | Local node rail row                | `.nodes-row:has-text("nimbus")`                            |
 * | Node form (rail row's own pane)    | `.nodes-form`                                              |
 * | Node form's control picker         | `.nodes-form label:has-text("Control") select`             |
 * | Engines section                    | `.engines-section`                                         |
 * | Declared-engine rows               | `.engines-list .engine-row`                                |
 * | Row fields                         | `.engine-row-resource`, `.engine-row-kind`,                |
 * |                                     | `.engine-row-gpu` (text `GPU <claim.join("+")>`)           |
 * | Managed (instance) marker          | `.engine-row .instance-tag` (text "instance")              |
 * | Managed row's host-port caption    | `.engine-row-port` (text `host port <n>`)                  |
 * | Declared-row Edit button           | `.engine-row:has-text(RESOURCE) button:has-text("Edit")`   |
 * | Managed-row Move (plain toggle)    | `.engine-row:has-text(RESOURCE) button:text-is("Move")`    |
 * | Managed-row Move (ArmedButton)     | `.engine-row:has-text(RESOURCE) .armed-wrap                |
 * |                                     | button:text-is("⚠ Move")` — same instance arms then confirms|
 * | Managed-row Remove (ArmedButton)   | `.engine-row:has-text(RESOURCE) .armed-wrap                |
 * |                                     | button:text-is("⚠ Remove")`                                |
 * | GPU checkbox picker (shared class) | `.engine-gpu-picker input[type=checkbox]` — rendered by    |
 * |                                     | InstanceFormPanel, EngineFormPanel, AND EngineRow's own    |
 * |                                     | Move fieldset; only ever ONE is mounted at a time in this  |
 * |                                     | gate's own click sequence (verified per-step below)        |
 * | Create-instance form                | `.instance-form` (== `.engine-form.instance-form`)         |
 * | Instance form's kind select         | `.instance-form label:has-text("Kind") select`             |
 * | Instance form's env input           | `.instance-form label:has-text(ENV_NAME) input`            |
 * | Instance form Save/Cancel           | `.instance-form .engine-form-actions .primary`             |
 * |                                     | (text "+ Create instance"), `button:text-is("Cancel")`     |
 *
 * R4 (controller ruling): `:has-text()` substring-matches, case-insensitively.
 * Two DISTINCT collisions this file ran into and fixed, both found only by
 * running the gate and reading the "found 2" refusal `assertUnique` throws
 * (not by inspection — the second one especially is easy to miss on sight):
 *   1. "Move" is a substring of the ArmedButton's own rendered text
 *      ("⚠ Move"), so the plain toggle button (exact text "Move") is
 *      selected with `:text-is()`, never `:has-text()` — same collision
 *      precedent e1-board.gate.mjs's Force-park sequence documents for
 *      "Park" vs. "Force park".
 *   2. "Move" is ALSO a substring of "Remove" itself ("re-MOVE") — so
 *      `.armed-wrap button:has-text("Move")` matches BOTH the row's Move
 *      AND Remove ArmedButtons the instant both are on screen (as soon as
 *      Move is toggled open, both fieldsets/buttons render together). Every
 *      Move/Remove ArmedButton selector below therefore matches the
 *      button's exact rendered text — `:text-is("⚠ Move")` /
 *      `:text-is("⚠ Remove")` — never a bare `:has-text()`.
 * "HIPFIRE_MODEL" and "HIPFIRE_IDLE_TIMEOUT" share no substring relationship
 * in either direction, so their env-label selectors need no such guard. */

const assertUnique = makeAssertUnique("e1-instances");

export async function run() {
  return reportingRun(async (results) => {
    const scenario = JSON.parse(
      await readFile(join(HERE, "fixtures/inst-partial/scenario.json"), "utf8"),
    );
    const stub = await startStub({ scenario, distDir: join(HERE, "../dist"), port: 0 });
    const { browser, page, consoleErrors } = await launch();
    try {
      await page.goto(stub.url, { waitUntil: "networkidle" });

      // --------------------------------------------------------------
      // Items 1-2 — the default "deck" view. gguf-a's [2,3] claim leaves
      // gpu4 as the fixture's ONE unmanaged card (the controller's own
      // item-1/2 revision — see the fixture's _note for why [2] alone
      // would have left two).
      // --------------------------------------------------------------
      await page.waitForSelector(".resource-panel");
      const cardCount = await page.locator(".resource-panel").count();
      results.check(
        "inst item1: exactly 3 resource-panel cards render (gpu2, gpu3, gpu4)",
        cardCount === 3,
        String(cardCount),
      );
      const unmanagedCount = await page.locator(".resource-panel .unmanaged-tag").count();
      results.check(
        "inst item1: exactly one card is tagged unmanaged (gpu4 — gguf-a's [2,3] claim leaves gpu2/gpu3 declared)",
        unmanagedCount === 1,
        String(unmanagedCount),
      );

      const addHereCount = await page.locator(".resource-panel .add-engine-here").count();
      results.check(
        'inst item2: "+ add engine here" renders exactly once for this fixture',
        addHereCount === 1,
        String(addHereCount),
      );
      await assertUnique(page, '.resource-panel:has-text("GPU 4")', "GPU 4's card");
      const addHereOnGpu4 = await page
        .locator('.resource-panel:has-text("GPU 4") .add-engine-here')
        .count();
      results.check(
        'inst item2: "+ add engine here" renders specifically on GPU 4\'s card (the lone unmanaged one), not gpu2/gpu3',
        addHereOnGpu4 === 1,
        String(addHereOnGpu4),
      );

      // --------------------------------------------------------------
      // Item 3 — clicking it opens the Nodes screen with the instance form,
      // GPU 4 pre-checked. App.tsx's onAddEngineHere sets instanceSeed and
      // switches view in the same click; NodesView selects the seed's node
      // and EnginesSection consumes the seed once per mount — no second
      // click needed to land on the form.
      // --------------------------------------------------------------
      await page.click('.resource-panel:has-text("GPU 4") .add-engine-here');
      await page.waitForSelector(".instance-form");
      const nodesTabClass = await page
        .locator('.view-tabs button:has-text("Nodes")')
        .getAttribute("class");
      results.check(
        "inst item3: the click switched to the Nodes screen (Nodes tab is now the active/primary one)",
        nodesTabClass === "primary",
        String(nodesTabClass),
      );

      const item3Checks = await page.$$eval(
        ".instance-form .engine-gpu-picker input[type=checkbox]",
        (els) => els.map((e) => e.checked),
      );
      results.check(
        "inst item3: GPU 4 pre-checked, GPU 2/3 not ([false,false,true] in world.gpus order)",
        JSON.stringify(item3Checks) === JSON.stringify([false, false, true]),
        JSON.stringify(item3Checks),
      );

      // --------------------------------------------------------------
      // Item 4 — the hipfire-kind form (the instance kind picker's default,
      // available[0] === "hipfire" per instanceKindsFor's own local-capable
      // + instance filter) refuses Save until HIPFIRE_MODEL is typed.
      // --------------------------------------------------------------
      await assertUnique(page, '.instance-form .engine-form-actions .primary', "Create-instance Save button");
      results.check(
        "inst item4: Save is genuinely disabled until HIPFIRE_MODEL is typed",
        await isTrulyDisabled(page, ".instance-form .engine-form-actions .primary"),
      );
      await assertUnique(page, '.instance-form label:has-text("HIPFIRE_MODEL")', "HIPFIRE_MODEL env label");
      await page.fill('.instance-form label:has-text("HIPFIRE_MODEL") input', "qwen-instance-test");
      results.check(
        "inst item4: Save is enabled once HIPFIRE_MODEL is filled",
        !(await isTrulyDisabled(page, ".instance-form .engine-form-actions .primary")),
      );

      // --------------------------------------------------------------
      // Item 5 — Create dispatches the POST with exactly what was typed.
      // --------------------------------------------------------------
      await page.click(".instance-form .engine-form-actions .primary");
      await page.waitForSelector(".instance-form", { state: "detached" });
      const created = stub
        .requests()
        .find((r) => r.method === "POST" && r.path === "/api/nodes/local/instances");
      results.check(
        "inst item5: Create dispatched POST /api/nodes/local/instances with {kind:hipfire, gpu_indices:[4], env:{HIPFIRE_MODEL}}",
        created?.body?.kind === "hipfire" &&
          JSON.stringify(created?.body?.gpu_indices) === JSON.stringify([4]) &&
          created?.body?.env?.HIPFIRE_MODEL === "qwen-instance-test",
        JSON.stringify(created ?? null),
      );

      // --------------------------------------------------------------
      // Item 6 — the causal /api/nodes refetch (afterCreate's reload())
      // renders hipfire-1 as a managed "instance" row on GPU 4.
      // --------------------------------------------------------------
      await page.waitForSelector('.engine-row:has-text("hipfire-1")');
      await assertUnique(page, '.engine-row:has-text("hipfire-1") .instance-tag', "hipfire-1's instance tag");
      const hipfireGpuText = await page
        .locator('.engine-row:has-text("hipfire-1") .engine-row-gpu')
        .innerText();
      results.check(
        "inst item6: hipfire-1 renders as an instance row on GPU 4",
        hipfireGpuText === "GPU 4",
        hipfireGpuText,
      );

      // --------------------------------------------------------------
      // Item 10 (controller-added) — editing the DECLARED (non-managed)
      // gguf-a shows its real [2,3] claim in the shared GPU-picker fieldset,
      // and Cancel dispatches nothing. Deliberately run here, between items
      // 6 and 7, while only ONE `.engine-gpu-picker` is mounted (gguf-a's
      // own Edit form) — hipfire-1's `moving` state is still false, so its
      // Move fieldset (same class name) has not rendered yet.
      // --------------------------------------------------------------
      await page.click('.engine-row:has-text("gguf-a") button:has-text("Edit")');
      await page.waitForSelector(".engine-form:not(.instance-form)");
      const item10Checks = await page.$$eval(
        ".engine-form .engine-gpu-picker input[type=checkbox]",
        (els) => els.map((e) => e.checked),
      );
      results.check(
        "inst item10: editing gguf-a shows its real [2,3] claim checked ([true,true,false] in world.gpus order), gpu4 unchecked",
        JSON.stringify(item10Checks) === JSON.stringify([true, true, false]),
        JSON.stringify(item10Checks),
      );
      // Counts MUTATIONS only (method !== GET), not the raw request total —
      // /api/state and /api/storage/state poll every 3000ms in the
      // background regardless of anything this gate clicks, so a raw count
      // could differ across the click for a reason that has nothing to do
      // with Cancel.
      const mutationsBeforeCancel = stub.requests().filter((r) => r.method !== "GET").length;
      await assertUnique(page, ".engine-form .engine-form-actions button:text-is(\"Cancel\")", "gguf-a edit form's Cancel button");
      await page.click(".engine-form .engine-form-actions button:text-is(\"Cancel\")");
      await page.waitForSelector(".engine-form", { state: "detached" });
      const mutationsAfterCancel = stub.requests().filter((r) => r.method !== "GET").length;
      results.check(
        "inst item10: Cancel dispatches no request",
        mutationsAfterCancel === mutationsBeforeCancel,
        `before=${mutationsBeforeCancel} after=${mutationsAfterCancel}`,
      );

      // --------------------------------------------------------------
      // Item 7 — Move: arming dispatches nothing, confirming dispatches the
      // POST with the picker's new claim. max_gpus 1 REPLACES on toggle
      // (toggleIndex, engineForm.ts), so checking GPU 3 while GPU 4 was the
      // only prior claim leaves GPU 3 on and GPU 4 off, not both on.
      // --------------------------------------------------------------
      await assertUnique(page, '.engine-row:has-text("hipfire-1") button:text-is("Move")', "hipfire-1's Move toggle");
      await page.click('.engine-row:has-text("hipfire-1") button:text-is("Move")');
      await assertUnique(page, '.engine-row:has-text("hipfire-1") .engine-gpu-picker', "hipfire-1's Move GPU-picker fieldset (only one .engine-gpu-picker mounted at this point)");
      // GPU 3 is the SECOND checkbox in world.gpus order [2,3,4].
      await page.locator('.engine-row:has-text("hipfire-1") .engine-gpu-picker input[type=checkbox]').nth(1).click();
      const moveChecks = await page.$$eval(
        '.engine-row:has-text("hipfire-1") .engine-gpu-picker input[type=checkbox]',
        (els) => els.map((e) => e.checked),
      );
      results.check(
        "inst item7: checking GPU 3 in the Move picker checks 3 and unchecks 4 (max_gpus 1 replaces, not accumulates)",
        JSON.stringify(moveChecks) === JSON.stringify([false, true, false]),
        JSON.stringify(moveChecks),
      );

      await assertUnique(page, '.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Move")', "hipfire-1's Move ArmedButton");
      await page.click('.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Move")');
      // Scoped to the Move ArmedButton's OWN `.armed-wrap` (`:has(button:
      // has-text("Move"))`), not a bare `.armed-hint` — armed-hint's own
      // text (messages.forceConfirm) is generic, shared by every
      // ArmedButton on the page, so a bare selector would be ambiguous once
      // Remove is armed too (item 8, below): EngineRow's `moving`/`moveArmed`
      // local state is never reset after a SUCCESSFUL move (only a refusal
      // increments moveRefusalSeq — model/armed.ts's own isArmedFor), so the
      // Move picker and its armed hint stay on screen for the rest of this
      // row's life once armed here.
      await page.waitForSelector('.engine-row:has-text("hipfire-1") .armed-wrap:has(button:text-is("⚠ Move")) .armed-hint');
      const moveAfterArm = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/nodes/local/instances/hipfire-1/move");
      results.check(
        "inst item7: arming Move (first click) dispatches no request",
        moveAfterArm.length === 0,
        String(moveAfterArm.length),
      );

      await page.click('.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Move")');
      await page.waitForTimeout(300);
      const moveAfterConfirm = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/nodes/local/instances/hipfire-1/move");
      results.check(
        "inst item7: confirming armed Move (second click, same instance) dispatches POST .../hipfire-1/move {gpu_indices:[3]}",
        moveAfterConfirm.length === 1 &&
          JSON.stringify(moveAfterConfirm[0].body?.gpu_indices) === JSON.stringify([3]),
        JSON.stringify(moveAfterConfirm),
      );

      // Fix round 1 (INST I1 T12 controller ruling) — a real component
      // defect the RED proof below caught: EngineRow.doMove's success path
      // used to call onForgotten() without disarming first. UNLIKE
      // doForget/doRemove, a successful move does NOT unmount this row (the
      // resource keeps its name/key across a move, D-I1-1) — so
      // `moveArmedForSeq`/`moveRefusalSeq`, which only ever change on a
      // REFUSAL (model/armed.ts's `isArmedFor`), stayed exactly as armed as
      // the operator's own second click left them, and the picker never
      // closed. Waits for the CAUSAL refetch to land first (the row's own
      // GPU text becoming "GPU 3"), then asserts the Move ArmedButton's own
      // armed-hint is genuinely gone — not a vacuous absence check: this
      // exact selector already matched (and was waited on) earlier in this
      // same run, right after arming Move, above, so it is proven able to
      // find something before this check leans on it finding nothing.
      await page.waitForSelector('.engine-row:has-text("hipfire-1") .engine-row-gpu:text-is("GPU 3")');
      const moveHintAfterRefetch = await page
        .locator('.engine-row:has-text("hipfire-1") .armed-wrap:has(button:text-is("⚠ Move")) .armed-hint')
        .count();
      results.check(
        "inst item7: the Move ArmedButton is disarmed after the causal refetch (a successful move must not leave it stuck armed)",
        moveHintAfterRefetch === 0,
        String(moveHintAfterRefetch),
      );

      // --------------------------------------------------------------
      // Item 8 — Remove: arms then dispatches DELETE; the row is gone after
      // the causal refetch (afterMutate's reload()).
      // --------------------------------------------------------------
      await assertUnique(page, '.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Remove")', "hipfire-1's Remove ArmedButton");
      await page.click('.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Remove")');
      // Scoped the same way item 7's own wait is, and for the same reason:
      // Move's ArmedButton is left stuck "armed" after item 7's successful
      // confirm (see that item's own comment), so a bare `.armed-hint`
      // would match two elements once Remove is armed here too.
      await page.waitForSelector('.engine-row:has-text("hipfire-1") .armed-wrap:has(button:text-is("⚠ Remove")) .armed-hint');
      const removeAfterArm = stub
        .requests()
        .filter((r) => r.method === "DELETE" && r.path === "/api/nodes/local/instances/hipfire-1");
      results.check(
        "inst item8: arming Remove (first click) dispatches no request",
        removeAfterArm.length === 0,
        String(removeAfterArm.length),
      );

      await page.click('.engine-row:has-text("hipfire-1") .armed-wrap button:text-is("⚠ Remove")');
      // Bounded, not the 30s locator default: a mutation firing doRemove on
      // the first click would already have removed the row by now, and a
      // broken confirm would otherwise hang here waiting for a detach that
      // is never coming — same rationale e1-engines.gate.mjs's own Forget
      // sequence documents for this exact shape.
      let removedInTime = true;
      try {
        await page.waitForSelector('.engine-row:has-text("hipfire-1")', { state: "detached", timeout: 3000 });
      } catch (err) {
        if (!(err instanceof playwrightErrors.TimeoutError)) throw err;
        removedInTime = false;
      }
      const removeAfterConfirm = stub
        .requests()
        .filter((r) => r.method === "DELETE" && r.path === "/api/nodes/local/instances/hipfire-1");
      results.check(
        "inst item8: confirming armed Remove dispatches DELETE /api/nodes/local/instances/hipfire-1",
        removeAfterConfirm.length === 1,
        String(removeAfterConfirm.length),
      );
      results.check(
        "inst item8: hipfire-1's row is gone after the causal refetch",
        removedInTime,
        String(removedInTime),
      );

      // --------------------------------------------------------------
      // Item 9 — instances never go through the declare-only route. Paired
      // with a positive claim (requests().length > 0) per the README's
      // no-vacuous-negations rule: an empty capture list would make the
      // absence check pass for the wrong reason.
      // --------------------------------------------------------------
      const allRequests = stub.requests();
      const declareRouteHits = allRequests.filter((r) => r.path === "/api/nodes/local/engines");
      results.check(
        "inst item9: no request this run targeted /api/nodes/local/engines (instances never go through the declare-only route)",
        allRequests.length > 0 && declareRouteHits.length === 0,
        `totalRequests=${allRequests.length} declareRouteHits=${JSON.stringify(declareRouteHits)}`,
      );

      // --------------------------------------------------------------
      // Item 11 (controller-added) — the node form's control <select>
      // offers exactly the three declared operabilities and reads
      // "instances" for this fixture's local entry. Read-only: no click,
      // no dispatch.
      // --------------------------------------------------------------
      await assertUnique(page, '.nodes-form label:has-text("Control") select', "the node form's control select");
      const controlOptions = await textsOf(page, '.nodes-form label:has-text("Control") select option');
      results.check(
        'inst item11: the control select offers exactly "none", "swap (serving slot)", "instances (deck-created engines)"',
        JSON.stringify(controlOptions) ===
          JSON.stringify(["none", "swap (serving slot)", "instances (deck-created engines)"]),
        controlOptions.join(","),
      );
      const controlValue = await page.locator('.nodes-form label:has-text("Control") select').inputValue();
      results.check(
        'inst item11: the control select\'s value is "instances" for this fixture\'s local entry (unchanged)',
        controlValue === "instances",
        controlValue,
      );

      // R14 (controller ruling): blanket console-errors is the default.
      results.check(
        "e1-instances: no console errors",
        consoleErrors.length === 0,
        consoleErrors.join(" | "),
      );
    } finally {
      await browser.close();
      await stub.stop();
    }
  });
}
