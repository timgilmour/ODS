import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createResults } from "./lib/check.mjs";
import { launch } from "./lib/browser.mjs";
import { isTrulyDisabled, textsOf } from "./lib/dom.mjs";
import { startStub } from "./stub-server.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const name = "e1-board";

/** E1 items 9-11: the Set Builder's dynamic per-resource board — one draft
 * card, per-kind rows inside it, a working model-chip drop confined to the
 * one load-verb resource's card, the 409 overwrite-lockdown invariant, and
 * durable (survives-a-reload) model removal.
 *
 * Selector table, read out of ui/src/components/SetBuilder.tsx at bfa75274
 * (labels/messages confirmed in ui/src/model/messages.ts; supporting reads:
 * ui/src/model/setDraft.ts, ui/src/model/nodes.ts, ui/src/components/
 * ModelLibrary.tsx, ui/src/components/SavedSets.tsx, ui/src/ui/Panel.tsx,
 * ui/src/ui/Banner.tsx):
 *
 * | What                        | Selector / text                                          |
 * |-----------------------------|-----------------------------------------------------------|
 * | Set Builder tab              | `.view-tabs button:has-text("Set Builder")`               |
 * | Draft node card               | `.builder-node` (a `Panel`; exactly ONE ever renders)      |
 * | Per-resource card             | `.builder-gpu` (one per DECLARED resource, sorted by       |
 * |                                | gpu_index then name — `sortedResourceEntries`)             |
 * | Resource name inside a card   | `.tenant-name`                                             |
 * | The one load-verb card's drop target | `.builder-gpu.set-builder-drop` (only the resource |
 * |                                | `pickLoadResource` selects — lemonade in this fixture —    |
 * |                                | ever gets `onDragOver`/`onDrop` wired at all)               |
 * | Empty-card dropzone           | `.builder-dropzone` (text `labels.dropToAssign`)           |
 * | Placed-model chip             | `.tenant-meta` (filename span + GB span), `.durable-chip`  |
 * |                                | when a durable route survives with no derivable filename   |
 * | "remove model" button         | `button:has-text("remove model")` (lowercase, verbatim)    |
 * | Model library row (drag src)  | `.model-library-row`                                       |
 * | Draft name input (no OWN      | `.builder-field input` (the "name" field is the only       |
 * |   disabled prop — see R5/R6)  | `<input>` under `.builder-field`; "notes" is a textarea)   |
 * | Save draft                    | `.builder-actions button.primary` (text `Save draft`)      |
 * | Cancel (resets the draft)     | `.builder-actions button:has-text("Cancel")` (exact)       |
 * | Overwrite confirm banner      | `.ui-banner button:has-text("Overwrite")`                  |
 * | Saved-set row / Load button   | `.saved-set-row:has-text(NAME) button:has-text("Load")`    |
 *
 * ⭐R5 (controller ruling) finding, recorded rather than invented into a
 * check: item 9's checklist wording is "one card per node" (plural-shaped).
 * Reading the component shows this is not what it does, and is not what it
 * is FOR — SetBuilder's own header comment states outright that `ConfigSet`
 * "has one leg per DECLARED resource ... there is no spark leg — so the
 * builder drafts the local node and nothing else. A second node card here
 * would be a control for a set field that does not exist." There is exactly
 * ONE draft card, ever, standing for the local node; what genuinely varies
 * "per node"-shaped is the ROW inside that one card, one per declared
 * resource (which is also one per kind, since a resource's rendering
 * branches on `tenant.engine`). The checks below assert the true claim
 * (exactly one `.builder-node`, N per-resource rows inside it) rather than
 * inventing a "per node" check with nothing behind it — R5's own instruction
 * for exactly this situation.
 *
 * R4 (controller ruling): `:has-text()` substring-matches. The three
 * declared resource names (hipfire/lemonade/comfyui) never collide with
 * each other or with any other on-screen text this gate reads, but every
 * selector built from one is still run through `assertUnique` before use,
 * rather than trusted by inspection alone. */

async function assertUnique(page, selector, what) {
  const n = await page.locator(selector).count();
  if (n !== 1) {
    throw new Error(
      `e1-board gate: expected exactly 1 ${what} (selector ${selector}), found ${n}. ` +
        `A selector that matches zero elements can make a check pass vacuously; one ` +
        `that matches more than expected can assert against the wrong element.`,
    );
  }
}

export async function run() {
  const results = createResults();
  const scenario = JSON.parse(
    await readFile(join(HERE, "fixtures/e1-board/scenario.json"), "utf8"),
  );
  const stub = await startStub({ scenario, distDir: join(HERE, "../dist"), port: 0 });
  const { browser, page, consoleErrors } = await launch();
  try {
    await page.goto(stub.url, { waitUntil: "networkidle" });

    await assertUnique(page, '.view-tabs button:has-text("Set Builder")', "Set Builder tab button");
    await page.click('.view-tabs button:has-text("Set Builder")');
    await page.waitForSelector(".builder-node");

    // Item 9 (part 1) — exactly one draft card, one row per declared
    // resource, per kind (see R5 finding in the header comment above: this
    // is the TRUE claim behind the checklist's "one card per node" wording).
    const cards = await page.locator(".builder-node").count();
    results.check("item9: exactly one draft node card renders", cards === 1, String(cards));

    const names = await textsOf(page, ".builder-gpu .tenant-name");
    results.check(
      "item9: one row per declared resource, per kind (hipfire, comfyui, lemonade — sortedResourceEntries order)",
      JSON.stringify(names) === JSON.stringify(["hipfire", "comfyui", "lemonade"]),
      names.join(","),
    );

    await assertUnique(page, '.builder-gpu:has-text("lemonade")', "the lemonade (load-verb) card");
    await assertUnique(page, '.builder-gpu:has-text("hipfire")', "the hipfire (non-load-verb) card");

    // Item 9 (part 2) — empty-card state, before any drop.
    const emptyBefore = await page
      .locator('.builder-gpu:has-text("lemonade") .builder-dropzone')
      .count();
    results.check(
      "item9: empty-card state renders before any drop",
      emptyBefore === 1,
      String(emptyBefore),
    );

    await assertUnique(page, ".model-library-row", "the model library row");

    // Item 10 (part 1) — drop guard confined to a single card. Dropping the
    // SAME model chip on the WRONG card (hipfire, a non-load-verb resource
    // — `KIND_DRAFT_SPEC.hipfire.supportsModel` is false) must place
    // nothing anywhere. This is checked on the LOAD-RESOURCE card
    // (lemonade), not on the drop target itself: `handleDrop` always
    // resolves to the single `loadResource` regardless of which DOM node
    // the event fired on (SetBuilder.tsx:249-257's `placeModel`), so if the
    // `isLoadResource` gate wrapping `onDragOver`/`onDrop` were ever removed
    // (wiring drop handlers onto every card), a drop on the WRONG card
    // would still relocate the model onto the RIGHT one — this check is
    // aimed at exactly that failure, not at the (structurally impossible
    // either way) idea of the model appearing on the hipfire card itself.
    await page.locator(".model-library-row").dragTo(page.locator('.builder-gpu:has-text("hipfire")'));
    const stillEmptyAfterWrongDrop = await page
      .locator('.builder-gpu:has-text("lemonade") .builder-dropzone')
      .count();
    results.check(
      "item10: dropping on a non-load-resource card places nothing (drop guard confined to a single card)",
      stillEmptyAfterWrongDrop === 1,
      String(stillEmptyAfterWrongDrop),
    );

    // Item 9 (part 3) — the correct-card drop actually works.
    await page.locator(".model-library-row").dragTo(page.locator('.builder-gpu:has-text("lemonade")'));
    await page.waitForSelector('.builder-gpu:has-text("lemonade") .tenant-meta');
    const placedText = await page
      .locator('.builder-gpu:has-text("lemonade") .tenant-meta')
      .innerText();
    results.check(
      "item9: model-chip drop onto the correct card works",
      placedText.includes("test-model.gguf"),
      placedText,
    );
    const dropzoneGoneAfterPlace = await page
      .locator('.builder-gpu:has-text("lemonade") .builder-dropzone')
      .count();
    results.check(
      "item9: dropzone is replaced by the placed-model chip",
      dropzoneGoneAfterPlace === 0,
      String(dropzoneGoneAfterPlace),
    );

    // Discard this ad-hoc placement — it was never saved — so item 11
    // starts from a clean, unsaved draft. `resetDraft()` via Cancel.
    await assertUnique(page, '.builder-actions button:has-text("Cancel")', "the draft Cancel button");
    await page.click('.builder-actions button:has-text("Cancel")');
    await page.waitForSelector('.builder-gpu:has-text("lemonade") .builder-dropzone');

    // Item 11 (part 1) — load a SAVED set that already carries a placed
    // model, and prove the POSITIVE round-trip first: `populateFromSet` /
    // `derivePlacedModel` (setDraft.ts) is a DIFFERENT code path than the
    // drag-drop `placeModel` just exercised above, and without this
    // positive proof the later "no ghost" check would risk passing
    // vacuously if Load never restored a model at all, working or not.
    await page.waitForSelector('.saved-set-row:has-text("with-model")');
    await assertUnique(
      page,
      '.saved-set-row:has-text("with-model") button:has-text("Load")',
      "the with-model row's Load button",
    );
    await page.click('.saved-set-row:has-text("with-model") button:has-text("Load")');
    await page.waitForSelector('.builder-gpu:has-text("lemonade") .tenant-meta');
    const loadedText = await page
      .locator('.builder-gpu:has-text("lemonade") .tenant-meta')
      .innerText();
    results.check(
      "item11: loading a saved set with a placed model restores the chip",
      loadedText.includes("test-model.gguf"),
      loadedText,
    );

    // Remove the model, then re-save over the same slug — this is a real
    // collision (the set was just Loaded from a name that already exists on
    // the fixture's GET /api/sets), so the first Save attempt (overwrite
    // implicitly false — `onClick={() => handleSave(false)}`) 409s for
    // real. This IS item 10's fieldset-lockdown proof: no throwaway attempt
    // is needed, it is the natural first half of this exact save.
    await assertUnique(
      page,
      '.builder-gpu:has-text("lemonade") button:has-text("remove model")',
      "the remove-model button",
    );
    await page.click('.builder-gpu:has-text("lemonade") button:has-text("remove model")');
    await assertUnique(page, ".builder-actions button.primary", "Save draft button");
    await page.click(".builder-actions button.primary");
    await page.waitForSelector('.ui-banner button:has-text("Overwrite")');

    // Item 10 (part 2) — 409 locks the fieldset. THE ancestor-fieldset
    // case: `.builder-field input` (the draft name field) carries no
    // `disabled` prop of its own anywhere in SetBuilder.tsx — any disabled
    // state on it can only come from the ancestor `<fieldset disabled>`,
    // which is exactly what `isTrulyDisabled`'s `:disabled` match (not a
    // raw `.disabled`/`getAttribute` read) is built to see.
    await assertUnique(page, ".builder-field input", "the draft name input");
    results.check(
      "item10: 409 locks the fieldset (ancestor-fieldset case, isTrulyDisabled)",
      await isTrulyDisabled(page, ".builder-field input"),
    );

    await assertUnique(page, '.ui-banner button:has-text("Overwrite")', "the Overwrite banner action");
    await page.click('.ui-banner button:has-text("Overwrite")');
    await page.waitForTimeout(300);

    // Item 11 (part 2, DISPATCH) — the stub NEVER derives state (design §5):
    // its GET /api/sets responses are hand-authored fixture data, not a
    // reflection of what this gate's own POST actually carried. That means
    // the render check below (part 3) — "reload, Load, no ghost chip" — can
    // only ever prove SetBuilder renders correctly whatever ConfigSet it is
    // GIVEN; it structurally CANNOT catch a `removeModel()` regression that
    // fails to clear something before Save (e.g. forgetting `setDurable
    // (null)`), because the fixture's post-reload GET response was authored
    // by this gate to already read "removed" regardless. Mirrors item 4's
    // dispatch/render split (Task 7, R12): read the actual POST body off
    // the wire instead and assert the removal really was what got sent.
    const overwritePosts = stub.requests().filter((r) => r.method === "POST" && r.path === "/api/sets");
    const savedAfterRemoval = overwritePosts[overwritePosts.length - 1];
    results.check(
      "item11: the removal is actually POSTed (durable cleared, resource no longer 'loaded')",
      savedAfterRemoval?.body?.durable === null &&
        savedAfterRemoval?.body?.ephemeral?.resources?.lemonade?.desired !== "loaded",
      JSON.stringify(savedAfterRemoval ?? null),
    );

    // Item 11 (part 3, RENDER) — reload the page (a genuine navigation, not the
    // 3000ms poll — see the fixture's own _note on why GET /api/sets is
    // safe to script a transition on) and load the SAME set again. No
    // ghost: the removal must have been durably saved, not merely cleared
    // in local component state.
    await page.reload({ waitUntil: "networkidle" });
    await assertUnique(page, '.view-tabs button:has-text("Set Builder")', "Set Builder tab button (post-reload)");
    await page.click('.view-tabs button:has-text("Set Builder")');
    await page.waitForSelector(".builder-node");
    await page.waitForSelector('.saved-set-row:has-text("with-model")');
    await assertUnique(
      page,
      '.saved-set-row:has-text("with-model") button:has-text("Load")',
      "the with-model row's Load button (post-reload)",
    );
    await page.click('.saved-set-row:has-text("with-model") button:has-text("Load")');
    await page.waitForTimeout(300);
    const ghostChip = await page
      .locator('.builder-gpu:has-text("lemonade") .tenant-meta, .builder-gpu:has-text("lemonade") .durable-chip')
      .count();
    const dropzoneBack = await page
      .locator('.builder-gpu:has-text("lemonade") .builder-dropzone')
      .count();
    results.check(
      "item11: removing a model from a draft clears it durably (no ghost after refresh)",
      ghostChip === 0 && dropzoneBack === 1,
      `ghostChip=${ghostChip} dropzoneBack=${dropzoneBack}`,
    );

    // R14 (controller ruling): blanket console-errors is the default; a
    // NAMED allowlist is the only permitted fallback, never a blanket drop.
    // This gate's ONE deliberate exception: Chrome logs a console "error"
    // for ANY non-2xx fetch response purely because the HTTP status was
    // non-OK — not because a script threw. Item 10/11's Save-then-409 is
    // real, DESIGNED application behaviour (SetBuilder.tsx's `handleSave`
    // specifically catches `ApiError` with `status === 409` to drive the
    // Overwrite confirmation), not a runtime failure, so it is named here
    // rather than silently swallowed with everything else a blanket drop
    // would also hide.
    const ALLOWED_CONSOLE_ERRORS = new Set([
      "Failed to load resource: the server responded with a status of 409 (Conflict)",
    ]);
    const unexpectedConsoleErrors = consoleErrors.filter((e) => !ALLOWED_CONSOLE_ERRORS.has(e));
    results.check(
      "e1-board: no console errors other than the scripted 409 (Chrome logs any non-2xx fetch, including the one this gate deliberately triggers)",
      unexpectedConsoleErrors.length === 0,
      unexpectedConsoleErrors.join(" | "),
    );
  } finally {
    await browser.close();
    await stub.stop();
  }
  return results;
}
