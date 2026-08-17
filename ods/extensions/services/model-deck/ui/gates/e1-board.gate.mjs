import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createResults } from "./lib/check.mjs";
import { launch } from "./lib/browser.mjs";
import { isTrulyDisabled, textsOf } from "./lib/dom.mjs";
import { startStub } from "./stub-server.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const name = "e1-board";

/** E1 items 9-13: the Set Builder's dynamic per-resource board — one draft
 * card, per-kind rows inside it, a working model-chip drop confined to the
 * one load-verb resource's card, the 409 overwrite-lockdown invariant,
 * durable (survives-a-reload) model removal — plus (Task 9) the live Deck
 * board's own per-resource controls: generic (never `/api/spark`) dispatch
 * URLs, the Force-park two-click arm/confirm contract, the cold-GGUF
 * pull-through banner, and ResourcePanel actually rendering verbs (not
 * silently nothing, not SparkSwap) for a non-spark resource.
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
 * Items 12-13 selector table, read out of ui/src/components/ResourcePanel.tsx
 * and ui/src/components/PlacementActions.tsx at 023e02c9 (both LIVE on the
 * default "deck" view — Board -> NodeCard -> ResourcePanel — never the Set
 * Builder tab; supporting reads: ui/src/ui/ArmedButton.tsx, ui/src/ui/
 * Banner.tsx, ui/src/model/armed.ts, ui/src/model/refusals.ts,
 * app/routers/control.py, app/engines/hipfire.py):
 *
 * | What                          | Selector / text                                          |
 * |-------------------------------|-----------------------------------------------------------|
 * | Resource card (Panel)         | `.resource-panel` (one per declared resource; title = the |
 * |                                | resource's own name, via `resource.label`)                 |
 * | Per-resource verb row         | `.tenant-actions` — ⚠SHARED with SparkSwap.tsx, which     |
 * |                                | renders the SAME class. The only DOM fact distinguishing   |
 * |                                | "PlacementActions dispatched" from "SparkSwap dispatched"  |
 * |                                | is `.tenant-name`'s text: SparkSwap hardcodes the literal  |
 * |                                | "spark" (`SPARK_CONTROL`); PlacementActions renders the    |
 * |                                | resource's own name. This fixture declares no swap node    |
 * |                                | at all, so every `.tenant-actions` on screen is provably   |
 * |                                | non-spark only via that name check, never by assumption.   |
 * | hipfire verbs                 | `button:has-text("Park")`, `button:has-text("Resume")`     |
 * | lemonade verbs                | `select[aria-label="model to load"]`,                     |
 * |                                | `button:has-text("Load")`, `button:has-text("Unload")`     |
 * | comfyui verbs                 | `button:has-text("Free")`                                  |
 * | Refusal banner                | `.ui-banner` (`.ui-banner-body` carries the raw backend    |
 * |                                | detail verbatim — `messages.guardRefused`)                 |
 * | Force-park button             | `.armed-wrap button:has-text("Force park")` — ONE shared   |
 * |                                | `ArmedButton`; first click arms (no dispatch), second      |
 * |                                | click on the SAME instance confirms (dispatches). The      |
 * |                                | armed state is NOT a DOM attribute — `armed-hint` (text    |
 * |                                | "Click again to confirm") is the only visible signal, so   |
 * |                                | this item is proved by REQUEST COUNT across the two        |
 * |                                | clicks, not by DOM state.                                  |
 * | Pull-through ("cold") banner  | `.ui-banner:has-text("Model is cold")`,                    |
 * |                                | `button:has-text("Pull + load")`; after confirming,        |
 * |                                | `.ui-banner:has-text("Pulling from cold storage")`         |
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
 * rather than trusted by inspection alone.
 *
 * Item 12 (controller ruling, task brief): mostly a DISPATCH assertion, not
 * a DOM one. `PlacementActions.tsx` carries only `.tenant-actions` /
 * `.tenant-name` — nothing in its markup says "this URL was generic." The
 * real claim (the alias-removal regression pin: E1's tenant verbs dispatch
 * through Task 7's generic `/api/tenants/{resource}/{verb}` route, never the
 * removed `/api/spark/*` routes api.ts:550 documents) is read off
 * `stub.requests()` at the very end of the run, scanning EVERY request the
 * whole run captured (items 9-13 alike) rather than only the ones item 12
 * itself dispatched — a regression that leaked just ONE stray `/api/spark`
 * call from anywhere on the page would still be worth catching here.
 *
 * R12 (controller ruling) for the two routes items 12-13 add: `POST
 * /api/tenants/hipfire/park` and `POST /api/tenants/lemonade/load` are each
 * fetched only on a click (PlacementActions' own button handlers), never on
 * App's 3000ms poll, so a 2-entry click-ordered script (409 then 200) is
 * causal, not clock-satisfiable — the same posture `POST /api/sets` already
 * has in this fixture. The two-entry script matches BOTH the plain and the
 * `?pull=true` retry to `POST /api/tenants/lemonade/load` because the stub
 * keys on `url.pathname` only (stub-server.mjs:53) — a query string carries
 * no route identity there, which is exactly why the alias-removal check
 * above cannot rely on query strings either and reads the recorded `path`
 * instead. */

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
    // R15 (controller ruling): this inner try/catch exists purely so a
    // throw mid-run does not lose the checks already gathered. `results` is
    // local to this function and would otherwise vanish with an uncaught
    // exception — createResults().check() already printed each PASS/FAIL to
    // the console as it ran, but run.mjs (the caller) has no way to recover
    // that once this function throws past its `return results`. Attaching
    // it to the error lets run.mjs still write a real report file instead
    // of nothing, which is exactly the gap that forced this task's original
    // item-9 RED proof into a hand-saved terminal capture.
    try {
      await page.goto(stub.url, { waitUntil: "networkidle" });

      // --------------------------------------------------------------
      // Items 12-13 (Task 9) run FIRST, on the default "deck" view —
      // Board -> NodeCard -> ResourcePanel — before the Set Builder tab
      // click below switches away from it. App.tsx's `view` state starts
      // as "deck" (App.tsx:39), so nothing needs to be clicked to get here.
      // --------------------------------------------------------------

      // Item 13 — ResourcePanel dispatches PlacementActions (verbs), not
      // SparkSwap, for a non-spark resource. This fixture declares no swap
      // node at all (state.nodes: []), so every one of the three declared
      // resources is exactly the case this item is about.
      await page.waitForSelector(".resource-panel");
      const resourcePanelCount = await page.locator(".resource-panel").count();
      results.check(
        "item13: one resource-panel card per declared resource (hipfire, lemonade, comfyui)",
        resourcePanelCount === 3,
        String(resourcePanelCount),
      );
      await assertUnique(page, '.resource-panel:has-text("hipfire")', "hipfire's resource panel");
      await assertUnique(page, '.resource-panel:has-text("lemonade")', "lemonade's resource panel");
      await assertUnique(page, '.resource-panel:has-text("comfyui")', "comfyui's resource panel");

      // `.tenant-actions`/`.tenant-name` are SHARED with SparkSwap.tsx (see
      // header comment). The only DOM fact that tells the two apart is the
      // name text: SparkSwap hardcodes the literal "spark"; PlacementActions
      // renders the resource's own name (PlacementActions.tsx:152, always —
      // not gated on `hasPlacement`). Plain `results.check` here (not
      // `assertUnique`+click — nothing below is clicked off these
      // selectors), so a regression that blanks this row out reads as a
      // clean FAIL rather than a thrown exception.
      for (const resource of ["hipfire", "lemonade", "comfyui"]) {
        const tenantNames = await textsOf(
          page,
          `.resource-panel:has-text("${resource}") .tenant-actions .tenant-name`,
        );
        results.check(
          `item13: ${resource}'s tenant-actions row is named "${resource}" (PlacementActions dispatched, not SparkSwap's literal "spark")`,
          JSON.stringify(tenantNames) === JSON.stringify([resource]),
          tenantNames.join(","),
        );
      }

      // Kind-correct verbs actually render — sourced from GET
      // /api/engine-kinds' human_verbs (park/resume for hipfire, load/unload
      // for lemonade, free for comfyui), never invented in the component.
      const hipfireVerbCount = await page
        .locator('.resource-panel:has-text("hipfire") .tenant-actions button:has-text("Park"), ' +
          '.resource-panel:has-text("hipfire") .tenant-actions button:has-text("Resume")')
        .count();
      results.check("item13: hipfire's Park and Resume buttons both render", hipfireVerbCount === 2, String(hipfireVerbCount));

      const lemonadeVerbCount = await page
        .locator('.resource-panel:has-text("lemonade") .tenant-actions button:has-text("Load"), ' +
          '.resource-panel:has-text("lemonade") .tenant-actions button:has-text("Unload")')
        .count();
      results.check("item13: lemonade's Load and Unload buttons both render", lemonadeVerbCount === 2, String(lemonadeVerbCount));

      const comfyuiVerbCount = await page
        .locator('.resource-panel:has-text("comfyui") .tenant-actions button:has-text("Free")')
        .count();
      results.check("item13: comfyui's Free button renders", comfyuiVerbCount === 1, String(comfyuiVerbCount));

      // Item 12 (part 1) — force-park requires arming. The plain "Park"
      // click dispatches unconditionally and 409s on the host-agent-busy
      // guard (a phrase forceParkCanOverride does NOT fail closed on —
      // model/refusals.ts), which offers Force park. `armed` is not a DOM
      // attribute (model/armed.ts), so this is proved by REQUEST COUNT
      // across two clicks on the SAME ArmedButton instance, not by reading
      // any disabled/armed styling.
      await assertUnique(page, '.resource-panel:has-text("hipfire") .tenant-actions button:has-text("Park")', "hipfire's Park button");
      await page.click('.resource-panel:has-text("hipfire") .tenant-actions button:has-text("Park")');
      await page.waitForSelector('.resource-panel:has-text("hipfire") .ui-banner-body:has-text("host agent is busy")');

      const parkAfterPlainClick = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/tenants/hipfire/park");
      results.check(
        "item12: plain Park dispatches exactly one request and 409s (arms Force park)",
        parkAfterPlainClick.length === 1,
        String(parkAfterPlainClick.length),
      );

      await assertUnique(page, '.resource-panel:has-text("hipfire") .armed-wrap button:has-text("Force park")', "the Force park button");
      await page.click('.resource-panel:has-text("hipfire") .armed-wrap button:has-text("Force park")');
      await page.waitForSelector('.resource-panel:has-text("hipfire") .armed-hint:has-text("Click again to confirm")');

      const parkAfterArmClick = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/tenants/hipfire/park");
      results.check(
        "item12: arming Force park (first click on it) dispatches NO new request — arming, not firing",
        parkAfterArmClick.length === 1,
        String(parkAfterArmClick.length),
      );

      await assertUnique(page, '.resource-panel:has-text("hipfire") .armed-wrap button:has-text("Force park")', "the armed Force park button (2nd click)");
      await page.click('.resource-panel:has-text("hipfire") .armed-wrap button:has-text("Force park")');
      await page.waitForTimeout(300);

      const parkAfterConfirmClick = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/tenants/hipfire/park");
      results.check(
        "item12: confirming the armed Force park (second click, same instance) dispatches the override",
        parkAfterConfirmClick.length === 2,
        String(parkAfterConfirmClick.length),
      );

      // Item 12 (part 2) — pull-through banner. Selecting a resident-but-cold
      // GGUF (App's `coldGgufs`, this fixture's one storage unit on
      // "cold-store", whose declared `engine` is not "lemonade") and
      // clicking Load dispatches the PLAIN load first — it 409s on the
      // backend's cold-model guard — which arms the "Pull + load" banner.
      // Confirming it retries with `?pull=true` and actually loads.
      await assertUnique(page, '.resource-panel:has-text("lemonade") select[aria-label="model to load"]', "lemonade's model-to-load select");
      await page.selectOption('.resource-panel:has-text("lemonade") select[aria-label="model to load"]', "cold-test-model.gguf");
      // `:has-text("Load")` (substring) matches "Unload" too ("Unload"
      // contains "load" case-insensitively) — R4's exact failure mode.
      // `:text-is()` is Playwright's exact-text pseudo-class; used here
      // instead of assertUnique+has-text for that reason.
      await assertUnique(page, '.resource-panel:has-text("lemonade") .tenant-actions button:text-is("Load")', "lemonade's Load button");
      await page.click('.resource-panel:has-text("lemonade") .tenant-actions button:text-is("Load")');
      await page.waitForSelector('.resource-panel:has-text("lemonade") .ui-banner:has-text("Model is cold")');

      const loadAfterColdAttempt = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/tenants/lemonade/load");
      results.check(
        "item12: selecting a cold GGUF and clicking Load dispatches the plain (non-pull) attempt, which 409s and renders the pull-through banner",
        loadAfterColdAttempt.length === 1 && loadAfterColdAttempt[0].body?.model === "cold-test-model.gguf",
        JSON.stringify(loadAfterColdAttempt),
      );

      await assertUnique(page, '.resource-panel:has-text("lemonade") .ui-banner button:has-text("Pull + load")', "the Pull + load banner action");
      await page.click('.resource-panel:has-text("lemonade") .ui-banner button:has-text("Pull + load")');
      await page.waitForSelector('.resource-panel:has-text("lemonade") .ui-banner:has-text("Pulling from cold storage")');

      const loadAfterPullConfirm = stub
        .requests()
        .filter((r) => r.method === "POST" && r.path === "/api/tenants/lemonade/load");
      results.check(
        "item12: confirming Pull + load dispatches the retry (both requests carry the same model)",
        loadAfterPullConfirm.length === 2 && loadAfterPullConfirm[1].body?.model === "cold-test-model.gguf",
        JSON.stringify(loadAfterPullConfirm),
      );

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

      // Item 11 (part 2, DISPATCH) — the stub NEVER derives state (design
      // §5): its GET /api/sets responses are hand-authored fixture data,
      // not a reflection of what this gate's own POST actually carried.
      // That means the render check below (part 3) — "Load renders no
      // chip" — can only ever prove SetBuilder renders correctly whatever
      // ConfigSet it is GIVEN; it structurally CANNOT catch a
      // `removeModel()` regression that fails to clear something before
      // Save (e.g. forgetting `setDurable(null)`), because the fixture's
      // post-reload GET response was authored by this gate to already read
      // "removed" regardless. Mirrors item 4's dispatch/render split
      // (Task 7, R12): read the actual POST body off the wire instead and
      // assert the removal really was what got sent.
      //
      // Two things this check requires, not one (review fix round 1):
      //   (1) BOTH POSTs to /api/sets actually fired — the colliding
      //       overwrite=false attempt AND the overwrite=true retry. Both
      //       carry the SAME body (the frozen `overwriteSnapshot.draft`),
      //       so reading only the body of "whichever POST happened last"
      //       cannot tell a genuine two-click Overwrite flow apart from a
      //       broken one where Overwrite silently never dispatched at all
      //       (in which case "last" would just be the 409 attempt, whose
      //       body happens to be identical) — asserting the COUNT is what
      //       makes the two distinguishable.
      //   (2) The body reflects the removal POSITIVELY (`=== "unloaded"`),
      //       not merely `!== "loaded"` — a `!==` reads true for `undefined`
      //       too, so a typo'd path, a renamed key, or the whole resource
      //       entry going missing would ALL pass silently. `removeModel()`
      //       sets the resource's desired state to the literal string
      //       "unloaded" (SetBuilder.tsx's `setResourceChoice` call), so
      //       that is the exact value asserted.
      const overwritePosts = stub.requests().filter((r) => r.method === "POST" && r.path === "/api/sets");
      const savedAfterRemoval = overwritePosts[overwritePosts.length - 1];
      results.check(
        "item11: the removal is actually POSTed (both the colliding attempt and the Overwrite retry fired; the retry's body has durable cleared and the resource genuinely 'unloaded')",
        overwritePosts.length === 2 &&
          savedAfterRemoval?.body?.durable === null &&
          savedAfterRemoval?.body?.ephemeral?.resources?.lemonade?.desired === "unloaded",
        JSON.stringify({ postCount: overwritePosts.length, savedAfterRemoval: savedAfterRemoval ?? null }),
      );

      // Item 11 (part 3, RENDER-ONLY — review fix round 1 renamed this).
      // Reload the page (a genuine navigation, not the 3000ms poll — see
      // the fixture's own _note on why GET /api/sets is safe to script a
      // transition on) and Load the same set again. This check's fixture
      // entries for the post-reload GET /api/sets are IDENTICAL whether or
      // not the Overwrite POST above actually fired (both are hand-authored
      // "already removed" data) — the stub never derives state, so this
      // check cannot, by itself, tell "genuinely persisted" apart from
      // "always renders empty." That causal claim is the DISPATCH check's
      // job (above). What THIS check proves, honestly: given a ConfigSet
      // that carries no placed model, Load does not invent a ghost chip
      // from stale local component state that survived the reload.
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
        "item11: given the persisted (model-cleared) set, Load renders no ghost chip (render only — the DISPATCH check above proves the removal was actually saved)",
        ghostChip === 0 && dropzoneBack === 1,
        `ghostChip=${ghostChip} dropzoneBack=${dropzoneBack}`,
      );

      // Item 12 (part 3) — the alias-removal regression pin. Scanned across
      // EVERY request this whole run captured (items 9-13 alike), not just
      // the ones item 12 itself dispatched — see the header comment's "Item
      // 12" note. `stub.requests().length > 0` pairs the absence claim with
      // a positive one (R6/NO VACUOUS NEGATIONS): an empty capture list
      // would make the `.every` below vacuously true for the wrong reason
      // (nothing ran), not because every dispatch was generic.
      const allRequests = stub.requests();
      const sparkRequests = allRequests.filter((r) => r.path.startsWith("/api/spark"));
      results.check(
        "item12: no request this run captured ever targeted /api/spark (generic /api/tenants/{resource}/{verb} dispatch, never the removed spark-specific alias)",
        allRequests.length > 0 && sparkRequests.length === 0,
        `totalRequests=${allRequests.length} sparkRequests=${JSON.stringify(sparkRequests)}`,
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
    } catch (err) {
      err.partialRows = results.rows();
      throw err;
    }
  } finally {
    await browser.close();
    await stub.stop();
  }
  return results;
}
