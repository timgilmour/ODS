import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { renderMarkdown, renderJson } from "./lib/report.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

function arg(flag, fallback = null) {
  const i = process.argv.indexOf(flag);
  return i === -1 ? fallback : process.argv[i + 1];
}

const tier = arg("--tier", "fixture");
const reportDir = arg("--report-dir");
const pattern = arg("--pattern");

// R2 (controller ruling), capture half: "capture" produces no PASS/FAIL
// rows — it reads the live deck and WRITES the vocabulary fixture — so it
// is handled here, before the GATES dispatch below, rather than being
// registered into it as a gate. Every OTHER unregistered tier still fails
// loudly (see below): this branch does not weaken that refusal, it only
// gives "capture" a real destination instead of falling into it.
if (tier === "capture") {
  const deckUrl = arg("--deck-url");
  if (!deckUrl) {
    console.error(`deck-gate: --capture requires --deck-url`);
    process.exit(2);
  }
  const { extract, readLive } = await import("./capture.mjs");
  const payloads = await readLive(deckUrl);
  const vocabulary = extract(payloads);
  const outPath = join(HERE, "fixtures/e1-seeded-triple/vocabulary.json");
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(vocabulary, null, 2) + "\n");
  console.log(`deck-gate: captured vocabulary from ${deckUrl} -> ${outPath}`);
  process.exit(0);
}

// R1 (controller ruling): Task 8 registers fidelity.gate.mjs into "live" —
// each task adds its own gate to its own tier when the module actually
// exists. A dynamic import() of a not-yet-written module would throw at
// dispatch time, which would make that task's own acceptance step
// unrunnable. Task 6 adds e1-engines.gate.mjs here, into "fixture".
const GATES = {
  fixture: ["./smoke.gate.mjs", "./e1-engines.gate.mjs", "./e1-board.gate.mjs"],
};

// R2 (controller ruling): a tier with no registered gates is not "nothing to
// do" — GATES[tier] ?? [] would run zero gates and exit 0, a silent no-op
// that is exactly the failure the design's fail-loudly rule exists to
// prevent. Fail loudly instead, naming the tier.
const gatePaths = GATES[tier];
if (!gatePaths) {
  console.error(`deck-gate: no gates registered for tier "${tier}"`);
  process.exit(2);
}

// Fix round 1, item 4: captured before the loop runs — a report labelled
// "started" must carry the time the run began, not the time the last gate
// finished.
const startedAt = new Date();
const stamp = startedAt.toISOString().replace(/[:.]/g, "-");
// Task 4 carry-over, closed by Task 6: createResults()'s duplicate-name
// guard is per-instance, i.e. per-gate. With a second gate now registered,
// two DIFFERENT gates could emit the same check name and one FAIL would
// hide behind the other gate's same-named PASS in the merged report below —
// a silent failure the whole point of this harness is to rule out. Checked
// here, across every gate's rows, before they are merged; a collision
// refuses loudly (exit 2) naming both the check and the gate that repeated
// it, rather than writing a report that could hide one behind the other.
const rows = [];
const seenNames = new Set();
// R15 (controller ruling): a gate that THROWS must still produce a report.
// Before this, an exception from mod.run() propagated straight out of this
// loop with nothing written — the exact gap that forced Task 8's item-9
// RED proof into a hand-saved terminal capture instead of a real report
// file. A gate's `createResults().check()` prints PASS/FAIL synchronously
// as it goes, so those checks are real even when the gate later throws;
// they are recovered here IF the gate attaches them to the error as
// `partialRows` before rethrowing (e1-board.gate.mjs does this — see its
// own try/catch). A gate that does not do this simply contributes no rows
// of its own, and the synthetic row below still names it and its error, so
// the crash is never silent even in that case.
for (const path of gatePaths) {
  const mod = await import(path);
  if (pattern && !mod.name.includes(pattern)) continue;
  let results;
  try {
    results = await mod.run({ deckUrl: arg("--deck-url") });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`deck-gate: gate "${mod.name}" threw: ${message}`);
    if (Array.isArray(err?.partialRows)) {
      for (const row of err.partialRows) {
        if (!seenNames.has(row.name)) {
          seenNames.add(row.name);
          rows.push(row);
        }
      }
    }
    rows.push({ name: `${mod.name}: gate threw before completing`, ok: false, detail: message });
    break;
  }
  for (const row of results.rows()) {
    if (seenNames.has(row.name)) {
      console.error(
        `deck-gate: duplicate check name across gates: "${row.name}" ` +
          `(reintroduced by gate "${mod.name}"). Duplicate-name protection ` +
          `is per-gate (createResults()); a name repeated in a second gate ` +
          `can hide a FAIL behind another gate's PASS once rows are merged.`,
      );
      process.exit(2);
    }
    seenNames.add(row.name);
  }
  rows.push(...results.rows());
}

// Fix round 1, item 1: a -k pattern that matches no gate (or, more
// generally, any run that produces zero checks) is the same silent no-op
// success ruling R2 forbids one level below the tier check — refuse rather
// than report a green "0 passed / 0 failed". No report is written, mirroring
// the tier-check refusal above.
if (rows.length === 0) {
  console.error(
    `deck-gate: no checks ran` +
      (pattern ? ` — pattern "${pattern}" matched no gate in tier "${tier}"` : ""),
  );
  process.exit(2);
}

const meta = { tier, startedIso: startedAt.toISOString(), target: arg("--deck-url") };
await mkdir(reportDir, { recursive: true });
await writeFile(join(reportDir, `${stamp}.md`), renderMarkdown(rows, meta));
await writeFile(join(reportDir, `${stamp}.json`), renderJson(rows, meta));

const failed = rows.filter((r) => !r.ok).length;
console.log(`\n${rows.length - failed} passed / ${failed} failed — ${reportDir}/${stamp}.md`);
process.exit(failed ? 1 : 0);
