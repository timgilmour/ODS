import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { renderMarkdown, renderJson } from "./lib/report.mjs";

function arg(flag, fallback = null) {
  const i = process.argv.indexOf(flag);
  return i === -1 ? fallback : process.argv[i + 1];
}

const tier = arg("--tier", "fixture");
const reportDir = arg("--report-dir");
const pattern = arg("--pattern");

// R1 (controller ruling): only "smoke" is registered here. Task 6 registers
// e1-engines.gate.mjs into "fixture" and Task 8 registers fidelity.gate.mjs
// into "live" — each task adds its own gate to its own tier when the module
// actually exists. A dynamic import() of a not-yet-written module would
// throw at dispatch time, which would make this task's own acceptance step
// unrunnable.
const GATES = {
  fixture: ["./smoke.gate.mjs"],
};

// R2 (controller ruling): a tier with no registered gates is not "nothing to
// do" — GATES[tier] ?? [] would run zero gates and exit 0, a silent no-op
// that is exactly the failure the design's fail-loudly rule exists to
// prevent (e.g. --capture maps to TIER=capture in deck-gate, but Task 5 is
// what adds the capture branch here). Fail loudly instead, naming the tier.
const gatePaths = GATES[tier];
if (!gatePaths) {
  console.error(`deck-gate: no gates registered for tier "${tier}"`);
  process.exit(2);
}

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const rows = [];
for (const path of gatePaths) {
  const mod = await import(path);
  if (pattern && !mod.name.includes(pattern)) continue;
  const results = await mod.run({ deckUrl: arg("--deck-url") });
  rows.push(...results.rows());
}

const meta = { tier, startedIso: new Date().toISOString(), target: arg("--deck-url") };
await mkdir(reportDir, { recursive: true });
await writeFile(join(reportDir, `${stamp}.md`), renderMarkdown(rows, meta));
await writeFile(join(reportDir, `${stamp}.json`), renderJson(rows, meta));

const failed = rows.filter((r) => !r.ok).length;
console.log(`\n${rows.length - failed} passed / ${failed} failed — ${reportDir}/${stamp}.md`);
process.exit(failed ? 1 : 0);
