/** Tier 2 — the fidelity gate. Tier 1 (fixture) is deterministic and
 * dispatch-shaped: it proves the UI reacts correctly to a SCRIPTED payload,
 * but the script is authored by whoever holds the vocabulary, so it cannot
 * catch a value that was copied from a mockup instead of read out of the
 * backend — the bug class that has recurred FIVE times on this project.
 * Tier 2 re-reads a LIVE deck and proves the committed fixture's shape and
 * vocabulary are still real.
 *
 * `compare(live, committed)` is the whole gate, and it is a PURE export
 * (R3, controller ruling): no I/O, no console output, no throwing on
 * anything but a genuinely malformed row. That is what makes the
 * one-directional rule unit-testable without a live deck — `fidelity.
 * test.mjs` never touches the network. `run()` is the only impure part: it
 * reads the live deck (via `capture.mjs`'s `readLive`, GET-only) and the
 * committed fixture off disk, then folds `compare`'s output through
 * `lib/check.mjs` so it reports the same way every other gate does.
 *
 * WHAT THIS ASSERTS — three things only, per the design (spec §6): key
 * sets per payload section (`shape`), value TYPES are not checked here
 * (the three token families capture.mjs harvests are already string-typed
 * by construction — see capture.mjs), and the observed enum-token
 * inventory (`tokens`). It does NOT diff values: `/api/state` churns
 * constantly (VRAM, queue depths, `last_healthy_ts`, timestamps), and a
 * gate that reddens because VRAM moved is a gate people stop reading.
 *
 * ONE-DIRECTIONAL, DELIBERATELY. Every token and key OBSERVED live must be
 * KNOWN to the committed fixture; this gate does NOT require every token
 * the fixture knows to still appear live. Sparky being down (or mid-boot,
 * reading `unreachable` on both its resources) changes the payload
 * wholesale — fewer lifecycle entries, fewer remote_gpus, a `kind` or two
 * that simply doesn't show up this run — and a gate that reddens when a
 * box reboots is a gate people stop reading. The deck makes this same
 * distinction itself (`unreachable` != `down`); this gate makes it too.
 *
 * DOCUMENTED LIMITATION: because the shape check is one-directional, a
 * field REMOVED from inside a node-dependent section (e.g. one lifecycle
 * entry losing its `reason` key) is INDISTINGUISHABLE from "that node
 * isn't present this run" — both just make live's key list at that path
 * smaller, which the one-directional rule lets through by design. This is
 * not solved here. It is partially clawed back for the two sections that
 * are structurally guaranteed regardless of which nodes/resources exist —
 * see UNCONDITIONAL_SHAPE_PATHS below — and left honestly open everywhere
 * else. A regression drill in that gap is exactly the FULL-COVERAGE tier
 * this project has deliberately not built (spec §6's tier 2 is a proof
 * against DRIFT off a known-good snapshot, not a schema validator). */

import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { reportingRun } from "./lib/check.mjs";
import { extract, readLive } from "./capture.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const COMMITTED_PATH = join(HERE, "fixtures/e1-seeded-triple/vocabulary.json");

export const name = "fidelity";

/** The only two shape paths whose key set is guaranteed regardless of
 * which nodes or resources happen to be declared/reachable this run:
 *  - `/api/state.node` — the local node's own identity block
 *    (app/routers/status.py's `_local_identity`: `{id, label}`, always
 *    exactly one, never conditional on topology).
 *  - `/api/state` — the top-level snapshot's own key set (`get_state`'s
 *    seven-key dict: lifecycle, models, node, nodes, policy, provenance,
 *    world — always all seven, whatever nodes/resources are or aren't
 *    live). This is the path the task brief's own worked example uses
 *    (`committed.shape["/api/state"] = ["lifecycle", "node"]`), and
 *    "lifecycle" — the one lifecycle-shaped member of a set that is
 *    otherwise identity/config data — is why the brief calls this pairing
 *    "node, and the lifecycle key set".
 *
 * For every OTHER shape path (world.tenants, lifecycle.<resource>,
 * nodes[], remote_gpus.<node>, ...) only the general one-directional rule
 * runs: which keys of a path that exists in BOTH live and committed are
 * observed live but unknown to committed. Whether the path itself exists
 * at all is topology, and topology is exactly what one-directional is
 * built to tolerate. */
const UNCONDITIONAL_SHAPE_PATHS = ["/api/state", "/api/state.node"];

/** Shape half. Two kinds of row:
 *   1. ONE aggregate row across every shape path present in BOTH live and
 *      committed: every key OBSERVED live at that path must be KNOWN to
 *      committed. A key that exists live but never made it into the
 *      committed fixture is the shape-side version of the same defect
 *      class the token half exists for — a field the UI reads that the
 *      author never captured.
 *   2. One row PER unconditional path (at most two): every key the
 *      committed fixture KNOWS at that path must still be observed live.
 *      This is the opposite direction, deliberately narrow — see the
 *      module doc's "documented limitation". */
function diffShape(live, committed) {
  const rows = [];
  const liveShape = live.shape ?? {};
  const committedShape = committed.shape ?? {};

  const sharedPaths = Object.keys(committedShape).filter((p) => p in liveShape);
  const unknownFindings = [];
  for (const path of sharedPaths) {
    const known = new Set(committedShape[path]);
    for (const key of liveShape[path]) {
      if (!known.has(key)) unknownFindings.push(`${path}: "${key}"`);
    }
  }
  rows.push({
    name: "shape: every key observed live at a path the fixture also has is known to the fixture",
    ok: unknownFindings.length === 0,
    detail:
      unknownFindings.length === 0
        ? `${sharedPaths.length} shared path(s) checked, all observed keys known`
        : `unknown key(s) — ${unknownFindings.join("; ")}`,
  });

  for (const path of UNCONDITIONAL_SHAPE_PATHS) {
    if (!(path in committedShape) || !(path in liveShape)) continue;
    const observed = new Set(liveShape[path]);
    const missing = committedShape[path].filter((k) => !observed.has(k));
    rows.push({
      name: `shape: unconditional section "${path}" still carries every key the fixture knows`,
      ok: missing.length === 0,
      detail:
        missing.length === 0
          ? `all ${committedShape[path].length} known key(s) present`
          : `missing — ${missing.join(", ")}`,
    });
  }

  return rows;
}

/** Vocabulary half. One row per token family the committed fixture
 * declares (status, eventKind, kind — see capture.mjs for why `origin`
 * isn't one of them): every token OBSERVED live in that family must be
 * KNOWN to the committed fixture. Fewer live tokens than committed knows
 * is not a finding — see the module doc's one-directional rationale. */
function diffTokens(live, committed) {
  const liveTokens = live.tokens ?? {};
  const committedTokens = committed.tokens ?? {};
  const rows = [];
  for (const family of Object.keys(committedTokens)) {
    const known = new Set(committedTokens[family]);
    const observed = liveTokens[family] ?? [];
    const unknown = observed.filter((t) => !known.has(t));
    rows.push({
      name: `tokens: every observed "${family}" token is known to the committed fixture`,
      ok: unknown.length === 0,
      detail:
        unknown.length === 0
          ? `${observed.length} observed token(s), all known`
          : `unknown ${family} token(s): ${unknown.join(", ")}`,
    });
  }
  return rows;
}

/** Sanity half (review fix round 1, IMPORTANT finding). `diffShape` and
 * `diffTokens` only ever compare paths/families the COMMITTED fixture
 * declares — an empty or truncated `committed` (`{shape: {}, tokens: {}}`)
 * makes `sharedPaths` empty and `Object.keys(committedTokens)` empty, so
 * both loops silently iterate zero times and produce zero findings. Before
 * this check, that degenerate fixture read as a clean PASS (an aggregate
 * shape row saying "0 shared path(s) checked, all keys known" is
 * technically true and utterly meaningless) — a corrupted or truncated
 * `vocabulary.json` would gate GREEN, the exact opposite of what a
 * fidelity check exists to do. Two rows, checked against `committed` alone
 * (no live data involved, so a degenerate fixture is caught even before
 * `readLive` runs — see `run()`): committed must declare at least one
 * shape path, and every token family it declares must carry at least one
 * known token (an empty family is the same "reads as coverage, catches
 * nothing" problem R13 fixed for `eventKind` — see capture.mjs). */
function diffSanity(committed) {
  const committedShape = committed.shape ?? {};
  const committedTokens = committed.tokens ?? {};
  const families = Object.keys(committedTokens);
  const emptyFamilies = families.filter((f) => (committedTokens[f] ?? []).length === 0);

  return [
    {
      name: "sanity: the committed fixture declares at least one shape path",
      ok: Object.keys(committedShape).length > 0,
      detail: `${Object.keys(committedShape).length} shape path(s) declared`,
    },
    {
      name: "sanity: every token family the committed fixture declares carries at least one known token",
      ok: families.length > 0 && emptyFamilies.length === 0,
      detail:
        families.length === 0
          ? "no token families declared"
          : emptyFamilies.length === 0
            ? `${families.length} family(ies) declared, all non-empty`
            : `empty famil(y/ies): ${emptyFamilies.join(", ")}`,
    },
  ];
}

/** PURE. `live` and `committed` are both `{shape, tokens}` — the shape
 * `extract()` (capture.mjs) produces and the shape the committed
 * `vocabulary.json` fixture is stored in. No I/O, no console output, never
 * throws on well-formed input (see `run()` for the impure shell that reads
 * the live deck and the fixture off disk). */
export function compare(live, committed) {
  return [...diffSanity(committed), ...diffShape(live, committed), ...diffTokens(live, committed)];
}

/** The impure shell. GET-only — `readLive` (capture.mjs) is the ONLY
 * function in this module's live-facing call path, and it in turn makes
 * only `fetch(url)` calls with no method override, so "no HTTP method
 * other than GET ever appears in a live-facing code path" is confirmable
 * by reading that one function. */
export async function run({ deckUrl }) {
  if (!deckUrl) {
    throw new Error("fidelity gate: --deck-url is required for the live tier");
  }
  const committed = JSON.parse(await readFile(COMMITTED_PATH, "utf8"));
  const payloads = await readLive(deckUrl);
  const live = extract(payloads);

  // R15 (controller ruling): `reportingRun` (lib/check.mjs) attaches
  // whatever checks already ran to a mid-run throw as `err.partialRows` —
  // the same contract every other gate uses. Nothing above this point (the
  // `--deck-url` precondition, the live read, the fixture read) is eligible
  // for that attachment: there is no `results` accumulator yet when any of
  // those can throw, same as before this helper existed.
  return reportingRun(async (results) => {
    for (const row of compare(live, committed)) {
      results.check(row.name, row.ok, row.detail);
    }
  });
}
