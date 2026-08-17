/** Tier 2 (capture/fidelity) support: read the live deck's actual API
 * shape and vocabulary, and turn it into something a fixture can be
 * compared against — WITHOUT ever recording a value that could churn
 * (VRAM, timestamps, queue depths) or a value that could be sensitive
 * (an address, a credential). Only key names and a handful of known,
 * enum-like token strings are ever recorded.
 *
 * `extract` is the ONE extractor: `--capture` (Task 5, this module) writes
 * its output to `fixtures/e1-seeded-triple/vocabulary.json`; Task 11's
 * fidelity gate re-captures live and diffs the two.
 */

const TOKEN_FAMILIES = ["status", "eventKind", "kind"];

// R17 (controller ruling, review fix round 1): a live snapshot only ever
// shows the handful of tokens the box happens to be in AT THAT INSTANT —
// a healthy box with both sparky resources up will never show "status":
// "unreachable" no matter how many times you re-capture it, and a QUIET
// box's event tail can easily miss "apply-vetoed" or "load-failed" even
// though both are ordinary operator-triggered outcomes, not edge cases.
// Tier 2 is one-directional (unknown-live-token fails, fewer-than-committed
// passes), so a committed fixture that only ever holds whatever one
// snapshot observed WILL eventually redden on a token that was always
// legitimate — the first sparky reboot, the first vetoed apply, the first
// failed load — for exactly the reason tier 2 is designed not to.
//
// This is no longer a "hand-verify before committing" burden on whoever
// runs `--capture` (a warning in a docstring is not a guard). `run.mjs`'s
// `--capture` branch UNIONS its freshly observed vocabulary with whatever
// is already committed, rather than overwriting — a token or shape key,
// once known, can only be DROPPED by the explicit `--allow-shrink` flag,
// never by a quiet/healthy snapshot simply not reproducing it. That is the
// structural fix; this comment records why it exists and where the two
// families' CURRENT seed values came from, so their history is on record
// rather than reconstructed from a single capture:
//  - `status` — app/lifecycle.py:25-35's `STATUSES` tuple, a small CLOSED
//    enum (serving, drifted, down, parked, unexpected, unmanaged, idle,
//    unreachable, quarantined, warming). Seeded wholesale from that tuple
//    (10/10) rather than a live sample, since the whole set is finite and
//    directly readable from source.
//  - `eventKind` — NOT actually "no closed list" (review fix round 1's
//    correction of this module's own earlier claim): every kind literal
//    IS grep-enumerable, at every `log_event(...)` call site (~25 files)
//    and every `self._log("<kind>", ...)` dedup-wrapper call site
//    (app/arbiter.py's `Watcher._log`, app/storage.py's `_log`,
//    app/mover.py's `_log` — engine_kinds.py's load/unload/free paths call
//    through the Watcher's wrapper, not `log_event` directly, so a plain
//    grep for `log_event(` alone under-counts by about half). Seeded from
//    that full trace (58 kinds — 55 from current code paths, plus 3 retired
//    literals `load_lemonade`/`unload_lemonade`/`free_comfyui` that
//    app/sets.py:58-59 and app/arbiter.py:20 note are gone from the step
//    naming scheme but can still be genuinely observed live until an old
//    events.jsonl rotates them out), not invented.
//  - `kind` does not need seeding: it mirrors a small REGISTERED catalog
//    (`GET /api/engine-kinds`), which a live capture already sees in full.


function collectShape(shape, path, value) {
  if (value === null || typeof value !== "object") return;
  if (Array.isArray(value)) {
    // Arrays are transparent to the path: every element's keys fold into
    // the SAME path, sorted-union'd, rather than minting one shape entry
    // per index (indices are position, not vocabulary).
    for (const item of value) collectShape(shape, path, item);
    return;
  }
  const keys = Object.keys(value).sort();
  const existing = shape[path];
  shape[path] = existing ? [...new Set([...existing, ...keys])].sort() : keys;
  for (const [key, child] of Object.entries(value)) {
    collectShape(shape, `${path}.${key}`, child);
  }
}

/** Harvest the three token families this gate suite cares about. Each is a
 * SPECIFIC key found at a SPECIFIC structural position — never a generic
 * "any string value" sweep, which would just re-record churny data under a
 * different name:
 *  - status:    every `lifecycle.<resource>.status` (app/lifecycle.py's
 *               STATUSES — what the UI's status badges branch on).
 *  - eventKind: every `events[].kind` (app/events.py — what EventsView's
 *               severity map keys on). Populated from `GET /api/events`
 *               (Task 11 / R13), NOT `/api/state` — events are served by a
 *               separate route (app/routers/status.py:94; ui/src/api.ts:497)
 *               that `/api/state` carries no key for at all. This family
 *               was committed EMPTY for that reason until R13: an empty
 *               token family reads as coverage while catching nothing — it
 *               would not have caught the events-severity-map bug (the UI
 *               keyed on `refused`/`pull`/`reconciled` while log_event
 *               actually emits kinds like `apply-vetoed`/`load-failed`), the
 *               exact defect class tier 2 exists to guard.
 *  - kind:      every `engine-kinds.kinds[].kind` (the engine-kind catalog)
 *               AND every declared engine's `kind`
 *               (`nodes[].engines[].kind` — DeclaredEngine.kind, the same
 *               field POST/PUT /api/nodes/{id}/engines* accept/echo).
 *
 * A fourth family, `origin` (ResolvedEntry / FactEntry's "derived" |
 * "declared"), was considered and DROPPED (Task 11 / R13) rather than
 * wired: the only GET route that ever emits it, `/api/facts`, is keyed by
 * artifact id (`model/Qwen3.6-...`, `engine/sparky/vllm`) — a box's own
 * inventory, not vocabulary. `extract`'s shape half treats every object key
 * as a vocabulary token (that is what makes the `/api/state` structural
 * checks work at all), so folding `/api/facts` into the same uniform
 * shape+token sweep would put those inventory-dependent ids into `shape`
 * too — a new model downloaded, or a new engine brought up, would then
 * read as an "unknown key" and redden the gate for a reason that has
 * nothing to do with vocabulary drift. That is precisely the "reddens for
 * a benign reason, so people stop reading it" failure this design exists to
 * avoid. Teaching `extract` to shape-check some payloads and only
 * token-harvest others would fix it, but is a real redesign, not a
 * same-shaped extension like `/api/events` — out of proportion for a
 * family nothing in the committed E1 fixture currently exercises. An empty,
 * undroppable family is false assurance either way; removing it is the
 * honest record of that call, not a bug. */
function harvestTokens(value, tokens) {
  if (value === null || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const item of value) harvestTokens(item, tokens);
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    if (key === "lifecycle" && child && typeof child === "object" && !Array.isArray(child)) {
      for (const entry of Object.values(child)) {
        if (entry && typeof entry.status === "string") tokens.status.add(entry.status);
      }
    }
    if (key === "events" && Array.isArray(child)) {
      for (const item of child) {
        if (item && typeof item.kind === "string") tokens.eventKind.add(item.kind);
      }
    }
    if ((key === "kinds" || key === "engines") && Array.isArray(child)) {
      for (const item of child) {
        if (item && typeof item.kind === "string") tokens.kind.add(item.kind);
      }
    }
    harvestTokens(child, tokens);
  }
}

/** payloads: Record<path, body> (e.g. `{"/api/state": {...}, ...}`) ->
 * `{shape: Record<path, string[]>, tokens: Record<family, string[]>}`.
 * Deterministic and value-free: same inputs, same output, and nothing but
 * key names and the three token families above ever lands in it. */
export function extract(payloads) {
  const shape = {};
  const tokens = Object.fromEntries(TOKEN_FAMILIES.map((f) => [f, new Set()]));

  for (const [path, body] of Object.entries(payloads)) {
    collectShape(shape, path, body);
    harvestTokens(body, tokens);
  }

  return {
    shape,
    tokens: Object.fromEntries(
      TOKEN_FAMILIES.map((f) => [f, [...tokens[f]].sort()]),
    ),
  };
}

/** Tier 2's ONLY way of touching the live deck. GET only — no other HTTP
 * method appears in this function, which is what makes "tier 2 never
 * writes" reviewable by reading one function rather than auditing the
 * whole module.
 *
 * `/api/events?n=500` (R13): a wide-ish N so a single capture sees a real
 * spread of kinds (a fresh/quiet box's tail could otherwise be all one
 * kind, e.g. a poll-loop's own routine entries, and commit a fixture that
 * looks populated but still only knows one token). Still GET-only, still
 * read-only — `n` only bounds how much of the existing append-only log is
 * read back, nothing is written.
 *
 * `key` is deliberately the QUERY-FREE route (`/api/events`), while `path`
 * is what's actually fetched (`/api/events?n=500`) — review fix round 1: an
 * earlier version used the query-carrying string as the `payloads` key too,
 * which baked `n` into every `shape`/`tokens` path this route contributes.
 * Changing `n` later would then silently stop `compare()` from matching
 * that path against the committed fixture at all (a quiet widening of the
 * one-directional gap, not a loud failure) rather than the two consistently
 * naming the same route. */
export async function readLive(deckUrl) {
  const routes = [
    { path: "/api/state", key: "/api/state" },
    { path: "/api/engine-kinds", key: "/api/engine-kinds" },
    { path: "/api/nodes", key: "/api/nodes" },
    { path: "/api/events?n=500", key: "/api/events" },
    { path: "/openapi.json", key: "/openapi.json" },
  ];
  const payloads = {};
  for (const { path, key } of routes) {
    const res = await fetch(`${deckUrl}${path}`);
    if (!res.ok) {
      throw new Error(`deck-gate capture: GET ${path} -> ${res.status} ${res.statusText}`);
    }
    payloads[key] = await res.json();
  }
  return payloads;
}
