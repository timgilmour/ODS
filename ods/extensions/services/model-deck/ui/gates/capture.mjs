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

// ⚠ Task 11 / DEPLOY TRAP: the committed `tokens.status` array is NOT the
// raw output of the last `--capture` run. `status` is a small, CLOSED
// enum (app/lifecycle.py:25-35's `STATUSES` tuple — serving, drifted,
// down, parked, unexpected, unmanaged, idle, unreachable, quarantined,
// warming), but a single live snapshot only ever shows the handful of
// statuses the box happens to be in at that instant — a healthy box with
// both sparky resources up will never show "unreachable", "down",
// "drifted", "quarantined", or "warming" no matter how many times you
// re-capture it. Tier 2 is one-directional (unknown-live-token fails,
// fewer-than-committed passes), so if the committed fixture only ever
// carries whatever one snapshot observed, the FIRST time a box legitimately
// goes unreachable (including a routine sparky reboot) tier 2 reddens for
// exactly the reason it is designed not to.
//
// The fix is NOT in `extract()` — it correctly records what it sees. The
// fix is that whoever re-runs `--capture` for `status` must hand-verify
// the result still contains the full STATUSES set (cross-checked against
// app/lifecycle.py, not invented) before committing, the same way any
// other hand-reconciled fixture value is verified rather than trusted from
// a single sample. `eventKind` and `kind` do NOT need this: `kind` mirrors
// a small REGISTERED catalog (GET /api/engine-kinds), not a closed source
// enum, and `eventKind` is deliberately sample-based — there is no single
// closed list of event kinds to check it against (log_event is called with
// a literal string at ~30 call sites across the codebase, some through a
// variable), so eventKind's job is catching a kind the fixture has TRULY
// never seen, and widening its sample over time (via `--live n`) is the
// correct way to grow its coverage, unlike status which has one true
// finite answer today.


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
 * key names and the four token families above ever lands in it. */
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
 * read back, nothing is written. */
export async function readLive(deckUrl) {
  const routes = ["/api/state", "/api/engine-kinds", "/api/nodes", "/api/events?n=500", "/openapi.json"];
  const payloads = {};
  for (const path of routes) {
    const res = await fetch(`${deckUrl}${path}`);
    if (!res.ok) {
      throw new Error(`deck-gate capture: GET ${path} -> ${res.status} ${res.statusText}`);
    }
    payloads[path] = await res.json();
  }
  return payloads;
}
