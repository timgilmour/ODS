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

const TOKEN_FAMILIES = ["status", "eventKind", "kind", "origin"];

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

/** Harvest the four token families this gate suite cares about. Each is a
 * SPECIFIC key found at a SPECIFIC structural position — never a generic
 * "any string value" sweep, which would just re-record churny data under a
 * different name:
 *  - status:    every `lifecycle.<resource>.status` (app/lifecycle.py's
 *               STATUSES — what the UI's status badges branch on).
 *  - eventKind: every `events[].kind` (app/events.py — what EventsView's
 *               severity map keys on).
 *  - kind:      every `engine-kinds.kinds[].kind` (the engine-kind catalog)
 *               AND every declared engine's `kind`
 *               (`nodes[].engines[].kind` — DeclaredEngine.kind, the same
 *               field POST/PUT /api/nodes/{id}/engines* accept/echo).
 *  - origin:    any `origin` field, wherever it appears (ResolvedEntry /
 *               FactEntry both carry "derived" | "declared").
 */
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
    if (key === "origin" && typeof child === "string") {
      tokens.origin.add(child);
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
 * whole module. */
export async function readLive(deckUrl) {
  const routes = ["/api/state", "/api/engine-kinds", "/api/nodes", "/openapi.json"];
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
