/**
 * Which verbs a DECLARED REMOTE engine offers, and which of them would be a
 * no-op right now. Pure, componentless — the same "logic inline in a
 * component is logic no test can reach" rule `nodeForm.ts`/`engineForm.ts`
 * name.
 *
 * The vocabulary is DATA, never a literal here: it comes from
 * `GET /api/engine-kinds`'s per-kind `human_verbs`
 * (app/routers/nodes.py:476-504's `list_engine_kinds`, which serves
 * `sorted(ENGINE_KINDS[kind].human_verbs())` at :502 — so the payload's
 * order is already deterministic and is rendered as given, never re-sorted).
 * sglang-omni's is exactly {"load", "unload"} and deliberately never
 * park/resume (app/engine_kinds.py:966-970).
 *
 * The BACKEND stays the authority on what is allowed: `engine_verb`
 * (app/routers/serving.py:247-357) re-checks membership itself and refuses
 * by name — 405 for a verb the kind does not declare, 501 for one the
 * node-agent engine channel has no call for. This module only decides what
 * to OFFER, so a disagreement surfaces as that route's own sentence in a
 * banner rather than as a silently missing button.
 */

import type { EngineKindsResponse } from "../api";

/** One offered verb. `disabled` is a no-op/unreachable judgement, not a
 * permission one — see `remoteEngineVerbs`. */
export interface EngineVerb {
  verb: string;
  disabled: boolean;
}

/** The engine states that mean it is RESIDENT on its node's GPU —
 * app/engine_kinds.py:949-955's `active()`, verbatim: "`observe` emits only
 * busy/idle/down/unknown, so these two ARE the resident states". "idle"
 * describes the request queue, not residency (it holds ~62 GiB of weights
 * between renders).
 *
 * Exported because `model/nodes.ts` needs the SAME reading for its
 * no-lifecycle-entry fallback, and two copies of "which states mean loaded"
 * is exactly how a board and its buttons come to disagree. */
export function isResident(state: string): boolean {
  return state === "busy" || state === "idle";
}

/** The verb each observed state would make a no-op, keyed by the CHANNEL's
 * two verbs (app/routers/serving.py:228's `_REMOTE_VERBS`: load -> the
 * agent's `up`, unload -> its `down`).
 *
 * "unknown" appears in neither arm on purpose: it is `unknown()`'s record
 * (app/engine_kinds.py:931-947), i.e. the deck failed to LOOK — proof of
 * nothing, so it withholds nothing. A verb this table has no row for is
 * never disabled here; the kind declared it, and the backend is what
 * refuses it if it cannot be served.
 */
function isNoOp(verb: string, state: string): boolean {
  if (verb === "load") return isResident(state);
  if (verb === "unload") return state === "down";
  return false;
}

/**
 * The verb buttons for one declared remote engine of `kind`, observed in
 * `state`, on a node that is `stale` (unreachable) or not.
 *
 * Empty when the catalog has not landed (or its fetch failed), and empty for
 * a kind the catalog does not list: a verb this UI cannot name is a verb it
 * must not guess. The card still renders — status without controls is a
 * degraded surface, an invented verb is a wrong one.
 *
 * `stale` disables everything, for the reason ResourcePanel already withholds
 * a local resource's controls behind the same flag: a last-known reading is a
 * memory, and nothing can act on a memory. It travels with the verb (rather
 * than being re-checked at the panel) so the card can keep showing what it
 * last knew while offering nothing that cannot land.
 */
export function remoteEngineVerbs(
  kinds: EngineKindsResponse | null,
  kind: string,
  state: string,
  stale: boolean,
): EngineVerb[] {
  const def = kinds?.kinds.find((k) => k.kind === kind);
  if (!def) return [];
  return def.human_verbs.map((verb) => ({
    verb,
    disabled: stale || isNoOp(verb, state),
  }));
}
