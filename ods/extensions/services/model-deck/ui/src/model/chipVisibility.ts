/**
 * Whether an ENGINE-kind chip (comfyui local, sglang-omni remote) earns a
 * place on its GPU card. Ruling 2026-08-18: an engine with no model shows
 * NOTHING — except failures, which must stay visible.
 *
 * `model/nodes.ts`'s `tenantPlacement` comfyui arm and `remoteEngineResources`
 * both call this before building a placement; Task 3's per-GPU card builders
 * are the other consumers this is written for.
 */

import { isResident } from "./engineVerbs";
import type { LifecycleStatus } from "../api";

/** Failure/transition statuses that always show, regardless of kind or
 * engine state (approved exception to the "no model, no chip" ruling): a
 * crashed or quarantined load must stay visible, and warming is a load in
 * flight, not an empty slot. Mirrors `app/lifecycle.py`'s STATUSES —
 * `LifecycleStatus` in `../api.ts` is the same enum. */
const ALWAYS_VISIBLE: ReadonlySet<LifecycleStatus> = new Set(["down", "quarantined", "warming"]);

/**
 * `kind` branches on the closed backend enum (`app/engine_kinds.py:177-192`),
 * same as `setDraft.ts`'s `KIND_DRAFT_SPEC` — a new kind needs a branch here
 * too.
 */
export function engineChipVisible(
  kind: string,
  tenantState: string,
  queue: number | null | undefined,
  lifecycleStatus: LifecycleStatus | undefined,
): boolean {
  if (lifecycleStatus && ALWAYS_VISIBLE.has(lifecycleStatus)) return true;
  if (kind === "comfyui") {
    // app/engine_kinds.py:613-635 — comfyui's "idle" describes the request
    // queue, NOT residency (the opposite reading from sglang-omni's "idle"
    // below), so its visibility question is "is there work", not "is it
    // loaded": busy right now, or a non-empty queue.
    return tenantState === "busy" || (queue ?? 0) > 0;
  }
  // sglang-omni (and any other load/unload kind): busy|idle ARE the
  // resident states (app/engine_kinds.py:949-955 — "observe emits only
  // busy/idle/down/unknown, so these two ARE the resident states"; "idle"
  // here describes the queue, the weights stay on the GPU).
  return isResident(tenantState);
}
