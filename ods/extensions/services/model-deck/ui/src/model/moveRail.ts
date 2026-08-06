import type { StorageJob } from "../api";

export type RailState = "pending" | "active" | "done";
export interface RailStop {
  label: string;
  state: RailState;
}

/** The move rail renders mover.py's OBSERVABLE job states, one stop each:
 * "copying" (JobQueue._process, mover.py:281) — which internally covers
 * copy + in-flight sha256 + atomic rename AND source removal
 * (Mover._move_file, mover.py:124-126) — then "verifying" (mover.py:335,
 * brief post-move bookkeeping), then "done" (mover.py:348). "Removing
 * original" is never a separately observable state, so the rail does not
 * pretend it is (design decision 4's honesty rule, applied to what the
 * backend actually reports). A same-filesystem move (os.replace fast path,
 * mover.py:74-88) simply passes through in milliseconds — the rail arrives
 * at Moved with no special casing.
 *
 * failed/cancelled return null: from a terminal failure the job state alone
 * cannot say WHICH phase died, and a rail guessing would lie. The caller's
 * banner (job.error / cancelled copy) carries the outcome instead. */
export function moveRail(state: StorageJob["state"]): RailStop[] | null {
  switch (state) {
    case "queued":
      return [
        { label: "Copying", state: "pending" },
        { label: "Verifying", state: "pending" },
        { label: "Moved", state: "pending" },
      ];
    case "copying":
      return [
        { label: "Copying", state: "active" },
        { label: "Verifying", state: "pending" },
        { label: "Moved", state: "pending" },
      ];
    case "verifying":
      return [
        { label: "Copying", state: "done" },
        { label: "Verifying", state: "active" },
        { label: "Moved", state: "pending" },
      ];
    case "done":
      return [
        { label: "Copying", state: "done" },
        { label: "Verifying", state: "done" },
        { label: "Moved", state: "done" },
      ];
    case "failed":
    case "cancelled":
      return null;
  }
}
