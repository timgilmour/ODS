/**
 * Board display order.
 *
 * This lives in localStorage, not the backend, deliberately: it is a
 * per-browser view preference, not deck state. Making it follow the operator
 * between browsers is a real endpoint and should be added on purpose rather
 * than defaulted into.
 */

import type { DeckNode } from "./nodes";

const KEY = "model-deck.node-order";

/** Saved order first, then any node the saved order has never seen, in
 * discovery order. A newly registered node appears at the bottom rather than
 * silently vanishing because it is absent from the saved list. */
export function applyOrder(nodes: DeckNode[], order: string[]): DeckNode[] {
  const known = new Map(nodes.map((n) => [n.id, n]));
  const ordered: DeckNode[] = [];

  for (const id of order) {
    const node = known.get(id);
    if (node) {
      ordered.push(node);
      known.delete(id);
    }
  }
  return [...ordered, ...known.values()];
}

/** Computes the id list after dragging `dragging` onto `target`.
 *
 * Both indices are taken BEFORE the removal, which is what makes this
 * direction-aware for free: after splicing the dragged id out, `to` lands
 * just past the target for a forward drag and on it for a backward one.
 * Computing `to` after removal instead ("insert before target") makes a
 * forward drag onto the very next card a silent no-op — pull A out of
 * [A, B], reinsert before B, and you have [A, B] again. `dragging ===
 * target` (dropping a card back on itself) is a no-op by definition, not an
 * error — the array is still returned as a fresh copy.
 *
 * An id that is not in `ids` is also a no-op, and that guard is load-bearing
 * rather than defensive: a poll landing mid-drag can drop a node from the
 * board, and `splice(-1, 1)` deletes the LAST element instead of nothing,
 * which would then be written to localStorage as a corrupted order. The
 * previous `.filter()`-based removal did not have this failure mode; the
 * index arithmetic that made forward drags work introduced it. */
export function reorder(ids: string[], dragging: string, target: string): string[] {
  const next = [...ids];
  if (dragging === target) return next;

  const from = next.indexOf(dragging);
  const to = next.indexOf(target);
  if (from === -1 || to === -1) return next;

  next.splice(from, 1);
  next.splice(to, 0, dragging);
  return next;
}

export function loadOrder(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    // A corrupt or unavailable store is not worth failing a render over.
    return [];
  }
}

export function saveOrder(order: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(order));
  } catch {
    // Private-mode or quota failures are non-fatal: order just doesn't stick.
  }
}
