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
