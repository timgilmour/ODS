import { describe, expect, it } from "vitest";
import type { DeckNode } from "./nodes";
import { applyOrder, reorder } from "./nodeOrder";

const node = (id: string): DeckNode => ({
  id, label: id, status: "reachable", lastSeen: null, resources: [],
});

describe("applyOrder", () => {
  it("keeps discovery order when nothing is saved", () => {
    const nodes = [node("local"), node("sparky")];
    expect(applyOrder(nodes, []).map((n) => n.id)).toEqual(["local", "sparky"]);
  });

  it("applies a saved order", () => {
    const nodes = [node("local"), node("sparky")];
    expect(applyOrder(nodes, ["sparky", "local"]).map((n) => n.id)).toEqual(["sparky", "local"]);
  });

  it("appends nodes the saved order has never seen", () => {
    const nodes = [node("local"), node("sparky"), node("hera")];
    expect(applyOrder(nodes, ["sparky", "local"]).map((n) => n.id)).toEqual([
      "sparky", "local", "hera",
    ]);
  });

  it("ignores saved ids for nodes that no longer exist", () => {
    const nodes = [node("local")];
    expect(applyOrder(nodes, ["ghost", "local"]).map((n) => n.id)).toEqual(["local"]);
  });
});

describe("reorder", () => {
  it("moves a card onto the very next card (forward-adjacent)", () => {
    // The case that was silently broken: "insert before target" computed
    // AFTER removal reproduces the original order for adjacent forward
    // drags. This must actually change the order.
    expect(reorder(["A", "B"], "A", "B")).toEqual(["B", "A"]);
  });

  it("moves a card onto the card just before it (backward-adjacent)", () => {
    expect(reorder(["A", "B"], "B", "A")).toEqual(["B", "A"]);
  });

  it("moves a card forward across a gap", () => {
    expect(reorder(["A", "B", "C"], "A", "C")).toEqual(["B", "C", "A"]);
  });

  it("moves a card backward across a gap", () => {
    expect(reorder(["A", "B", "C"], "C", "A")).toEqual(["C", "A", "B"]);
  });

  it("is a no-op when a card is dropped on itself", () => {
    expect(reorder(["A", "B", "C"], "B", "B")).toEqual(["A", "B", "C"]);
  });
});
