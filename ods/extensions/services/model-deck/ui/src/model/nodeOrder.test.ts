import { describe, expect, it } from "vitest";
import type { DeckNode } from "./nodes";
import { applyOrder } from "./nodeOrder";

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
