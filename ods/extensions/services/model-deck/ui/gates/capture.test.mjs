import { describe, it, expect } from "vitest";
import { extract } from "./capture.mjs";

describe("extract", () => {
  it("records key sets per payload section, sorted", () => {
    const out = extract({
      "/api/state": { node: { id: "local", label: "autarch" }, lifecycle: {} },
    });
    expect(out.shape["/api/state"]).toEqual(["lifecycle", "node"]);
    expect(out.shape["/api/state.node"]).toEqual(["id", "label"]);
  });

  it("collects the status vocabulary across every lifecycle entry", () => {
    // The recurring bug class is vocabulary, not values: the events severity
    // map keyed on kinds no log_event ever emits, warming derived from a
    // swap state the backend documents differently. Tokens are what tier 2
    // guards.
    const out = extract({
      "/api/state": {
        lifecycle: {
          "local/hipfire": { status: "serving" },
          "sparky/slot0": { status: "unreachable" },
        },
      },
    });
    expect(out.tokens.status).toEqual(["serving", "unreachable"]);
  });

  it("does not record values that churn", () => {
    // A gate that reddens because VRAM moved is a gate people stop reading.
    const out = extract({
      "/api/state": { world: { gpus: [{ index: 0, free: 12345 }] } },
    });
    expect(JSON.stringify(out)).not.toContain("12345");
  });
});
