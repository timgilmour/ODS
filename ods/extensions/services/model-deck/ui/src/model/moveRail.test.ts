import { describe, expect, it } from "vitest";
import { moveRail } from "./moveRail";

describe("moveRail", () => {
  it("queued: nothing lit", () => {
    expect(moveRail("queued")).toEqual([
      { label: "Copying", state: "pending" },
      { label: "Verifying", state: "pending" },
      { label: "Moved", state: "pending" },
    ]);
  });

  it("copying: first stop active", () => {
    expect(moveRail("copying")?.map((s) => s.state)).toEqual([
      "active", "pending", "pending",
    ]);
  });

  it("verifying: copy done, second stop active", () => {
    expect(moveRail("verifying")?.map((s) => s.state)).toEqual([
      "done", "active", "pending",
    ]);
  });

  it("done: everything lit", () => {
    expect(moveRail("done")?.map((s) => s.state)).toEqual(["done", "done", "done"]);
  });

  it("failed and cancelled have no rail — the banner carries the outcome", () => {
    expect(moveRail("failed")).toBeNull();
    expect(moveRail("cancelled")).toBeNull();
  });
});
