import { describe, expect, it } from "vitest";
import { forceParkCanOverride } from "./refusals";

describe("forceParkCanOverride", () => {
  // Producer: app/engines/docker_ctl.py _guard —
  // f"container {name!r} is not in the park allowlist". ?force=true never
  // skips this guard, so Force must not be offered.
  it("is false for the park-allowlist refusal", () => {
    expect(
      forceParkCanOverride("container 'ods-newthing' is not in the park allowlist"),
    ).toBe(false);
  });

  // Producer: app/engines/hipfire.py park() — raised BEFORE the force check,
  // so force cannot skip it either.
  it("is false for the default-route refusal", () => {
    expect(
      forceParkCanOverride(
        "refusing to park 'ods-hipfire': litellm's default route currently targets hipfire",
      ),
    ).toBe(false);
  });

  // Producer: app/engines/hipfire.py ensure_not_busy — the guard
  // ?force=true exists to skip.
  it("is true for the activity-window refusal", () => {
    expect(
      forceParkCanOverride(
        "refusing to park 'ods-hipfire': hipfire served a request 12s ago " +
          "(activity window 300s; pass force=true to override)",
      ),
    ).toBe(true);
  });

  // Producer: app/engines/hipfire.py ensure_not_busy queue-depth branch —
  // also skipped wholesale by force.
  it("is true for the request-in-flight refusal", () => {
    expect(
      forceParkCanOverride(
        "refusing to park 'ods-hipfire': hipfire request in flight (queue_depth=1)",
      ),
    ).toBe(true);
  });

  // Unknown refusal text fails OPEN: a future guard degrades to today's
  // behavior (Force offered, click 409s again), never to a hidden button.
  it("fails open on unknown refusal text", () => {
    expect(forceParkCanOverride("some future guard nobody wrote yet")).toBe(true);
  });
});
