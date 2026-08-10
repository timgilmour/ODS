import { describe, expect, it } from "vitest";
import { eventSeverity } from "./eventSeverity";

// Kinds below are real, taken from app/arbiter.py, app/sets.py,
// app/storage.py and app/mover.py's log_event/_log call sites — not
// invented labels. The vocabulary mixes "-" and "_" as separators
// depending on which module logs the kind; both conventions are exercised.

describe("eventSeverity — failure", () => {
  it("classifies a hyphenated -failed kind", () => {
    expect(eventSeverity("load-failed")).toBe("failure");
  });

  it("classifies an underscored _failed kind the same as a hyphenated one", () => {
    expect(eventSeverity("move_failed")).toBe("failure");
  });

  it("classifies -vetoed", () => {
    expect(eventSeverity("apply-vetoed")).toBe("failure");
  });

  it("classifies -unreachable", () => {
    expect(eventSeverity("lifecycle-spark-unreachable")).toBe("failure");
  });

  it("classifies -quarantined", () => {
    expect(eventSeverity("lifecycle-quarantined")).toBe("failure");
  });

  it("classifies -error, including the bare tick-error kind", () => {
    expect(eventSeverity("tick-error")).toBe("failure");
    expect(eventSeverity("storage-tick-error")).toBe("failure");
  });

  it("classifies a failed restore as a failure, not a success — it ends in -failed, not -restore", () => {
    expect(eventSeverity("lifecycle-restore-failed")).toBe("failure");
  });
});

describe("eventSeverity — outcome outranks the suffix", () => {
  it("classifies a FAILED apply-end as a failure, not a success", () => {
    // app/sets.py logs BOTH terminal results under the one kind "apply-end":
    // :835 {"outcome": "failed", "step", "error"} and :850 {"outcome": "ok"}.
    // The kind alone therefore cannot classify it, and the suffix rule made
    // a failed apply render GREEN [max-review #14].
    expect(eventSeverity("apply-end", { outcome: "failed", step: "load_lemonade" }))
      .toBe("failure");
  });

  it("leaves a successful apply-end a success", () => {
    expect(eventSeverity("apply-end", { outcome: "ok" })).toBe("success");
  });

  it("falls back to the suffix convention when there is no outcome", () => {
    // The bare-kind expectation below stays valid; only the failed case was
    // ever wrong. Also covers a detail that is not an object at all.
    expect(eventSeverity("apply-end", { step: "load_lemonade" })).toBe("success");
    expect(eventSeverity("apply-end", null)).toBe("success");
    expect(eventSeverity("apply-end", "nonsense")).toBe("success");
  });

  it("does not let an outcome field promote an unrelated kind", () => {
    // Only "failed" outranks; nothing else in a detail may reclassify.
    expect(eventSeverity("noop", { outcome: "ok" })).toBe("neutral");
  });
});

describe("eventSeverity — attention kinds without the suffix", () => {
  it("flags a superseded pull-through: the requested load did NOT happen", () => {
    // app/routers/control.py logs this when an operator action overtook a
    // pull-through mid-copy. Neutral would read as "nothing to see here".
    expect(eventSeverity("pull-through-superseded")).toBe("attention");
  });

  it("flags an unknown policy tenant", () => {
    // app/policy.py's boundary gate (task 3) surfaces a tenant it dropped.
    expect(eventSeverity("policy-unknown-tenant")).toBe("attention");
  });

  it("classifies the per-artifact update-check error by its suffix", () => {
    // Two kinds, two DETAIL SHAPES, deliberately distinct (app/update_check.py):
    // the whole-pass "update-check-error" carries {error}, the per-artifact
    // one adds {artifact_id}. Both end in "-error", so both are failures with
    // no new mapping — pinned so a future rename can't silently drop one.
    expect(eventSeverity("update-check-error")).toBe("failure");
    expect(eventSeverity("update-check-artifact-error")).toBe("failure");
  });
});

describe("eventSeverity — success", () => {
  it("classifies -end", () => {
    expect(eventSeverity("apply-end")).toBe("success");
  });

  it("classifies an underscored _done kind", () => {
    expect(eventSeverity("move_done")).toBe("success");
  });

  it("classifies -restore on its own (not failed)", () => {
    expect(eventSeverity("lifecycle-restore")).toBe("success");
  });

  it("classifies the named exception reconciled, if it ever appears", () => {
    expect(eventSeverity("reconciled")).toBe("success");
  });
});

describe("eventSeverity — attention", () => {
  it("classifies -warn", () => {
    expect(eventSeverity("apply-warn")).toBe("attention");
  });

  it("classifies the named exception storage_shortfall", () => {
    expect(eventSeverity("storage_shortfall")).toBe("attention");
  });

  it("classifies the named exception host-agent-busy", () => {
    expect(eventSeverity("host-agent-busy")).toBe("attention");
  });

  it("classifies the named exception free-raced", () => {
    expect(eventSeverity("free-raced")).toBe("attention");
  });
});

describe("eventSeverity — neutral", () => {
  it("classifies real routine kinds as neutral", () => {
    expect(eventSeverity("noop")).toBe("neutral");
    expect(eventSeverity("activate")).toBe("neutral");
    expect(eventSeverity("park_hipfire")).toBe("neutral");
    expect(eventSeverity("resume_hipfire")).toBe("neutral");
    expect(eventSeverity("policy_patch")).toBe("neutral");
    expect(eventSeverity("unload_lemonade")).toBe("neutral");
    expect(eventSeverity("free_comfyui")).toBe("neutral");
    expect(eventSeverity("load-retriggered")).toBe("neutral");
    expect(eventSeverity("move_cancelled")).toBe("neutral");
    expect(eventSeverity("storage_skip")).toBe("neutral");
    expect(eventSeverity("storage_notify_deferred")).toBe("neutral");
  });

  it("falls through to neutral for a kind nobody enumerated here — the whole point of matching by convention", () => {
    expect(eventSeverity("some-future-kind-nobody-wrote-yet")).toBe("neutral");
  });
});

describe("update-checking kinds", () => {
  it("classifies a moved origin as attention", () => {
    expect(eventSeverity("origin-moved")).toBe("attention");
  });

  it("classifies a failed check as a failure via the existing suffix", () => {
    expect(eventSeverity("update-check-failed")).toBe("failure");
  });

  // Two kinds, because they carry two different detail shapes:
  // `update-check-failed` is per source ({artifact_id, source, note},
  // app/update_check.py's _log_transitions); `update-check-error` is the
  // tick-level supervisor catch ({error}, UpdateChecker.tick). Both must
  // still read as failures in the Events tab.
  it("classifies the tick-level check error as a failure too", () => {
    expect(eventSeverity("update-check-error")).toBe("failure");
  });

  it("leaves an available update neutral — informational, not an alarm", () => {
    expect(eventSeverity("update-available")).toBe("neutral");
  });
});
