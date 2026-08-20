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
    // :1111 {"outcome": "failed", "step", "error"} and :1134 {"outcome": "ok"}.
    // The kind alone therefore cannot classify it, and the suffix rule made
    // a failed apply render GREEN [max-review #14].
    expect(eventSeverity("apply-end", { outcome: "failed", step: "load" }))
      .toBe("failure");
  });

  it("leaves a successful apply-end a success", () => {
    expect(eventSeverity("apply-end", { outcome: "ok" })).toBe("success");
  });

  it("falls back to the suffix convention when there is no outcome", () => {
    // The bare-kind expectation below stays valid; only the failed case was
    // ever wrong. Also covers a detail that is not an object at all.
    expect(eventSeverity("apply-end", { step: "load" })).toBe("success");
    expect(eventSeverity("apply-end", null)).toBe("success");
    expect(eventSeverity("apply-end", "nonsense")).toBe("success");
  });

  it("does not let an outcome field promote an unrelated kind", () => {
    // Only "failed" outranks; nothing else in a detail may reclassify.
    expect(eventSeverity("noop", { outcome: "ok" })).toBe("neutral");
  });

  it("still classifies a resource-tagged failed apply-end as failure (E1 Task 8's resource field must not disturb the outcome check)", () => {
    // app/sets.py's failing-step branch now also carries "resource" in the
    // detail (T8 review M1) — the apply-end-failures-render-green bug class
    // this whole describe block guards against, re-pinned with the new
    // resource-tagged shape so a regression there is caught the same way.
    expect(
      eventSeverity("apply-end", {
        outcome: "failed", step: "unload", resource: "gguf-a", error: "boom",
      }),
    ).toBe("failure");
  });
});

describe("eventSeverity — exact overrides that beat the suffix convention", () => {
  it("classifies notify-restart-failed as attention, not failure, despite the -failed suffix", () => {
    // app/notify.py:102-104: one resource's restart failure, logged in
    // ISOLATION — every sibling resource sharing that destination still
    // gets its own restart attempt regardless (the module's per-resource
    // Let-It-Crash design), and the eventual raise that fails the calling
    // job already gets its own "-failed"/502 event elsewhere. This is a
    // degraded-but-isolated per-resource notice, not the terminal failure.
    expect(eventSeverity("notify-restart-failed")).toBe("attention");
  });

  it("still classifies an UNRELATED kind's real -failed suffix as failure — the override is exact-match only, not a blanket exemption", () => {
    expect(eventSeverity("load-failed")).toBe("failure");
  });
});

describe("eventSeverity — attention kinds without the suffix", () => {
  it("flags a superseded pull-through: the requested load did NOT happen", () => {
    // app/routers/control.py logs this when an operator action overtook a
    // pull-through mid-copy. Neutral would read as "nothing to see here".
    expect(eventSeverity("pull-through-superseded")).toBe("attention");
  });

  it("flags a misconfigured node", () => {
    // app/arbiter.py's _node_observations logs this when a node-agent's
    // serving payload carries its probe-URL warning (node-agent serving.py's
    // PROBE_URL_WARNING: vllm profiles configured, probe URL unset —
    // detection blind). Amber per the colour rule: it wants a decision
    // (fix the node's env), nothing has failed yet.
    expect(eventSeverity("lifecycle-node-misconfigured")).toBe("attention");
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
    // E1 Task 8 renamed the per-step success kinds from kind-suffixed
    // (park_hipfire, unload_lemonade, free_comfyui, ...) to bare
    // verb-generic ones (app/sets.py's `_run_apply`: `log_event(events_path,
    // name, detail)` where `name = step["step"]`) — still unmatched by any
    // suffix/exact table, so still neutral, just under the new spelling.
    expect(eventSeverity("noop")).toBe("neutral");
    expect(eventSeverity("activate")).toBe("neutral");
    expect(eventSeverity("park")).toBe("neutral");
    expect(eventSeverity("resume")).toBe("neutral");
    expect(eventSeverity("policy_patch")).toBe("neutral");
    expect(eventSeverity("restore_settings")).toBe("neutral");
    expect(eventSeverity("unload")).toBe("neutral");
    expect(eventSeverity("load")).toBe("neutral");
    expect(eventSeverity("free")).toBe("neutral");
    expect(eventSeverity("load-retriggered")).toBe("neutral");
    expect(eventSeverity("move_cancelled")).toBe("neutral");
    expect(eventSeverity("storage_skip")).toBe("neutral");
    expect(eventSeverity("storage_notify_deferred")).toBe("neutral");
  });

  it("a resource-tagged detail never disturbs a bare verb-generic step's neutral classification", () => {
    // Kind alone still decides here — "resource" riding in the detail
    // (E1 Task 8) is not "outcome", so it cannot promote/demote anything.
    expect(eventSeverity("unload", { resource: "gguf-a", model: "extra.m.gguf" })).toBe("neutral");
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

describe("instance lifecycle kinds (INST I1)", () => {
  it("instance lifecycle kinds", () => {
    expect(eventSeverity("instance-created")).toBe("success");
    expect(eventSeverity("instance-removed")).toBe("success");
    expect(eventSeverity("instance-move-requested")).toBe("attention");
    expect(eventSeverity("instance-create-failed")).toBe("failure");
  });
});
