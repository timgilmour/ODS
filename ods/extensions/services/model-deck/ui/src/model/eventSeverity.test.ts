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
  });

  it("falls through to neutral for a kind nobody enumerated here — the whole point of matching by convention", () => {
    expect(eventSeverity("some-future-kind-nobody-wrote-yet")).toBe("neutral");
  });
});
