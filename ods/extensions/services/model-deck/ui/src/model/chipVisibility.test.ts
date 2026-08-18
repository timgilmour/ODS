import { describe, expect, it } from "vitest";
import type { LifecycleStatus } from "../api";
import { engineChipVisible } from "./chipVisibility";

// Ruling 2026-08-18: an ENGINE-kind chip (comfyui local, any remote declared
// engine) earns its place on the board only by holding a model — except a
// failure/transition, which must stay visible so the operator can see it
// crashed or is quarantined rather than assuming it quietly went away.

describe("engineChipVisible", () => {
  it.each(["down", "quarantined", "warming"] as const)(
    "shows any kind when the lifecycle status is %s, even with a benign engine state",
    (status: LifecycleStatus) => {
      // comfyui: state/queue alone would hide it (idle, queue 0) — the
      // failure status must override that.
      expect(engineChipVisible("comfyui", "idle", 0, status)).toBe(true);
      // sglang-omni: state "down" alone would hide it (not resident) — same
      // override.
      expect(engineChipVisible("sglang-omni", "down", null, status)).toBe(true);
    },
  );

  describe("sglang-omni (load/unload kind — busy|idle ARE resident, app/engine_kinds.py:949-955)", () => {
    it("shows when idle — weights stay on the GPU between renders", () => {
      expect(engineChipVisible("sglang-omni", "idle", null, undefined)).toBe(true);
    });

    it("shows when busy", () => {
      expect(engineChipVisible("sglang-omni", "busy", null, undefined)).toBe(true);
    });

    it("hides when down, with a benign (non-failure) lifecycle status", () => {
      expect(engineChipVisible("sglang-omni", "down", null, "idle")).toBe(false);
    });

    it("hides when unknown ('failed to look'), with no lifecycle entry at all", () => {
      expect(engineChipVisible("sglang-omni", "unknown", null, undefined)).toBe(false);
    });
  });

  describe("comfyui (app/engine_kinds.py:613-635 — 'idle' describes the request queue, NOT residency)", () => {
    it("hides when idle with an empty queue — the opposite reading from sglang-omni's 'idle'", () => {
      expect(engineChipVisible("comfyui", "idle", 0, undefined)).toBe(false);
    });

    it("hides when idle with no queue reading at all (null, not zero)", () => {
      expect(engineChipVisible("comfyui", "idle", null, undefined)).toBe(false);
    });

    it("shows when busy, regardless of queue depth", () => {
      expect(engineChipVisible("comfyui", "busy", 0, undefined)).toBe(true);
    });

    it("shows when the queue is non-empty, even while the reported state is idle", () => {
      expect(engineChipVisible("comfyui", "idle", 3, undefined)).toBe(true);
    });
  });

  it.each(["idle", "parked", "unmanaged", "serving"] as const)(
    "never lets the benign lifecycle status %s force a chip on its own — only down/quarantined/warming do that",
    (status: LifecycleStatus) => {
      expect(engineChipVisible("sglang-omni", "down", null, status)).toBe(false);
      expect(engineChipVisible("comfyui", "idle", 0, status)).toBe(false);
    },
  );
});
