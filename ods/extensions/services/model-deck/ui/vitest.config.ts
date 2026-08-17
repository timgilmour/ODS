import { defineConfig } from "vitest/config";

// Node environment only: the tested modules are pure data transforms with
// no DOM. Component/DOM testing does not happen HERE — it happens in
// ui/gates/, where real headless Chrome drives the built bundle
// (deck-gate; ~/notes/designs/2026-08-17-model-deck-browser-gate-harness-design.md).
// What this config covers is the pure half: src/model/* and the gate
// harness's own primitives.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "gates/**/*.test.mjs"],
  },
});
