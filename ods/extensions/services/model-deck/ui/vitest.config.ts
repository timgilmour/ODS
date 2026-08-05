import { defineConfig } from "vitest/config";

// Node environment only: the tested modules are pure data transforms with
// no DOM. Component/DOM testing is deliberately out of scope — the
// deck-drill livetests cover real behaviour end to end.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
