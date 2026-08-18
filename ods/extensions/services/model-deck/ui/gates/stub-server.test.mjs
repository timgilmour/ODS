import { describe, it, expect, afterEach } from "vitest";
import { startStub } from "./stub-server.mjs";

let running = null;
afterEach(async () => {
  if (running) await running.stop();
  running = null;
});

const scenario = () => ({
  routes: {
    "GET /api/state": [{ status: 200, body: { n: 1 } }, { status: 200, body: { n: 2 } }],
    "POST /api/nodes/local/engines": [{ status: 201, body: { resource: "gguf-test" } }],
  },
});

describe("stub server", () => {
  it("replays a route's responses in order", async () => {
    running = await startStub({ scenario: scenario(), distDir: null, port: 0 });
    const first = await (await fetch(running.url + "/api/state")).json();
    const second = await (await fetch(running.url + "/api/state")).json();
    expect(first).toEqual({ n: 1 });
    expect(second).toEqual({ n: 2 });
  });

  it("fails loudly when a route's script runs out", async () => {
    // A stub that re-serves the last response when the script is exhausted
    // turns "the UI stopped making this request" into a GREEN gate. Running
    // out is a defect in the scenario or in the UI; it must be loud either
    // way. 599 is outside the app's vocabulary on purpose.
    running = await startStub({ scenario: scenario(), distDir: null, port: 0 });
    await fetch(running.url + "/api/state");
    await fetch(running.url + "/api/state");
    const third = await fetch(running.url + "/api/state");
    expect(third.status).toBe(599);
    expect(await third.text()).toContain("GET /api/state");
  });

  it("fails loudly on a route the scenario never declared", async () => {
    running = await startStub({ scenario: scenario(), distDir: null, port: 0 });
    const res = await fetch(running.url + "/api/policy");
    expect(res.status).toBe(599);
    expect(await res.text()).toContain("GET /api/policy");
  });

  it("records what the UI dispatched, with parsed bodies", async () => {
    // This recording IS the dispatch gap: asserting that a click produced
    // the right request is the thing no unit test in this repo can do.
    running = await startStub({ scenario: scenario(), distDir: null, port: 0 });
    await fetch(running.url + "/api/nodes/local/engines", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ resource: "gguf-test", kind: "lemonade" }),
    });
    expect(running.requests()).toEqual([
      {
        method: "POST",
        path: "/api/nodes/local/engines",
        body: { resource: "gguf-test", kind: "lemonade" },
      },
    ]);
  });

  // --- R7: repeat is right for polling routes, wrong for scripted
  // transitions. `repeat: true` on a route's LAST entry serves that
  // response forever; anywhere else it's a startup error naming the route,
  // never a silent no-op (an author marking entry 1 as repeat must not
  // silently kill entries 2..n).

  it("serves a repeat-marked last entry indefinitely (polling routes)", async () => {
    running = await startStub({
      scenario: {
        routes: {
          "GET /api/state": [
            { status: 200, body: { n: 1 } },
            { status: 200, body: { n: 2, done: true }, repeat: true },
          ],
        },
      },
      distDir: null,
      port: 0,
    });
    const first = await (await fetch(running.url + "/api/state")).json();
    const second = await (await fetch(running.url + "/api/state")).json();
    const third = await (await fetch(running.url + "/api/state")).json();
    const fourth = await (await fetch(running.url + "/api/state")).json();
    expect(first).toEqual({ n: 1 });
    expect(second).toEqual({ n: 2, done: true });
    expect(third).toEqual({ n: 2, done: true });
    expect(fourth).toEqual({ n: 2, done: true });
  });

  it("refuses at startup when repeat is set on a non-last entry", async () => {
    await expect(
      startStub({
        scenario: {
          routes: {
            "GET /api/state": [
              { status: 200, body: { n: 1 }, repeat: true },
              { status: 200, body: { n: 2 } },
            ],
          },
        },
        distDir: null,
        port: 0,
      }),
    ).rejects.toThrow(/GET \/api\/state/);
  });

  it("still 599s on exhaustion when the last entry is not marked repeat", async () => {
    running = await startStub({
      scenario: {
        routes: {
          "GET /api/state": [{ status: 200, body: { n: 1 } }],
        },
      },
      distDir: null,
      port: 0,
    });
    await fetch(running.url + "/api/state");
    const res = await fetch(running.url + "/api/state");
    expect(res.status).toBe(599);
    expect(await res.text()).toContain("GET /api/state");
  });
});
