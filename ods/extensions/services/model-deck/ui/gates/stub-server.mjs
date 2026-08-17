import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json; charset=utf-8",
};

/** Serve one scenario plus (optionally) the built UI, on ONE origin — the
 * same shape the production container serves (dist statically at "/" beside
 * /api), so the gate exercises the artifact that actually ships rather than
 * a dev-server transform behind a proxy.
 *
 * The stub NEVER derives state (design §5). It replays a scripted sequence
 * per route and records what arrived. Anything it cannot answer from the
 * script is a 599 with the route in the body — an unmistakable status that
 * belongs to no real endpoint.
 *
 * R7 (controller ruling): a route's entries replay in order. The LAST entry
 * of a route, and only the last, may carry `repeat: true`, meaning "serve
 * this response for every subsequent request to this route" — required for
 * polling routes (App.tsx POLL_MS = 3000 re-fetches /api/state and
 * /api/storage/state every tick; a fixed-length script would drain within
 * seconds and die on a 599 that means nothing). `repeat` on any entry that
 * is NOT last is refused at startup, naming the route — otherwise an author
 * could mark entry 1 repeat and silently make entries 2..n dead. Exhaustion
 * of a route whose last entry is NOT marked repeat still 599s, exactly as a
 * scripted state transition must: repeat is opt-in and visible in the
 * fixture, never an implicit server-side fallback. */
export async function startStub({ scenario, distDir, port = 0 }) {
  for (const [key, entries] of Object.entries(scenario.routes)) {
    const nonLastRepeat = entries.slice(0, -1).some((e) => e.repeat);
    if (nonLastRepeat) {
      throw new Error(
        `deck-gate stub: "repeat: true" is only valid on the LAST entry of ` +
          `a route. Route "${key}" marks an earlier entry as repeat, which ` +
          `would silently strip the entries after it.`,
      );
    }
  }

  const remaining = new Map(
    Object.entries(scenario.routes).map(([k, v]) => [k, [...v]]),
  );
  const requests = [];

  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const key = `${req.method} ${url.pathname}`;

    if (url.pathname.startsWith("/api") || url.pathname === "/health") {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      requests.push({
        method: req.method,
        path: url.pathname,
        body: raw ? JSON.parse(raw) : null,
      });
      const queue = remaining.get(key);
      if (!queue || queue.length === 0) {
        res.writeHead(599, { "content-type": "text/plain" });
        res.end(
          `deck-gate stub: no scripted response left for ${key}. ` +
            `Either the scenario is short or the UI stopped dispatching it.`,
        );
        return;
      }
      // A repeat-marked entry is only ever the last one left (validated at
      // startup) — peek it forever instead of shifting it away.
      const next = queue.length === 1 && queue[0].repeat ? queue[0] : queue.shift();
      res.writeHead(next.status, { "content-type": "application/json" });
      res.end(JSON.stringify(next.body));
      return;
    }

    if (!distDir) {
      res.writeHead(404).end();
      return;
    }
    const rel = url.pathname === "/" ? "index.html" : normalize(url.pathname).slice(1);
    try {
      const buf = await readFile(join(distDir, rel));
      res.writeHead(200, { "content-type": TYPES[extname(rel)] ?? "application/octet-stream" });
      res.end(buf);
    } catch {
      // SPA fallback: unknown non-asset paths serve index.html.
      const buf = await readFile(join(distDir, "index.html"));
      res.writeHead(200, { "content-type": TYPES[".html"] });
      res.end(buf);
    }
  });

  return new Promise((resolve) => {
    server.listen(port, "127.0.0.1", () => {
      resolve({
        url: `http://127.0.0.1:${server.address().port}`,
        requests: () => requests.slice(),
        stop: () => new Promise((r) => server.close(r)),
      });
    });
  });
}
