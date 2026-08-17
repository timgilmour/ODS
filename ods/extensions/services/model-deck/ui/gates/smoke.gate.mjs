import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createResults } from "./lib/check.mjs";
import { launch } from "./lib/browser.mjs";
import { startStub } from "./stub-server.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const name = "smoke";

export async function run() {
  const results = createResults();
  const scenario = JSON.parse(
    await readFile(join(HERE, "fixtures/smoke/scenario.json"), "utf8"),
  );
  const stub = await startStub({ scenario, distDir: join(HERE, "../dist"), port: 0 });
  const { browser, page, consoleErrors } = await launch();
  try {
    await page.goto(stub.url, { waitUntil: "networkidle" });
    const title = await page.title();
    results.check("built bundle loads from the stub origin", title.length > 0, title);
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    results.check("always-dark body background", bg === "rgb(20, 22, 26)", bg);
    results.check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));
  } finally {
    await browser.close();
    await stub.stop();
  }
  return results;
}
