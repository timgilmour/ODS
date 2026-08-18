import { chromium } from "playwright-core";

/** Launch system Chrome headless. `channel: "chrome"` uses the installed
 * Google Chrome (/usr/bin/google-chrome) instead of downloading a ~400 MB
 * bundled build — the same choice the 2026-08-06 browser pass made.
 *
 * Console errors and pageerrors are collected, not printed and forgotten:
 * the gate scripts assert on them. */
export async function launch() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push("pageerror: " + e.message));
  return { browser, page, consoleErrors };
}
