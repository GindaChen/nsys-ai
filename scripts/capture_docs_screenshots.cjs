#!/usr/bin/env node

/* Capture the browser surfaces used by the documentation.
 *
 * Playwright is intentionally not a runtime or development dependency of
 * nsys-ai. Install it in a temporary directory and set NODE_PATH to that
 * directory before running this script; see docs/images/README.md.
 */

const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const match = process.argv[i].match(/^--([^=]+)=(.*)$/);
  if (match) args.set(match[1], match[2]);
}

const outputDir = path.resolve(args.get("output") || "docs/images");
const webUrl = args.get("web-url") || "http://127.0.0.1:18242/";
const timelineUrl = args.get("timeline-url") || "http://127.0.0.1:18244/";
const diffUrl = args.get("diff-url") || "http://127.0.0.1:18245/";

async function ready(page, url, waitMs = 2500) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.waitForTimeout(waitMs);
}

async function main() {
  fs.mkdirSync(outputDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });

  const tree = await context.newPage();
  await ready(tree, webUrl);
  await tree.screenshot({ path: path.join(outputDir, "web-tree.png") });
  await tree.close();

  const guided = await context.newPage();
  await ready(guided, timelineUrl, 5000);
  const loopPanel = guided.locator("#loopSidebar");
  if (!(await loopPanel.isVisible())) {
    await guided.locator("#loopBtn").click();
    await loopPanel.waitFor({ state: "visible", timeout: 10_000 });
  }
  await guided.screenshot({ path: path.join(outputDir, "guided-loop.png") });
  await guided.close();

  const timeline = await context.newPage();
  await ready(timeline, timelineUrl, 5000);
  const closeButton = timeline.locator("#inspectorRail .inspector-close");
  if (await closeButton.isVisible()) await closeButton.click();
  await timeline.screenshot({ path: path.join(outputDir, "timeline-web.png") });
  await timeline.close();

  const diff = await context.newPage();
  await ready(diff, diffUrl, 5000);
  await diff.screenshot({ path: path.join(outputDir, "diff-web.png") });
  await diff.close();

  await browser.close();
  console.log(`Captured documentation screenshots in ${outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
