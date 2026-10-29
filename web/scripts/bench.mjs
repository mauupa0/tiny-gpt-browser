// Usage: node scripts/bench.mjs <url> [runs]
// Presses "Measure on this device" several times (400 letters at full speed inside the worker,
// model forward pass and sampling) and prints the median letters per second.
import { createRequire } from "node:module";

const { chromium } = createRequire(process.env.GLOBAL_NODE_MODULES ? process.env.GLOBAL_NODE_MODULES + '/' : import.meta.url)("playwright");
const url = process.argv[2];
const runs = Number(process.argv[3] ?? 5);
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(url, { waitUntil: "load" });
await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
await page.getByRole("button", { name: /^(Write|Write again)$/ }).waitFor({ timeout: 60000 });
await page.evaluate(() => document.querySelector("#proof").scrollIntoView());
const out = [];
for (let i = 0; i < runs; i++) {
  await page.getByRole("button", { name: "Measure on this device" }).click();
  await page.getByRole("button", { name: "Measure on this device" }).waitFor({ timeout: 30000 });
  out.push(Number(await page.$eval(".speed .big", (e) => e.textContent.replace(/,/g, ""))));
}
out.sort((a, b) => a - b);
console.log(JSON.stringify({ runs: out, median: out[Math.floor(out.length / 2)], ua: await page.evaluate(() => navigator.userAgent) }));
await browser.close();
