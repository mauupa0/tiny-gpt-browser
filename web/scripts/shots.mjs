// Usage: node scripts/shots.mjs <url> <outDir>
// Screenshots of every chapter at 1440 and 360 px once the writing has settled and the figure
// has caught up with the selection. Needs a global `playwright`.
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

const { chromium } = createRequire(process.env.GLOBAL_NODE_MODULES ? process.env.GLOBAL_NODE_MODULES + '/' : import.meta.url)("playwright");
const [url, out] = [process.argv[2], process.argv[3]];
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ["--enable-gpu", "--use-angle=d3d11"] });

for (const [tag, vp, dpr] of [["1440", { width: 1440, height: 900 }, 1], ["360", { width: 360, height: 780 }, 2]]) {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: dpr });
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  await page.getByRole("button", { name: "Write again" }).waitFor({ timeout: 90000 });
  await page.waitForFunction(() => !document.querySelector(".cands.stale") && document.querySelector(".plate")?.getAttribute("aria-busy") === "false", null, { timeout: 15000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(out, `hero-${tag}.png`) });
  for (const [name, sel, wait] of [["inside", ".plate", 900], ["heads", "#heads", 900], ["temperature", "#temperature", 16000], ["break", "#break", 16000], ["proof", "#proof", 900]]) {
    await page.evaluate((s) => document.querySelector(s).scrollIntoView({ block: "start" }), sel);
    await page.waitForTimeout(wait);
    await page.locator(sel).first().screenshot({ path: join(out, `${name}-${tag}.png`) });
  }
  await ctx.close();
}
await browser.close();
console.log("done", out);
