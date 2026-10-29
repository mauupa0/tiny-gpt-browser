// Usage: node scripts/og.mjs <url> <out.png>
// Renders the share image: the first screen at 1330x698 (ends under the buttons, above the next divider) scaled to exactly 1200x630, taken once a
// poem has finished and the selected letter has a candidate spread worth looking at (not a 100% bar).
// Needs a global `playwright`.
import { createRequire } from "node:module";

const { chromium } = createRequire(process.env.GLOBAL_NODE_MODULES ? process.env.GLOBAL_NODE_MODULES + '/' : import.meta.url)("playwright");
const [url, out] = [process.argv[2], process.argv[3]];
const browser = await chromium.launch({ args: ["--enable-gpu", "--use-angle=d3d11"] });
const ctx = await browser.newContext({ viewport: { width: 1330, height: 698 }, deviceScaleFactor: 1200 / 1330 });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: "load" });
await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
await page.locator(".switch input").check({ force: true });

const spread = () =>
  page.evaluate(() => {
    const ps = [...document.querySelectorAll(".cands li:not(.empty) .p")].map((e) => parseFloat(e.textContent.replace("<", "")));
    return { top: ps[0], second: ps[1], n: ps.length, on: !!document.querySelector(".cands li.on") };
  });

let picked = false;
for (let i = 0; i < 12 && !picked; i++) {
  await page.getByRole("button", { name: /^(Write again|Write)$/ }).click();
  await page.getByRole("button", { name: "Write again" }).waitFor({ timeout: 30000 });
  await page.waitForFunction(() => !document.querySelector(".cands.stale") && document.querySelector(".plate")?.getAttribute("aria-busy") === "false", null, { timeout: 15000 });
  const s = await spread();
  console.log(`try ${i + 1}: top ${s.top}% second ${s.second}% chosen marked ${s.on}`);
  picked = s.n === 8 && s.top < 55 && s.top > 25 && s.second > 10 && s.on;
}
await page.locator(".switch input").uncheck({ force: true });
await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
await page.waitForTimeout(1200);
await page.mouse.move(0, 700);
await page.screenshot({ path: out });
console.log(picked ? "picked a spread letter" : "no ideal letter in 12 tries, kept the last", out);
await browser.close();
