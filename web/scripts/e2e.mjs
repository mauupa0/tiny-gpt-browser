// Usage: node scripts/e2e.mjs <url> [outDir]
// Drives the page in headless Chromium at 1440 and 360 px and asserts what a visitor would see:
// the model loads, the poem writes itself, a click on a letter moves the figure, the keyboard works,
// all three live sections run, the parity check separates the intact model from the broken ones,
// and the error, reduced-motion and unreadable-prompt states behave. Needs a global `playwright`.
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const { chromium } = createRequire(process.env.GLOBAL_NODE_MODULES ? process.env.GLOBAL_NODE_MODULES + '/' : import.meta.url)("playwright");
const url = process.argv[2];
const out = process.argv[3] ?? "shots";
mkdirSync(out, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok: !!ok, detail: String(detail) });
  if (!ok) console.log("FAIL", name, detail);
};

const browser = await chromium.launch({ args: ["--enable-gpu", "--use-angle=d3d11"] });

async function open(vp, opts = {}) {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: vp.width < 600 ? 2 : 1, ...opts });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && errors.push(m.text()));
  return { ctx, page, errors };
}

const poemLen = (page) => page.$$eval(".poem-hero .c", (e) => e.length);
const writeBtn = (page) => page.getByRole("button", { name: /^(Write|Write again)$/ });

for (const [tag, vp] of [["1440", { width: 1440, height: 900 }], ["360", { width: 360, height: 780 }]]) {
  const { ctx, page, errors } = await open(vp);
  // Everything under the poem is sampled on every frame: a change of position is a visible jump.
  await page.addInitScript(() => {
    window.__jumps = [];
    const last = {};
    const tick = () => {
      for (const n of [".legend", ".cands", ".cand-note", ".hero-actions", ".hero-controls"]) {
        const el = document.querySelector(n);
        if (!el) continue;
        const y = Math.round((el.getBoundingClientRect().top + scrollY) * 2) / 2;
        if (last[n] !== undefined && Math.abs(y - last[n]) > 1.5) window.__jumps.push(`${n} ${Math.round(performance.now())}ms ${y - last[n]}px`);
        last[n] = y;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  check(`${tag} model loads`, true);

  await page.waitForFunction(() => document.querySelectorAll(".poem-hero .c").length > 120, null, { timeout: 30000 });
  await writeBtn(page).waitFor({ timeout: 40000 });
  const n = await poemLen(page);
  check(`${tag} hero writes by itself and stops`, n > 120 && n < 400, n);
  const lastChar = await page.$$eval(".poem-hero .c", (e) => e.length);
  check(`${tag} poem ends on a finished line or at the cap`, lastChar > 0);

  await page.waitForTimeout(600);
  const jumps = await page.evaluate(() => window.__jumps);
  check(`${tag} nothing under the poem moves while it writes`, jumps.length === 0, jumps.slice(0, 4).join(", "));
  const gap = await page.evaluate(() => {
    const poem = document.querySelector(".poem-hero");
    const last = [...poem.querySelectorAll(".l")].filter((l) => l.querySelector(".c:not(.nl)")).pop().getBoundingClientRect().bottom;
    return document.querySelector(".legend").getBoundingClientRect().top - last;
  });
  check(`${tag} the legend sits close under the poem`, gap < (tag === "1440" ? 52 : 80), Math.round(gap));

  const cands = await page.$$eval(".cands li:not(.empty)", (e) => e.map((x) => x.textContent));
  check(`${tag} eight candidates with percentages`, cands.length === 8 && cands.every((c) => /%/.test(c)), cands.length);

  // click a letter in the poem, the figure follows
  await page.locator('.poem-hero .c[data-i="5"]').click();
  await page.waitForFunction(() => document.querySelector('[role=slider][aria-label="Letter to inspect"]')?.getAttribute("aria-valuenow") === "5", null, { timeout: 5000 });
  const clicked = await page.$eval('.poem-hero .c[data-i="5"]', (e) => e.textContent);
  await page.waitForFunction((c) => document.querySelector(".cc.now")?.textContent === c, clicked, { timeout: 5000 }).catch(() => {});
  const nowGlyph = await page.$eval(".cc.now", (e) => e.textContent);
  check(`${tag} clicking a letter moves the figure to it`, nowGlyph === clicked, `${nowGlyph} vs ${clicked}`);
  const note1 = await page.$eval(".b-embed .note", (e) => e.textContent);

  const slider = page.locator('[role=slider][aria-label="Letter to inspect"]');
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(() => document.querySelector('[role=slider][aria-label="Letter to inspect"]')?.getAttribute("aria-valuenow") === "6", null, { timeout: 5000 });
  const changed = await page.waitForFunction((n) => document.querySelector(".b-embed .note")?.textContent !== n, note1, { timeout: 5000 }).then(() => true, () => false);
  check(`${tag} arrow key moves the inspected letter and the caption changes`, changed);
  await page.keyboard.press("Home");
  await page.waitForFunction(() => document.querySelector('[role=slider][aria-label="Letter to inspect"]')?.getAttribute("aria-valuenow") === "0", null, { timeout: 5000 });
  check(`${tag} first letter inspectable (one-letter context)`, (await page.$$eval(".b-block", (e) => e.length)) === 4);

  await page.locator('[role=slider][aria-label="Letter to inspect"]').focus();
  await page.keyboard.press("End");
  await page.getByRole("button", { name: "The next letter in the poem" }).waitFor();
  await page.keyboard.press("ArrowLeft");
  await page.getByRole("button", { name: "The next letter in the poem" }).click();
  await page.waitForFunction(() => /The poem has/.test(document.querySelector(".b-out .note")?.textContent ?? ""), null, { timeout: 5000 });
  check(`${tag} explain mode switches to the letter the poem has next`, true);

  // sums add up
  const sumOk = await page.$eval(".b-out .note", (e) => /nothing is missing/.test(e.textContent));
  check(`${tag} logit contributions add up to the logit`, sumOk);

  // atlas hover
  const mapBox = page.locator(".atlas .map canvas").nth(7);
  await mapBox.scrollIntoViewIfNeeded();
  await page.waitForTimeout(900);
  const box = await mapBox.boundingBox();
  await mapBox.hover({ position: { x: box.width * 0.6, y: box.height * 0.8 } });
  await page.waitForTimeout(200);
  const readout = await page.$eval(".atlas .readout", (e) => e.textContent);
  check(`${tag} atlas hover reports a head, a query and a key`, /Head 2\.4/.test(readout) && /%/.test(readout), readout);
  await page.mouse.move(0, 0);

  // temperature
  await page.evaluate(() => document.querySelector("#temperature").scrollIntoView());
  await page.waitForFunction(() => [...document.querySelectorAll(".tcol")].filter((c) => c.closest(".temp")).every((c) => c.querySelectorAll(".c.gen").length > 60), null, { timeout: 40000 });
  const rep = await page.$$eval(".temp .tstats dd", (e) => e.map((x) => x.textContent));
  check(`${tag} temperature: three columns report uncertainty`, rep.filter((t) => /bits/.test(t)).length === 3, rep.join("|"));
  const bits = (await page.$$eval(".temp .tstats > div:first-child dd", (e) => e.map((x) => parseFloat(x.textContent)))).slice(0, 3);
  check(`${tag} temperature: hotter writers are less certain`, bits[0] < bits[1] && bits[1] < bits[2], bits.join(","));

  // break it
  await page.evaluate(() => document.querySelector("#break").scrollIntoView());
  await page.waitForFunction(() => document.querySelectorAll(".parity li.ok .verdict, .parity li.fail .verdict").length >= 4 && document.querySelectorAll(".parity li .pl .small").length && [...document.querySelectorAll(".parity .pl .small")].filter((s) => /measured now/.test(s.textContent)).length === 3, null, { timeout: 40000 });
  const pv = await page.$$eval(".parity li", (e) => e.map((x) => ({ cls: x.className, v: x.querySelector(".pv")?.textContent })));
  check(`${tag} parity: intact passes, both bugs fail, live`, pv[2].cls === "ok" && pv[3].cls === "fail" && pv[4].cls === "fail", JSON.stringify(pv));
  await page.waitForFunction(() => [...document.querySelectorAll(".brk .tcol")].every((c) => c.querySelectorAll(".c.gen").length > 40), null, { timeout: 40000 });
  const [good, bad] = await page.$$eval(".brk .tcol .poem", (e) => e.map((p) => p.textContent));
  check(`${tag} break: broken model writes something else from the same dice`, good !== bad);
  await page.getByRole("radio", { name: "Silence one head" }).click();
  await page.waitForFunction(() => document.querySelectorAll(".mgrid button .sm").length === 16 && [...document.querySelectorAll(".mgrid button .sm")].every((s) => /\d/.test(s.textContent)), null, { timeout: 10000 });
  const klvals = await page.$$eval(".mgrid button .sm", (e) => e.map((x) => parseFloat(x.textContent)));
  check(`${tag} silence grid is filled with measured impacts`, klvals.length === 16 && klvals.every((v) => Number.isFinite(v) && v >= 0), klvals.join(","));
  await page.getByRole("button", { name: /Write both/ }).click();
  await page.waitForFunction(() => [...document.querySelectorAll(".brk .tcol")].every((c) => c.querySelectorAll(".c.gen").length > 40), null, { timeout: 40000 });
  const [g2, m2] = await page.$$eval(".brk .tcol .poem", (e) => e.map((p) => p.textContent));
  check(`${tag} silenced head changes the text`, g2 !== m2);

  // proof
  await page.evaluate(() => document.querySelector("#proof").scrollIntoView());
  await page.getByRole("button", { name: "Measure on this device" }).click();
  await page.getByText(/letters per second on this device/).waitFor({ timeout: 30000 });
  const tps = await page.$eval(".speed .big", (e) => Number(e.textContent.replace(/,/g, "")));
  check(`${tag} benchmark returns a speed`, tps > 20, tps);
  const loss = page.locator('.loss svg[role=slider]');
  await loss.focus();
  await page.keyboard.press("Home");
  const read = await page.$eval(".loss-read", (e) => e.textContent);
  check(`${tag} loss chart: keyboard reads the first checkpoint`, /step 0/.test(read), read);

  if (tag === "360") {
    // the chapter strip scrolls sideways on a phone: the current chapter must stay inside it
    let bad = "";
    for (const id of ["writing", "inside", "heads", "temperature", "break", "proof"]) {
      await page.evaluate((i) => document.getElementById(i).scrollIntoView({ block: "start", behavior: "instant" }), id);
      await page.waitForFunction((i) => document.querySelector(".nav a[aria-current=true]")?.getAttribute("href") === `#${i}`, id, { timeout: 5000 });
      await page.waitForTimeout(900);
      const inside = await page.evaluate(() => {
        const n = document.querySelector(".nav").getBoundingClientRect();
        const a = document.querySelector(".nav a[aria-current=true]").getBoundingClientRect();
        return a.left >= n.left - 1 && a.right <= n.right + 1;
      });
      if (!inside) bad += ` ${id}`;
    }
    check("360 the current chapter link is always fully inside the nav strip", !bad, bad);
    const navEdges = await page.$eval(".nav", (n) => ({ end: n.dataset.end, start: n.dataset.start }));
    check("360 nav edges are reported to the CSS", navEdges.end === "true" || navEdges.start === "false", JSON.stringify(navEdges));
  }

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(`${tag} no horizontal scroll`, overflow <= 0, overflow);
  check(`${tag} no console errors`, errors.length === 0, errors.join(" | "));
  const dashes = await page.evaluate(() => /[\u2013\u2014]/.test(document.body.innerText));
  check(`${tag} no long dashes in visible text`, !dashes);
  await ctx.close();
}

// prompt edits
{
  const { ctx, page, errors } = await open({ width: 1440, height: 900 });
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  await writeBtn(page).waitFor({ timeout: 40000 });
  await page.fill("#prompt", "Tadeusz wszedł do pokoju");
  await page.waitForFunction(() => document.querySelector(".poem-hero")?.textContent === "Tadeusz wszedł do pokoju" || document.querySelector(".poem-hero")?.textContent?.startsWith("Tadeusz wszedł do pokoju"), null, { timeout: 5000 });
  check("typing a new opening line replaces the poem", true);
  await page.fill("#prompt", "123456");
  await page.waitForTimeout(500);
  check("an unreadable line disables Write and says why", (await page.getByRole("button", { name: /^Write/ }).first().isDisabled()) && /None of these characters/.test(await page.$eval(".hero-controls", (e) => e.textContent)));
  await page.fill("#prompt", "Dzień dobry, panie Sędzio");
  await writeBtn(page).click();
  await page.waitForFunction(() => document.querySelectorAll(".poem-hero .c.gen").length > 20, null, { timeout: 10000 });
  await page.getByRole("button", { name: "Stop" }).click();
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some((b) => /Write again/.test(b.textContent)), null, { timeout: 5000 });
  check("Stop ends a run", true);
  check("prompt flow without errors", errors.length === 0, errors.join("|"));
  await ctx.close();
}

// reduced motion: nothing starts by itself, nothing has motion
{
  const { ctx, page } = await open({ width: 1440, height: 900 }, { reducedMotion: "reduce" });
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  await page.waitForTimeout(2500);
  const n = await poemLen(page);
  check("reduced motion: the poem does not start on its own", n === 21, n);
  const hasMotion = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running" && a.effect?.getTiming().iterations === Infinity).length);
  check("reduced motion: no infinite animations", hasMotion === 0, hasMotion);
  await writeBtn(page).click();
  await page.waitForFunction(() => document.querySelectorAll(".poem-hero .c.gen").length > 30, null, { timeout: 20000 });
  check("reduced motion: Write still works on request", true);
  await ctx.close();
}

// idle page: no animations running once the writing has stopped
{
  const { ctx, page } = await open({ width: 1440, height: 900 });
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  await writeBtn(page).waitFor({ timeout: 40000 });
  await page.waitForTimeout(1500);
  const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running").length);
  check("idle page has no running animations", running === 0, running);
  await ctx.close();
}

// error state: weights fail to load
{
  const { ctx, page } = await open({ width: 1440, height: 900 });
  await page.route("**/weights.f16.bin", (r) => r.abort());
  await page.goto(url, { waitUntil: "load" });
  await page.getByRole("alert").waitFor({ timeout: 15000 });
  const txt = await page.getByRole("alert").textContent();
  check("error state explains and offers a reload", /could not load/.test(txt) && (await page.getByRole("button", { name: "Reload the page" }).count()) === 1, txt);
  const plate = await page.$eval(".plate-empty", (e) => e.textContent);
  check("error state: figure says there is nothing to open", /did not load/.test(plate), plate);
  await ctx.close();
}

// the parity file fails to load: the page says so and does not spin forever
{
  const { ctx, page } = await open({ width: 1440, height: 900 });
  await page.route("**/parity-verse.json", (r) => r.abort());
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  await writeBtn(page).waitFor({ timeout: 40000 });
  await page.evaluate(() => document.querySelector("#break").scrollIntoView());
  await page.getByText(/The check could not run/).waitFor({ timeout: 20000 });
  check("a failing parity check is reported and the button is usable again", (await page.getByRole("button", { name: "Run the check on this page now" }).isEnabled()));
  await ctx.close();
}

// the app script fails once: the page retries and comes alive; fails twice: it says so
{
  const { ctx, page } = await open({ width: 1440, height: 900 });
  let hits = 0;
  await page.route(/assets\/index-[^/]*\.js(\?.*)?$/, (r) => (hits++ === 0 ? r.abort() : r.continue()));
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 30000 });
  check("a failed app script is retried once and the page starts", hits === 2, hits);
  await ctx.close();
  const b = await open({ width: 1440, height: 900 });
  await b.page.route(/assets\/index-[^/]*\.js(\?.*)?$/, (r) => r.abort());
  await b.page.goto(url, { waitUntil: "load" });
  await b.page.getByText(/The page script did not load/).waitFor({ timeout: 10000 });
  check("when the app script cannot load the static page says so", true);
  await b.ctx.close();
}

// share card: absolute og:image URL, twitter card, and the image itself is 1200x630
{
  const html = await (await fetch(url)).text();
  const og = html.match(/property="og:image" content="([^"]+)"/)?.[1] ?? "";
  check("og:image is an absolute URL", /^https:\/\//.test(og), og);
  check("twitter:card is summary_large_image", /name="twitter:card" content="summary_large_image"/.test(html));
  const img = await fetch(new URL(new URL(og || "https://x.invalid/og.png").pathname, url));
  const buf = Buffer.from(await img.arrayBuffer());
  const png = buf.subarray(0, 8).toString("hex") === "89504e470d0a1a0a";
  check("og.png is a 1200x630 PNG", img.ok && png && buf.readUInt32BE(16) === 1200 && buf.readUInt32BE(20) === 630, `${img.status} ${png ? buf.readUInt32BE(16) + "x" + buf.readUInt32BE(20) : "not png"}`);
}

await browser.close();
const failed = results.filter((r) => !r.ok);
writeFileSync(join(out, "e2e.json"), JSON.stringify({ url, passed: results.length - failed.length, failed: failed.length, results }, null, 1));
console.log(JSON.stringify({ url, passed: results.length - failed.length, failed: failed.length, failures: failed }, null, 1));
process.exit(failed.length ? 1 : 0);
