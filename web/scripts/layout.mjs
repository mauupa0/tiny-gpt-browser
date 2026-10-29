// Usage: node scripts/layout.mjs <url> [runs]
// Measures the rhythm of the first screen at several widths: the gap between the end of the poem's
// text and the legend, between the legend and the candidate bars, the cumulative layout shift while
// the poem writes itself at reading pace, how many lines the finished poem takes (over `runs` fast
// runs), and whether the navigation fits or scrolls. Needs a global `playwright`.
import { createRequire } from "node:module";

const { chromium } = createRequire(process.env.GLOBAL_NODE_MODULES ? process.env.GLOBAL_NODE_MODULES + '/' : import.meta.url)("playwright");
const url = process.argv[2];
const runs = Number(process.argv[3] ?? 0);
const browser = await chromium.launch({ args: ["--enable-gpu", "--use-angle=d3d11"] });

const VIEWPORTS = [
  [1440, 900],
  [1280, 800],
  [1024, 768],
  [768, 1024],
  [390, 844],
  [360, 780],
  [320, 640],
];

const probe = () => {
  const r = (el) => {
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return { top: b.top + scrollY, bottom: b.bottom + scrollY, left: b.left, right: b.right, h: b.height };
  };
  const poem = document.querySelector(".poem-hero");
  const lines = [...poem.querySelectorAll(".l")];
  const lh = parseFloat(getComputedStyle(poem).lineHeight);
  const textBottom = Math.max(...lines.map((l) => r(l).bottom));
  const legend = r(document.querySelector(".legend"));
  const cands = r(document.querySelector(".cands"));
  const poemBox = r(poem);
  const nav = document.querySelector(".nav");
  const links = [...nav.querySelectorAll("a")].map((a) => {
    const b = a.getBoundingClientRect();
    const n = nav.getBoundingClientRect();
    return { t: a.textContent, cut: b.left < n.left - 0.5 || b.right > n.right + 0.5, active: a.getAttribute("aria-current") === "true" };
  });
  return {
    poemBox: Math.round(poemBox.h),
    poemText: Math.round(textBottom - poemBox.top),
    lines: Math.round((textBottom - poemBox.top) / lh),
    gapPoemLegend: Math.round(legend.top - textBottom),
    gapLegendCands: Math.round(cands.top - legend.bottom),
    navScroll: nav.scrollWidth > nav.clientWidth + 1,
    navCut: links.filter((l) => l.cut).map((l) => l.t),
    navActiveCut: links.some((l) => l.active && l.cut),
    docOverflow: document.documentElement.scrollWidth > innerWidth,
  };
};

for (const [w, h] of VIEWPORTS) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    // Position of everything under the poem on every frame: any change is a visible jump.
    window.__jumps = [];
    const names = [".legend", ".cands", ".cand-note", ".hero-actions", ".hero-controls"];
    const last = {};
    const tick = () => {
      for (const n of names) {
        const el = document.querySelector(n);
        if (!el) continue;
        const y = el.getBoundingClientRect().top + scrollY;
        if (last[n] !== undefined && Math.abs(y - last[n]) > 0.5) window.__jumps.push({ n, t: Math.round(performance.now()), d: Math.round(y - last[n]) });
        last[n] = y;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    window.__ls = [];
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) {
        if (!e.hadRecentInput) window.__ls.push({ v: e.value, t: Math.round(e.startTime), s: (e.sources || []).map((x) => (x.node && x.node.className) || (x.node && x.node.nodeName) || "?").join("|") });
      }
    }).observe({ type: "layout-shift", buffered: true });
  });
  await page.goto(url, { waitUntil: "load" });
  await page.getByText(/MB of weights loaded/).waitFor({ timeout: 60000 });
  const tStart = await page.evaluate(() => performance.now());
  await page.getByRole("button", { name: "Write again" }).waitFor({ timeout: 90000 });
  await page.waitForFunction(() => !document.querySelector(".cands.stale") && document.querySelector(".plate")?.getAttribute("aria-busy") === "false", null, { timeout: 15000 });
  await page.waitForTimeout(400);
  const m = await page.evaluate(probe);
  const ls = await page.evaluate(() => window.__ls);
  const cls = ls.reduce((a, e) => a + e.v, 0);
  const clsWrite = ls.filter((e) => e.t > 0).reduce((a, e) => a + e.v, 0);
  const big = ls.filter((e) => e.v > 0.0005).map((e) => `${e.t}ms ${e.v.toFixed(4)} [${e.s}]`);
  console.log(`\n== ${w}x${h}`);
  console.log(JSON.stringify(m));
  console.log(`CLS total ${cls.toFixed(4)} (n=${ls.length}, weights ready at ${Math.round(tStart)}ms)${big.length ? "\n  shifts: " + big.slice(0, 6).join("\n          ") : ""}`);
  void clsWrite;
  const jumps = (await page.evaluate(() => window.__jumps)).filter((j) => j.t > 700);
  console.log(`jumps of legend/bars/note/actions/controls after load: ${jumps.length}${jumps.length ? "  " + jumps.slice(0, 8).map((j) => `${j.n}@${j.t}ms ${j.d > 0 ? "+" : ""}${j.d}px`).join(", ") : ""}`);

  if (runs > 0) {
    await page.locator(".switch input").check({ force: true });
    const counts = [];
    const ends = [];
    for (let i = 0; i < runs; i++) {
      await page.getByRole("button", { name: /^(Write again)$/ }).click();
      await page.getByRole("button", { name: "Write again" }).waitFor({ timeout: 30000 });
      await page.waitForTimeout(150);
      const r = await page.evaluate(probe);
      counts.push(r.lines);
      ends.push(r.gapPoemLegend);
    }
    const hist = {};
    counts.forEach((c) => (hist[c] = (hist[c] || 0) + 1));
    console.log(`lines over ${runs} runs: ${JSON.stringify(hist)} max ${Math.max(...counts)}; gap poem to legend px: min ${Math.min(...ends)} median ${[...ends].sort((a, b) => a - b)[Math.floor(ends.length / 2)]} max ${Math.max(...ends)}`);
  }
  await ctx.close();
}
await browser.close();
