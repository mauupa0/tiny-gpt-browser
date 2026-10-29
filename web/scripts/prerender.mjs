// Usage: node scripts/prerender.mjs   (run by `npm run build` after `vite build`)
// Renders the app's first state to HTML with Vite's SSR loader and writes it into dist/index.html.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const vite = await createServer({ root, appType: "custom", logLevel: "error", server: { middlewareMode: true } });
try {
  const { renderShell } = await vite.ssrLoadModule("/src/shell.tsx");
  const html = renderShell();
  const file = resolve(root, "dist/index.html");
  const page = readFileSync(file, "utf8");
  if (!page.includes('<div id="root"></div>')) throw new Error("dist/index.html has no empty #root to fill");
  writeFileSync(file, page.replace('<div id="root"></div>', () => `<div id="root">${html}</div>`));
  console.log(`prerendered ${html.length} bytes of HTML into dist/index.html`);
} finally {
  await vite.close();
}
