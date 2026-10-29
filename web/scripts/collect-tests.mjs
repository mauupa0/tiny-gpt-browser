// Usage: node scripts/collect-tests.mjs
// Runs the test suite and records how many tests passed, so the page states a measured number.
import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const raw = execSync("npx vitest run --reporter=json", { cwd: resolve(here, ".."), encoding: "utf8", maxBuffer: 1 << 26, stdio: ["ignore", "pipe", "ignore"] });
const start = raw.indexOf("{");
const r = JSON.parse(raw.slice(start));
const files = r.testResults.map((f) => ({ file: f.name.split(/[\/]/).pop(), tests: f.assertionResults.length, passed: f.assertionResults.filter((a) => a.status === "passed").length }));
const out = { total: r.numTotalTests, passed: r.numPassedTests, failed: r.numFailedTests, files };
const path = resolve(here, "../src/generated/tests.json");
mkdirSync(dirname(path), { recursive: true });
writeFileSync(path, JSON.stringify(out));
console.log(JSON.stringify(out));
if (r.numFailedTests > 0) process.exit(1);
