// Usage: npx vite-node scripts/fault-loss.ts
// Validation loss (nats per letter) of the intact model and of each deliberately broken variant,
// on the same held-out windows as head-stats.ts. Writes src/generated/faults.json.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GPT } from "../src/model/gpt";
import type { Fault, Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const here = dirname(fileURLToPath(import.meta.url));
const pub = resolve(here, "../public/model");
const manifest = JSON.parse(readFileSync(resolve(pub, "manifest.json"), "utf8")) as Manifest;
const b = readFileSync(resolve(pub, "weights.f16.bin"));
const weights = parseWeights(manifest, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
const { ctx } = manifest.config;
const stoi = new Map(manifest.chars.map((c, i) => [c, i]));
const corpus = readFileSync(resolve(here, "../../data/corpus.txt"), "utf8");
const ids = Array.from(corpus.slice(-Math.floor(corpus.length / 10))).map((c) => stoi.get(c) ?? -1);
if (ids.some((i) => i < 0)) throw new Error("corpus has characters outside the vocabulary");

const WINDOWS = 48;

function loss(fault: Fault): number {
  const gpt = new GPT(manifest.config, weights, fault);
  let total = 0;
  let count = 0;
  for (let w = 0; w < WINDOWS; w++) {
    const start = w * (ctx + 1);
    const cache = gpt.createCache();
    for (let p = 0; p < ctx; p++) {
      const r = gpt.step(ids[start + p], cache);
      let max = -Infinity;
      for (const v of r.logits) if (v > max) max = v;
      let sum = 0;
      for (const v of r.logits) sum += Math.exp(v - max);
      total += Math.log(sum) + max - r.logits[ids[start + p + 1]];
      count++;
    }
  }
  return total / count;
}

const out = { windows: WINDOWS, none: loss("none"), scale: loss("scale"), mask: loss("mask") };
const rounded = Object.fromEntries(Object.entries(out).map(([k, v]) => [k, typeof v === "number" && k !== "windows" ? Math.round(v * 1e4) / 1e4 : v]));
writeFileSync(resolve(here, "../src/generated/faults.json"), JSON.stringify(rounded));
console.log(rounded);
