// Usage: npx vite-node scripts/head-stats.ts
// Measures what each of the 16 attention heads does on held-out text (the last tenth of the
// corpus, which the model never trained on) and writes src/generated/heads.json:
//   offsets: share of attention at distance 0, 1, 2, 3, 4-8, 9-32, 33+ (query positions 40 to 127)
//   spaces / punct / same: share of attention on whitespace, on punctuation, on the same letter as the query
//   dLoss: how much the validation loss (nats per character) grows when the head is zeroed everywhere
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GPT } from "../src/model/gpt";
import type { Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const here = dirname(fileURLToPath(import.meta.url));
const pub = resolve(here, "../public/model");
const manifest = JSON.parse(readFileSync(resolve(pub, "manifest.json"), "utf8")) as Manifest;
const b = readFileSync(resolve(pub, "weights.f16.bin"));
const gpt = new GPT(manifest.config, parseWeights(manifest, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer));
const { n_layer: L, n_head: H, ctx } = manifest.config;
const chars = manifest.chars;
const stoi = new Map(chars.map((c, i) => [c, i]));

const corpus = readFileSync(resolve(here, "../../data/corpus.txt"), "utf8");
const val = corpus.slice(-Math.floor(corpus.length / 10));
const ids = Array.from(val).map((c) => stoi.get(c) ?? -1);
if (ids.some((i) => i < 0)) throw new Error("corpus has characters outside the vocabulary");

const BUCKETS = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 8], [9, 32], [33, 127]] as const;
const WINDOWS = 48;
const FIRST_QUERY = 40;
const isSpace = (c: string) => c === " " || c === "\n";
const isPunct = (c: string) => ",.;:!?-()«»*".includes(c);

const offsets = Array.from({ length: L * H }, () => new Float64Array(BUCKETS.length));
const spaces = new Float64Array(L * H);
const punct = new Float64Array(L * H);
const same = new Float64Array(L * H);
let queries = 0;

function lossOver(mute: Uint8Array | null, collect: boolean): number {
  gpt.mute = mute;
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
      if (collect && p >= FIRST_QUERY) {
        queries++;
        const T = p + 1;
        for (let lh = 0; lh < L * H; lh++) {
          for (let t = 0; t < T; t++) {
            const a = r.attn[lh * T + t];
            const dist = p - t;
            BUCKETS.forEach(([lo, hi], k) => {
              if (dist >= lo && dist <= hi) offsets[lh][k] += a;
            });
            const ch = chars[ids[start + t]];
            if (isSpace(ch)) spaces[lh] += a;
            if (isPunct(ch)) punct[lh] += a;
            if (ids[start + t] === ids[start + p]) same[lh] += a;
          }
        }
      }
    }
  }
  gpt.mute = null;
  return total / count;
}

const base = lossOver(null, true);
console.log("validation loss on", WINDOWS, "windows:", base.toFixed(4));
const heads = [] as Record<string, unknown>[];
for (let lh = 0; lh < L * H; lh++) {
  const m = new Uint8Array(L * H);
  m[lh] = 1;
  const l = lossOver(m, false);
  const o = Array.from(offsets[lh], (v) => v / queries);
  heads.push({
    layer: Math.floor(lh / H) + 1,
    head: (lh % H) + 1,
    offsets: o.map((v) => Math.round(v * 1000) / 1000),
    spaces: Math.round((spaces[lh] / queries) * 1000) / 1000,
    punct: Math.round((punct[lh] / queries) * 1000) / 1000,
    same: Math.round((same[lh] / queries) * 1000) / 1000,
    dLoss: Math.round((l - base) * 10000) / 10000,
  });
  console.log(`L${Math.floor(lh / H) + 1}H${(lh % H) + 1}`, o.map((v) => (v * 100).toFixed(0).padStart(3)).join(""), "dLoss", (l - base).toFixed(4));
}
writeFileSync(
  resolve(here, "../src/generated/heads.json"),
  JSON.stringify({ windows: WINDOWS, queries, baseLoss: Math.round(base * 10000) / 10000, buckets: BUCKETS.map(([lo, hi]) => (lo === hi ? `${lo}` : `${lo}-${hi}`)), heads }),
);
console.log("written heads.json");
