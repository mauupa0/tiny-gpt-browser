import { describe, expect, it } from "vitest";
import { Engine } from "../src/model/engine";
import { GPT } from "../src/model/gpt";
import { distribution, mulberry32, sampleFrom, topN } from "../src/model/sample";
import type { ModelConfig, Weights } from "../src/model/types";
import { halfToFloat } from "../src/model/weights";

describe("halfToFloat", () => {
  it("decodes known fp16 bit patterns", () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 4);
    expect(halfToFloat(0x0001)).toBe(2 ** -24);
    expect(halfToFloat(0x7bff)).toBe(65504);
  });
});

describe("sampling", () => {
  const logits = new Float32Array([3, 1, 0.5, 2, -1]);
  it("top-k keeps exactly k tokens and sums to 1", () => {
    const p = distribution(logits, 1, 2);
    expect(Array.from(p).filter((x) => x > 0)).toHaveLength(2);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[0]).toBeGreaterThan(p[3]);
  });
  it("temperature 0 is greedy and low temperature sharpens", () => {
    expect(Array.from(distribution(logits, 0, 5))).toEqual([1, 0, 0, 0, 0]);
    expect(distribution(logits, 0.3, 5)[0]).toBeGreaterThan(distribution(logits, 1.5, 5)[0]);
  });
  it("seeded sampling is deterministic and follows the distribution", () => {
    const p = distribution(logits, 1, 5);
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 20; i++) expect(sampleFrom(p, a)).toBe(sampleFrom(p, b));
    const r = mulberry32(7);
    const counts = new Array(5).fill(0);
    for (let i = 0; i < 20000; i++) counts[sampleFrom(p, r)]++;
    expect(counts[0] / 20000).toBeCloseTo(p[0], 1);
  });
  it("topN orders by probability and drops zeros", () => {
    const top = topN(distribution(logits, 1, 3), 10);
    expect(top.map((t) => t.id)).toEqual([0, 3, 1]);
  });
});

function tinyModel(ctx: number): GPT {
  const cfg: ModelConfig = { vocab: 5, ctx, d_model: 8, n_layer: 1, n_head: 2 };
  const rnd = mulberry32(3);
  const arr = (n: number, s = 0.3) => Float32Array.from({ length: n }, () => (rnd() - 0.5) * s);
  const ones = (n: number) => new Float32Array(n).fill(1);
  const d = 8;
  const w: Weights = {
    tok: arr(5 * d),
    pos: arr(ctx * d),
    blocks: [
      {
        ln1w: ones(d), ln1b: new Float32Array(d), qkvw: arr(3 * d * d), qkvb: new Float32Array(3 * d),
        projw: arr(d * d), projb: new Float32Array(d), ln2w: ones(d), ln2b: new Float32Array(d),
        fcw: arr(4 * d * d), fcb: new Float32Array(4 * d), outw: arr(4 * d * d), outb: new Float32Array(d),
      },
    ],
    lnfw: ones(d),
    lnfb: new Float32Array(d),
  };
  return new GPT(cfg, w);
}

describe("engine", () => {
  it("prefill equals stepping one token at a time (KV cache is consistent)", () => {
    const ids = [1, 2, 3, 4, 0, 2];
    const a = new Engine(tinyModel(16)).start(ids);
    const e = new Engine(tinyModel(16));
    let last = e.start(ids.slice(0, 1));
    for (const id of ids.slice(1)) last = e.push(id);
    for (let i = 0; i < a.logits.length; i++) expect(last.logits[i]).toBeCloseTo(a.logits[i], 6);
  });
  it("keeps generating past the context window by rebuilding it", () => {
    const e = new Engine(tinyModel(8));
    e.start([1, 2, 3]);
    let resets = 0;
    for (let i = 0; i < 40; i++) {
      const r = e.push(i % 5);
      if (r.reset) resets++;
      expect(r.window.length).toBeLessThanOrEqual(8);
      expect(r.logits.every(Number.isFinite)).toBe(true);
    }
    expect(resets).toBeGreaterThan(0);
  });
  it("attention rows are probability distributions over the visible past", () => {
    const r = new Engine(tinyModel(16)).start([1, 2, 3, 4]);
    for (const row of r.rows) {
      const heads = row.attn.length / (row.pos + 1);
      for (let h = 0; h < heads; h++) {
        let s = 0;
        for (let t = 0; t <= row.pos; t++) s += row.attn[h * (row.pos + 1) + t];
        expect(s).toBeCloseTo(1, 5);
      }
    }
  });
});
