import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import headsJson from "../src/generated/heads.json";
import stats from "../src/generated/stats.json";
import { pct, sci, signed } from "../src/lib/format";
import { classifyHead, importance, type HeadStat, type HeadsData } from "../src/lib/heads";
import { logPos } from "../src/lib/logaxis";
import { activeNeurons, ordinalBack, peakOf, pushTotal, rankOf } from "../src/lib/narrate";
import {
  COOL,
  GROUND,
  GROUND_DEEP,
  INK,
  INK2,
  INK3,
  MARKER,
  PANEL,
  RAMP,
  contrast,
  divergingLut,
  lchInGamut,
  lchToCss,
  lchToRgb,
  rampLut,
  type Lch,
} from "../src/lib/palette";
import { paramLedger } from "../src/lib/params";
import { decode, encodeText, longShotIndex, repeatShare, splitLines } from "../src/lib/text";
import { GPT } from "../src/model/gpt";
import { Inspector } from "../src/model/inspect";
import type { Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const here = (p: string) => resolve(__dirname, p);
const heads = headsJson as HeadsData;

describe("palette", () => {
  const css = readFileSync(here("../src/styles/tokens.css"), "utf8");
  const tokens: Record<string, Lch> = {
    "--ground": GROUND,
    "--ground-deep": GROUND_DEEP,
    "--panel": PANEL,
    "--ink": INK,
    "--ink-2": INK2,
    "--ink-3": INK3,
    "--marker": MARKER,
    "--cool": COOL,
    ...Object.fromEntries(RAMP.map((c, i) => [`--ramp-${i}`, c])),
  };
  const declared = (name: string) => new RegExp(`${name}:\\s*(oklch\\([^)]*\\));`).exec(css)?.[1];

  it("CSS tokens and canvas colours are the same values", () => {
    for (const [name, c] of Object.entries(tokens)) expect(declared(name), name).toBe(lchToCss(c));
  });

  it("negative control: a drifted token is detected", () => {
    expect(declared("--marker")).not.toBe(lchToCss([0.78, 0.15, 55]));
  });

  it("the ramp rises in lightness and every colour fits sRGB", () => {
    RAMP.forEach((c, i) => {
      expect(lchInGamut(c)).toBe(true);
      if (i) expect(c[0]).toBeGreaterThan(RAMP[i - 1][0]);
    });
    const lut = rampLut(256);
    expect(lut).toHaveLength(256);
    expect(lut[0]).not.toBe(lut[255]);
  });

  it("the diverging scale is neutral in the middle and different at the poles", () => {
    const d = divergingLut(257);
    const [r, g, b] = lchToRgb(PANEL);
    expect(d[128]).toBe(((255 << 24) | (b << 16) | (g << 8) | r) >>> 0);
    expect(d[0]).not.toBe(d[256]);
  });

  it("text and signals are readable on the surfaces they sit on", () => {
    for (const bg of [GROUND, GROUND_DEEP, PANEL]) {
      expect(contrast(INK, bg)).toBeGreaterThan(7);
      expect(contrast(INK2, bg)).toBeGreaterThan(4.5);
      expect(contrast(INK3, bg)).toBeGreaterThan(4.5);
      expect(contrast(MARKER, bg)).toBeGreaterThan(4.5);
    }
    expect(contrast(GROUND_DEEP, MARKER)).toBeGreaterThan(4.5);
    // control: a dim grey really would fail
    expect(contrast([0.45, 0.02, 250], GROUND)).toBeLessThan(4.5);
  });
});

describe("head roles", () => {
  const at = (l: number, h: number) => heads.heads.find((x) => x.layer === l && x.head === h)!;
  const flat = (o: number[], over: Partial<HeadStat> = {}): HeadStat => ({ layer: 1, head: 1, offsets: o, spaces: 0.15, punct: 0.05, same: 0.05, dLoss: 0.01, ...over });

  it("reads the measured heads of this model", () => {
    expect(classifyHead(at(1, 2)).key).toBe("previous");
    expect(classifyHead(at(1, 4)).key).toBe("previous");
    expect(classifyHead(at(2, 4)).key).toBe("boundaries");
    expect(classifyHead(at(2, 2)).key).toBe("far");
    expect(classifyHead(at(1, 1)).key).toBe("two");
  });

  it("the most important head is the word-boundary head, by a wide margin", () => {
    const { top, ratio } = importance(heads);
    expect([top.layer, top.head]).toEqual([2, 4]);
    expect(ratio).toBeGreaterThan(5);
  });

  it("negative control: spread-out attention is not given a sharp role", () => {
    expect(classifyHead(flat([0.1, 0.15, 0.15, 0.1, 0.2, 0.2, 0.1])).key).toBe("mixed");
    expect(classifyHead(flat([0.05, 0.6, 0.1, 0.05, 0.1, 0.05, 0.05])).key).not.toBe("previous");
    expect(classifyHead(flat([0.02, 0.9, 0.02, 0.02, 0.02, 0.01, 0.01])).key).toBe("previous");
  });

  it("every head has seven reach buckets that sum to about one", () => {
    for (const h of heads.heads) expect(h.offsets.reduce((a, b) => a + b, 0)).toBeGreaterThan(0.98);
    expect(heads.heads).toHaveLength(16);
  });
});

describe("text helpers", () => {
  const chars = ["\n", " ", "a", "b", "ą"];
  it("encodes, decodes and counts what the alphabet lacks", () => {
    const { ids, dropped } = encodeText(chars, "ab ą\r\nxz");
    expect(decode(chars, ids)).toBe("ab ą\n");
    expect(dropped).toBe(2);
    expect(encodeText(chars, "").ids).toEqual([]);
  });
  it("repeatShare counts earlier 4-grams", () => {
    expect(repeatShare([1, 2, 3, 4, 5, 6, 7, 8])).toBe(0);
    expect(repeatShare([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])).toBeCloseTo(5 / 9, 10);
    expect(repeatShare([1, 2])).toBe(0);
  });
  it("splitLines keeps every character index and ends a line after each break", () => {
    const lines = splitLines([2, 3, 0, 3, 2], chars);
    expect(lines.map((l) => l.map((c) => c.index))).toEqual([
      [0, 1, 2],
      [3, 4],
    ]);
  });
});

describe("where the figure rests after a poem", () => {
  const chars = ["\n", " ", "a", "b"];
  const ids = [2, 3, 2, 3, 0, 2, 3, 2];
  it("sits just before the longest shot of the recent letters, skipping line breaks", () => {
    const ps = [1, 1, 0.9, 0.05, 0.001, 0.4, 0.2, 0.6];
    expect(longShotIndex(ids, ps, chars, 2)).toBe(2);
  });
  it("negative control: safe picks give no answer, and the prompt is never chosen", () => {
    expect(longShotIndex(ids, [1, 1, 0.9, 0.8, 0.9, 0.7, 0.6, 0.9], chars, 2)).toBe(-1);
    expect(longShotIndex(ids, [0.001, 0.001, 0.9, 0.8, 0.9, 0.7, 0.6, 0.9], chars, 2)).toBe(-1);
  });
});

describe("numbers", () => {
  it("parameter ledger equals the count PyTorch reported, and a wrong width does not", () => {
    expect(paramLedger({ vocab: 71, ctx: 128, d: 192, layers: 4 }).total).toBe(stats.params);
    expect(paramLedger({ vocab: 71, ctx: 128, d: 193, layers: 4 }).total).not.toBe(stats.params);
  });
  it("log axis maps decades evenly and clamps", () => {
    expect(logPos(1e-7)).toBeCloseTo(0, 10);
    expect(logPos(10)).toBeCloseTo(1, 10);
    expect(logPos(1e-3)).toBeCloseTo(0.5, 10);
    expect(logPos(0)).toBe(0);
    expect(logPos(1e9)).toBe(1);
    expect(logPos(1e-5)).toBeLessThan(logPos(1e-4));
  });
  it("formatters", () => {
    expect(pct(0.2567, 1)).toBe("25.7%");
    expect(pct(0.00004, 1)).toBe("<0.1%");
    expect(signed(-0.004, 2)).toBe("+0.00");
    expect(signed(-1.236, 2)).toBe("-1.24");
    expect(sci(6.7e-6)).toBe("6.7e-6");
  });
});

describe("captions are computed from the real inspection", () => {
  const manifest = JSON.parse(readFileSync(here("../public/model/manifest.json"), "utf8")) as Manifest;
  const bin = readFileSync(here("../public/model/weights.f16.bin"));
  const gpt = new GPT(manifest.config, parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer));
  const ids = Array.from("Litwo! Ojczyzno moja! ty jesteś jak zdrowie").map((c) => manifest.chars.indexOf(c));
  const insp = new Inspector(gpt).inspect(ids);

  it("peak, rank, total and neuron count agree with the raw arrays", () => {
    const pk = peakOf(insp, 0, 1);
    const row = insp.last.subarray(1 * insp.n, 2 * insp.n);
    expect(row[pk.pos]).toBe(Math.max(...Array.from(row)));
    expect(pk.back).toBe(insp.n - 1 - pk.pos);
    expect(rankOf(insp.probs, insp.top)).toBe(1);
    expect(rankOf(insp.probs, (insp.top + 1) % 71)).toBeGreaterThan(1);
    expect(Math.abs(pushTotal(insp) - insp.logit)).toBeLessThan(2e-4);
    const n = activeNeurons(insp, 0, 0.1);
    expect(n).toBe(Array.from(insp.mlp.subarray(0, 768)).filter((v) => v > 0.1).length);
    expect(n).toBeGreaterThan(0);
    expect(n).toBeLessThan(768);
  });
  it("wording of distances", () => {
    expect(ordinalBack(0)).toBe("on itself");
    expect(ordinalBack(1)).toBe("1 letter back");
    expect(ordinalBack(5)).toBe("5 letters back");
  });
});

describe("fault numbers shown on the page", () => {
  const manifest = JSON.parse(readFileSync(here("../public/model/manifest.json"), "utf8")) as Manifest;
  const bin = readFileSync(here("../public/model/weights.f16.bin"));
  const weights = parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer);
  const corpus = readFileSync(here("../../data/corpus.txt"), "utf8");
  const val = Array.from(corpus.slice(-Math.floor(corpus.length / 10))).map((c) => manifest.chars.indexOf(c));

  function loss(fault: "none" | "scale" | "mask", windows: number): number {
    const gpt = new GPT(manifest.config, weights, fault);
    let total = 0;
    let n = 0;
    for (let w = 0; w < windows; w++) {
      const start = w * 129;
      const cache = gpt.createCache();
      for (let p = 0; p < 128; p++) {
        const r = gpt.step(val[start + p], cache);
        const max = Math.max(...Array.from(r.logits));
        const lse = Math.log(Array.from(r.logits).reduce((a, v) => a + Math.exp(v - max), 0)) + max;
        total += lse - r.logits[val[start + p + 1]];
        n++;
      }
    }
    return total / n;
  }

  it("a quick re-measurement agrees with faults.json in direction and size", async () => {
    const faults = (await import("../src/generated/faults.json")).default;
    const none = loss("none", 3);
    const scale = loss("scale", 3);
    const mask = loss("mask", 3);
    expect(scale - none).toBeGreaterThan(0.1);
    expect(faults.scale - faults.none).toBeGreaterThan(0.1);
    expect(Math.abs(scale - none - (faults.scale - faults.none))).toBeLessThan(0.15);
    // the quiet bug: the mask leak barely moves the loss, so only the parity check can see it
    expect(Math.abs(mask - none)).toBeLessThan(0.05);
    expect(Math.abs(faults.mask - faults.none)).toBeLessThan(0.05);
    expect(faults.none).toBeCloseTo(heads.baseLoss, 4);
  });
});
