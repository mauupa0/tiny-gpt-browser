import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { GPT } from "../src/model/gpt";
import { Inspector, type Inspection } from "../src/model/inspect";
import { PARITY_LIMIT, parityCheck, type ParityRef } from "../src/model/parity";
import { argmax, entropyBits, klNats, packAttn, softmax, unpackAttn } from "../src/model/stats";
import type { Fault, Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const here = (p: string) => resolve(__dirname, p);

function load(manifestPath: string, binPath: string, fault: Fault = "none") {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
  const b = readFileSync(binPath);
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  return { manifest, gpt: new GPT(manifest.config, parseWeights(manifest, buf), fault) };
}

const shipped = load(here("../public/model/manifest.json"), here("../public/model/weights.f16.bin"));
const chars = shipped.manifest.chars;
const enc = (s: string) => Array.from(s).map((c) => chars.indexOf(c));
const VERSE = enc("Gdzie, przestępując miedzę, nie poznasz, że cudza!");

function flat(i: Inspection): number[][] {
  return [
    Array.from(i.attn),
    Array.from(i.last),
    Array.from(i.probs),
    ...i.lens.map((l) => Array.from(l)),
    Array.from(i.norms),
    Array.from(i.headNorm),
    Array.from(i.mlp),
    Array.from(i.embed),
    [i.logit, i.attribution.embed, i.attribution.bias, i.attribution.constant, ...i.attribution.heads, ...i.attribution.mlp],
  ];
}

function maxDiff(a: Inspection, b: Inspection): number {
  const fa = flat(a);
  const fb = flat(b);
  let m = 0;
  fa.forEach((x, k) => x.forEach((v, j) => (m = Math.max(m, Math.abs(v - fb[k][j])))));
  return m;
}

describe("quantised attention", () => {
  it("round-trips within the stated bound", () => {
    let worst = 0;
    for (let i = 0; i <= 20000; i++) {
      const p = i / 20000;
      const err = Math.abs(unpackAttn(packAttn(p)) - p);
      expect(err).toBeLessThanOrEqual(Math.sqrt(p) / 255 + 1 / 255 ** 2 + 1e-12);
      worst = Math.max(worst, err);
    }
    expect(unpackAttn(packAttn(0))).toBe(0);
    expect(unpackAttn(packAttn(1))).toBe(1);
    expect(worst).toBeGreaterThan(0);
  });
});

describe("trace", () => {
  it("stepTraced returns the same logits and attention as step", () => {
    const a = shipped.gpt.createCache();
    const b = shipped.gpt.createCache();
    VERSE.forEach((id) => {
      const r1 = shipped.gpt.step(id, a);
      const r2 = shipped.gpt.stepTraced(id, b);
      expect(Array.from(r2.logits)).toEqual(Array.from(r1.logits));
      expect(Array.from(r2.attn)).toEqual(Array.from(r1.attn));
    });
  });

  it("logit lens at the last stage is the real prediction", () => {
    const c = shipped.gpt.createCache();
    let last = shipped.gpt.stepTraced(VERSE[0], c);
    for (const id of VERSE.slice(1)) last = shipped.gpt.stepTraced(id, c);
    const stages = last.trace.lens.length;
    expect(stages).toBe(1 + 2 * shipped.manifest.config.n_layer);
    const real = Array.from(last.logits);
    last.trace.lens[stages - 1].forEach((v, i) => expect(Math.abs(v - real[i])).toBeLessThan(1e-4));
    // negative control: the first stage (embedding only) is a different prediction
    const early = Math.max(...Array.from(last.trace.lens[0]).map((v, i) => Math.abs(v - real[i])));
    expect(early).toBeGreaterThan(0.1);
  });

  it("the residual stream is the sum of what each part wrote", () => {
    const c = shipped.gpt.createCache();
    let tr = shipped.gpt.stepTraced(VERSE[0], c).trace;
    for (const id of VERSE.slice(1, 30)) tr = shipped.gpt.stepTraced(id, c).trace;
    const d = shipped.manifest.config.d_model;
    let prev = tr.embed;
    let worst = 0;
    for (const b of tr.blocks) {
      for (let i = 0; i < d; i++) {
        let mid = prev[i] + b.attnBias[i];
        for (const h of b.headOut) mid += h[i];
        worst = Math.max(worst, Math.abs(mid - b.xMid[i]), Math.abs(b.xMid[i] + b.mlpOut[i] - b.xOut[i]));
      }
      prev = b.xOut;
    }
    expect(worst).toBeLessThan(1e-4);
    // negative control: forgetting one head breaks the sum by more than float noise
    const b0 = tr.blocks[0];
    let off = 0;
    for (let i = 0; i < d; i++) {
      let mid = tr.embed[i] + b0.attnBias[i];
      for (const h of b0.headOut.slice(1)) mid += h[i];
      off = Math.max(off, Math.abs(mid - b0.xMid[i]));
    }
    expect(off).toBeGreaterThan(1e-2);
  });
});

describe("attribution", () => {
  const ins = new Inspector(shipped.gpt).inspect(VERSE.slice(0, 40));

  it("pushes sum exactly to the logit, for the favourite and for any other character", () => {
    for (const target of [ins.top, 0, 17, 40]) {
      const r = new Inspector(shipped.gpt).inspect(VERSE.slice(0, 40), { target });
      const a = r.attribution;
      const sum = a.embed + a.bias + a.constant + a.heads.reduce((x, y) => x + y, 0) + a.mlp.reduce((x, y) => x + y, 0);
      expect(Math.abs(sum - r.logit)).toBeLessThan(2e-4);
    }
  });

  it("negative control: leaving out the largest push breaks the sum", () => {
    const a = ins.attribution;
    const pushes = [a.embed, ...Array.from(a.heads), ...Array.from(a.mlp)];
    const biggest = pushes.reduce((m, v) => (Math.abs(v) > Math.abs(m) ? v : m), 0);
    const sum = a.embed + a.bias + a.constant + a.heads.reduce((x, y) => x + y, 0) + a.mlp.reduce((x, y) => x + y, 0);
    expect(Math.abs(Math.abs(biggest))).toBeGreaterThan(0.05);
    expect(Math.abs(sum - biggest - ins.logit)).toBeGreaterThan(0.05);
  });
});

describe("inspector", () => {
  it("attention rows are distributions and the byte coding survives the trip", () => {
    const ins = new Inspector(shipped.gpt).inspect(VERSE.slice(0, 30));
    const { layers, heads, n } = ins;
    for (let lh = 0; lh < layers * heads; lh++) {
      for (let p = 0; p < n; p++) {
        let s = 0;
        for (let t = 0; t <= p; t++) s += unpackAttn(ins.attn[(lh * n + p) * n + t]);
        expect(Math.abs(s - 1)).toBeLessThan(0.05);
        for (let t = p + 1; t < n; t++) expect(ins.attn[(lh * n + p) * n + t]).toBe(0);
      }
    }
  });

  it("scrubbing around with a reused cache gives exactly what a fresh inspector gives", () => {
    const reused = new Inspector(shipped.gpt);
    const seq = [
      VERSE.slice(0, 20),
      VERSE.slice(0, 21),
      VERSE.slice(0, 22),
      VERSE.slice(0, 12),
      VERSE.slice(0, 40),
      [...VERSE.slice(0, 15), 33, 34, 35],
      VERSE.slice(0, 15),
      VERSE.slice(0, 50),
    ];
    for (const ids of seq) {
      const a = reused.inspect(ids, { ablate: true });
      const b = new Inspector(shipped.gpt).inspect(ids, { ablate: true });
      expect(maxDiff(a, b)).toBe(0);
      expect(Array.from(a.ablation!.kl)).toEqual(Array.from(b.ablation!.kl));
    }
  });

  it("negative control: a cache that lies about its contents is caught by the same comparison", () => {
    const liar = new Inspector(shipped.gpt);
    liar.inspect(VERSE.slice(0, 30));
    const edited = [...VERSE.slice(0, 10), 40, ...VERSE.slice(11, 30)];
    (liar as unknown as { ids: number[] }).ids = edited.slice();
    const wrong = liar.inspect(edited);
    const right = new Inspector(shipped.gpt).inspect(edited);
    expect(maxDiff(wrong, right)).toBeGreaterThan(1e-3);
  });

  it("keeps only the newest ctx characters", () => {
    const long = Array.from({ length: 200 }, (_, i) => VERSE[i % VERSE.length]);
    const ins = new Inspector(shipped.gpt).inspect(long);
    expect(ins.n).toBe(shipped.manifest.config.ctx);
    expect(ins.ids).toEqual(long.slice(-shipped.manifest.config.ctx));
  });

  it("rejects an empty context", () => {
    expect(() => new Inspector(shipped.gpt).inspect([])).toThrow();
  });
});

describe("head ablation", () => {
  const ids = VERSE.slice(0, 45);
  const ins = new Inspector(shipped.gpt).inspect(ids, { ablate: true });
  const ab = ins.ablation!;

  it("zeroing a head moves the prediction, and the distribution stays valid", () => {
    expect(ab.kl.every((v) => Number.isFinite(v) && v >= 0)).toBe(true);
    expect(Math.max(...Array.from(ab.kl))).toBeGreaterThan(1e-3);
  });

  it("negative control: an empty mask changes nothing", () => {
    const g = new GPT(shipped.manifest.config, shipped.gpt.w);
    g.mute = new Uint8Array(16);
    const cache = g.createCache();
    let r = g.step(ids[0], cache);
    for (const id of ids.slice(1)) r = g.step(id, cache);
    const base = shipped.gpt.createCache();
    let r0 = shipped.gpt.step(ids[0], base);
    for (const id of ids.slice(1)) r0 = shipped.gpt.step(id, base);
    expect(klNats(softmax(r0.logits), softmax(r.logits))).toBe(0);
  });

  it("leaves the cache and the model as they were", () => {
    const insp = new Inspector(shipped.gpt);
    const first = insp.inspect(ids, { ablate: true });
    const again = insp.inspect(ids, { ablate: false });
    expect(maxDiff(first, again)).toBe(0);
    expect(shipped.gpt.mute).toBeNull();
  });

  it("agrees with a hand-made ablation of the strongest head", () => {
    const worst = argmax(ab.kl);
    const mask = new Uint8Array(16);
    mask[worst] = 1;
    const g = new GPT(shipped.manifest.config, shipped.gpt.w);
    g.mute = mask;
    const cache = g.createCache();
    let r = g.step(ids[0], cache);
    for (const id of ids.slice(1)) r = g.step(id, cache);
    // The page zeroes the head at the last position only; here it is zeroed everywhere, so the
    // two numbers are related but not equal. Both must be clearly above the untouched baseline.
    const everywhere = klNats(ins.probs, softmax(r.logits));
    expect(everywhere).toBeGreaterThan(1e-3);
    expect(ab.kl[worst]).toBeGreaterThan(1e-3);
  });
});

describe("live parity helper", () => {
  const ref = JSON.parse(readFileSync(here("../public/model/parity-verse.json"), "utf8")) as ParityRef;

  it("the shipped model sits far below the limit and each fault far above it", () => {
    const ok = parityCheck(shipped.gpt, ref);
    expect(ok.perPos).toHaveLength(ref.ids.length);
    expect(ok.max).toBeLessThan(PARITY_LIMIT / 50);
    for (const fault of ["scale", "mask"] as const) {
      const bad = parityCheck(load(here("../public/model/manifest.json"), here("../public/model/weights.f16.bin"), fault).gpt, ref);
      expect(bad.max).toBeGreaterThan(PARITY_LIMIT * 100);
    }
  });

  it("entropy of a one-hot distribution is zero and of a uniform one is log2 n", () => {
    expect(entropyBits(Float32Array.from([1, 0, 0, 0]))).toBe(0);
    expect(entropyBits(Float32Array.from([0.25, 0.25, 0.25, 0.25]))).toBeCloseTo(2, 10);
  });
});

