import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { GPT } from "../src/model/gpt";
import type { Fault, Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const TOL = 1e-3;
const here = (p: string) => resolve(__dirname, p);

interface RefCase {
  text: string;
  ids: number[];
  logits: number[][];
  attn?: number[][][][];
}
type Ref = Record<"float32" | "float16", Record<string, RefCase>>;

const ref = JSON.parse(readFileSync(here("fixtures/reference.json"), "utf8")) as Ref;

function load(manifestPath: string, binPath: string, fault: Fault = "none") {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
  const b = readFileSync(binPath);
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  return { manifest, gpt: new GPT(manifest.config, parseWeights(manifest, buf), fault) };
}

function compare(gpt: GPT, c: RefCase) {
  const cache = gpt.createCache();
  let maxLogit = 0;
  let maxAttn = 0;
  c.ids.forEach((id, t) => {
    const r = gpt.step(id, cache);
    for (let v = 0; v < r.logits.length; v++) {
      maxLogit = Math.max(maxLogit, Math.abs(r.logits[v] - c.logits[t][v]));
    }
    if (c.attn) {
      const T = t + 1;
      c.attn.forEach((layer, l) =>
        layer.forEach((head, h) => {
          for (let k = 0; k < T; k++) {
            maxAttn = Math.max(maxAttn, Math.abs(r.attn[(l * layer.length + h) * T + k] - head[t][k]));
          }
        }),
      );
    }
  });
  return { maxLogit, maxAttn };
}

const results: Record<string, unknown> = {};
const shipped = load(here("../public/model/manifest.json"), here("../public/model/weights.f16.bin"));
const f32 = load(here("fixtures/manifest.f32.json"), here("fixtures/weights.f32.bin"));

describe("TypeScript inference matches PyTorch", () => {
  for (const [dtype, m] of [["float32", f32], ["float16", shipped]] as const) {
    it(`${dtype} weights: logits and attention within ${TOL}`, () => {
      const per: Record<string, { maxLogit: number; maxAttn: number; tokens: number }> = {};
      for (const [name, c] of Object.entries(ref[dtype])) {
        const r = compare(m.gpt, c);
        per[name] = { ...r, tokens: c.ids.length };
        expect(r.maxLogit, `${dtype}/${name} logits`).toBeLessThan(TOL);
        expect(r.maxAttn, `${dtype}/${name} attention`).toBeLessThan(TOL);
      }
      const worst = Math.max(...Object.values(per).map((p) => p.maxLogit));
      results[dtype] = { tolerance: TOL, worstLogitDiff: worst, cases: per };
    });
  }

  it("covers a full 128-token context window", () => {
    expect(ref.float32.full_ctx.ids.length).toBe(f32.manifest.config.ctx);
  });
});

describe("negative control: a broken implementation must fail the same check", () => {
  for (const fault of ["scale", "mask"] as const) {
    it(`fault "${fault}" exceeds the tolerance`, () => {
      const broken = load(here("fixtures/manifest.f32.json"), here("fixtures/weights.f32.bin"), fault);
      const diffs = Object.entries(ref.float32).map(([name, c]) => ({ name, ...compare(broken.gpt, c) }));
      const worst = Math.max(...diffs.map((d) => d.maxLogit));
      results[`fault_${fault}`] = { worstLogitDiff: worst };
      expect(worst).toBeGreaterThan(TOL);
    });
  }
});

afterAll(() => {
  if (Object.keys(results).length >= 4) {
    writeFileSync(here("../src/generated/parity.json"), JSON.stringify(results, null, 1));
  }
});
