import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { GPT } from "../src/model/gpt";
import { Lanes, type LaneSpec } from "../src/model/lanes";
import type { Manifest } from "../src/model/types";
import { parseWeights } from "../src/model/weights";

const here = (p: string) => resolve(__dirname, p);
const manifest = JSON.parse(readFileSync(here("../public/model/manifest.json"), "utf8")) as Manifest;
const bin = readFileSync(here("../public/model/weights.f16.bin"));
const weights = parseWeights(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer);
const make = (s: LaneSpec) => {
  const g = new GPT(manifest.config, weights, s.fault);
  if (s.mute.length) {
    g.mute = new Uint8Array(16);
    s.mute.forEach((i) => (g.mute![i] = 1));
  }
  return g;
};
const enc = (s: string) => Array.from(s).map((c) => manifest.chars.indexOf(c));
const PROMPT = enc("Litwo! Ojczyzno moja!");
const none: LaneSpec = { temperature: 0.8, fault: "none", mute: [] };

function write(specs: LaneSpec[], seed: number, n: number, topK = 20): number[][] {
  const lanes = new Lanes(make, PROMPT, specs, seed, topK);
  const out: number[][] = specs.map(() => []);
  for (let i = 0; i < n; i++) lanes.step().forEach((t) => out[t.lane].push(t.id));
  return out;
}

describe("lanes", () => {
  it("lanes with the same settings write the same text (common random numbers)", () => {
    const [a, b, c] = write([none, none, none], 11, 80);
    expect(b).toEqual(a);
    expect(c).toEqual(a);
  });

  it("the same seed repeats a run and another seed does not", () => {
    const [a] = write([none], 5, 80);
    expect(write([none], 5, 80)[0]).toEqual(a);
    expect(write([none], 6, 80)[0]).not.toEqual(a);
  });

  it("a broken model and a silenced head write something else from the same stream", () => {
    const [good, scale, mask, muted] = write(
      [none, { ...none, fault: "scale" }, { ...none, fault: "mask" }, { ...none, mute: [4 * 4 - 1] }],
      21,
      120,
    );
    expect(scale).not.toEqual(good);
    expect(mask).not.toEqual(good);
    expect(muted).not.toEqual(good);
  });

  it("temperature changes the text, and a very low one repeats itself more", () => {
    const [cold, hot] = write([{ ...none, temperature: 0.2 }, { ...none, temperature: 1.4 }], 3, 200);
    const grams = (ids: number[]) => {
      const seen = new Set<string>();
      let rep = 0;
      for (let i = 0; i + 4 <= ids.length; i++) {
        const g = ids.slice(i, i + 4).join(",");
        if (seen.has(g)) rep++;
        seen.add(g);
      }
      return rep;
    };
    expect(cold).not.toEqual(hot);
    expect(grams(cold)).toBeGreaterThan(grams(hot));
  });

  it("every token reports a valid top list, entropy and its own probability", () => {
    const lanes = new Lanes(make, PROMPT, [none], 9, 20);
    for (let i = 0; i < 30; i++) {
      const t = lanes.step()[0];
      expect(t.top.length).toBeGreaterThan(0);
      expect(t.top.length).toBeLessThanOrEqual(8);
      expect(t.top[0].p).toBeGreaterThanOrEqual(t.top[t.top.length - 1].p);
      expect(t.p).toBeGreaterThan(0);
      expect(t.entropy).toBeGreaterThanOrEqual(0);
    }
    expect(lanes.stats()[0].tokens).toBe(30);
  });

  it("runs past the context window without throwing", () => {
    const out = write([none], 1, 300)[0];
    expect(out).toHaveLength(300);
  });
});
