import { Engine, type FeedResult } from "./engine";
import type { GPT } from "./gpt";
import { distribution, mulberry32, sampleFrom, topN } from "./sample";
import { entropyBits } from "./stats";
import type { Fault } from "./types";

export interface LaneSpec {
  temperature: number;
  fault: Fault;
  mute: number[];
}

export interface LaneToken {
  lane: number;
  id: number;
  top: { id: number; p: number }[];
  p: number;
  entropy: number;
  ms: number;
}

interface Lane {
  engine: Engine;
  feed: FeedResult;
  temperature: number;
  ms: number;
  tokens: number;
}

// Several writers reading one prompt in lockstep. Every step draws a single uniform
// number and hands it to all lanes (common random numbers), so lanes with the same
// settings write the same text and any difference between lanes comes from the model
// or the temperature, never from luck.
export class Lanes {
  private readonly lanes: Lane[];
  private readonly rand: () => number;
  private topK: number;

  constructor(make: (spec: LaneSpec) => GPT, prompt: number[], specs: LaneSpec[], seed: number, topK: number) {
    this.rand = mulberry32(seed);
    this.topK = topK;
    this.lanes = specs.map((s) => {
      const gpt = make(s);
      const engine = new Engine(gpt);
      return { engine, feed: engine.start(prompt), temperature: s.temperature, ms: 0, tokens: 0 };
    });
  }

  setParams(temperatures: number[], topK: number): void {
    this.topK = topK;
    this.lanes.forEach((l, i) => {
      if (temperatures[i] !== undefined) l.temperature = temperatures[i];
    });
  }

  step(): LaneToken[] {
    const u = this.rand();
    return this.lanes.map((l, lane) => {
      const t0 = performance.now();
      const p = distribution(l.feed.logits, l.temperature, this.topK);
      const id = sampleFrom(p, () => u);
      const top = topN(p, 8);
      const entropy = entropyBits(p);
      l.feed = l.engine.push(id);
      const ms = performance.now() - t0;
      l.ms += ms;
      l.tokens++;
      return { lane, id, top, p: p[id], entropy, ms };
    });
  }

  stats(): { tokens: number; computeMs: number }[] {
    return this.lanes.map((l) => ({ tokens: l.tokens, computeMs: l.ms }));
  }
}
