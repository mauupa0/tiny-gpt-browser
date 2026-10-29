import { GPT, type KVCache } from "./gpt";

export interface AttnRow {
  pos: number;
  attn: Float32Array;
}

export interface FeedResult {
  logits: Float32Array;
  rows: AttnRow[];
  // True when the context window was rebuilt, so row positions restart at 0.
  reset: boolean;
  window: number[];
}

// Wraps the model with a sliding context window. Positional embeddings are
// absolute, so when the cache is full we re-encode the newest half of the window
// instead of shifting keys.
export class Engine {
  readonly gpt: GPT;
  private cache: KVCache;
  private window: number[] = [];
  private readonly ctx: number;

  constructor(gpt: GPT) {
    this.gpt = gpt;
    this.ctx = gpt.cfg.ctx;
    this.cache = gpt.createCache();
  }

  private prefill(ids: number[]): FeedResult {
    this.cache = this.gpt.createCache();
    this.window = [];
    const rows: AttnRow[] = [];
    let logits: Float32Array = new Float32Array(this.gpt.cfg.vocab);
    for (const id of ids) {
      const r = this.gpt.step(id, this.cache);
      logits = r.logits;
      rows.push({ pos: this.window.length, attn: r.attn });
      this.window.push(id);
    }
    return { logits, rows, reset: true, window: this.window.slice() };
  }

  start(ids: number[]): FeedResult {
    if (ids.length === 0) throw new Error("Empty prompt");
    return this.prefill(ids.slice(-(this.ctx - 1)));
  }

  push(id: number): FeedResult {
    if (this.cache.len >= this.ctx) {
      return this.prefill([...this.window.slice(-(this.ctx >> 1)), id]);
    }
    const r = this.gpt.step(id, this.cache);
    const pos = this.window.length;
    this.window.push(id);
    return { logits: r.logits, rows: [{ pos, attn: r.attn }], reset: false, window: this.window };
  }
}
