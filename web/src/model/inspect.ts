import { GPT, type Attribution, type KVCache } from "./gpt";
import { argmax, klNats, l2, packAttn, softmax } from "./stats";

export interface Ablation {
  // KL(intact || head zeroed) in nats at the inspected position, index layer * heads + head.
  kl: Float32Array;
  // Probability of the intact top character after zeroing the head, minus before.
  dTop: Float32Array;
}

// Everything the page draws about one position of one context window.
export interface Inspection {
  ids: number[];
  n: number;
  layers: number;
  heads: number;
  // Attention of every query row, square-root coded bytes, [layer][head][row][col].
  attn: Uint8Array;
  // Exact attention of the last row, [layer][head][col].
  last: Float32Array;
  logits: Float32Array;
  probs: Float32Array;
  top: number;
  // Next-character probabilities read off the stream at each stage (logit lens).
  lens: Float32Array[];
  norms: Float32Array;
  headNorm: Float32Array;
  mlp: Float32Array;
  tok: Float32Array;
  pos: Float32Array;
  embed: Float32Array;
  target: number;
  logit: number;
  attribution: Attribution;
  ablation: Ablation | null;
}

// Re-reads a context through the model and keeps the KV cache between calls, so
// moving the inspected position by one character costs one step instead of a
// full pass. K and V of a position depend only on the tokens before it.
export class Inspector {
  readonly gpt: GPT;
  private cache: KVCache;
  private ids: number[] = [];
  private rows: Float32Array[] = [];

  constructor(gpt: GPT) {
    this.gpt = gpt;
    this.cache = gpt.createCache();
  }

  // `input` is the context the model sees; the prediction is for the character after its last one.
  // `target` is the character whose logit gets decomposed, default the model's favourite.
  inspect(input: number[], opts: { target?: number | null; ablate?: boolean } = {}): Inspection {
    const { gpt } = this;
    const { n_layer: L, n_head: H, ctx } = gpt.cfg;
    const ids = input.slice(-ctx);
    const n = ids.length;
    if (n === 0) throw new Error("Empty context");

    let lcp = 0;
    const known = Math.min(this.ids.length, n - 1);
    while (lcp < known && this.ids[lcp] === ids[lcp]) lcp++;
    this.cache.len = lcp;
    this.rows.length = lcp;
    for (let p = lcp; p < n - 1; p++) this.rows[p] = gpt.step(ids[p], this.cache).attn;

    const last = ids[n - 1];
    gpt.mute = null;
    const base = gpt.stepTraced(last, this.cache);
    this.rows[n - 1] = base.attn;
    this.ids = ids.slice();
    const probs = softmax(base.logits);
    const top = argmax(probs);
    const target = opts.target ?? top;

    let ablation: Ablation | null = null;
    if (opts.ablate) ablation = this.ablate(last, n - 1, probs, top);

    const trace = base.trace;
    const attn = new Uint8Array(L * H * n * n);
    for (let p = 0; p < n; p++) {
      const row = this.rows[p];
      const T = p + 1;
      for (let lh = 0; lh < L * H; lh++) {
        const dst = (lh * n + p) * n;
        const src = lh * T;
        for (let t = 0; t < T; t++) attn[dst + t] = packAttn(row[src + t]);
      }
    }
    const mlp = new Float32Array(L * trace.blocks[0].mlp.length);
    trace.blocks.forEach((b, l) => mlp.set(b.mlp, l * b.mlp.length));
    const headNorm = new Float32Array(L * H);
    trace.blocks.forEach((b, l) => headNorm.set(b.headNorm, l * H));
    this.cache.len = n;

    return {
      ids,
      n,
      layers: L,
      heads: H,
      attn,
      last: base.attn,
      logits: base.logits,
      probs,
      top,
      lens: trace.lens.map(softmax),
      norms: Float32Array.from(trace.stages, l2),
      headNorm,
      mlp,
      tok: trace.tok,
      pos: trace.pos,
      embed: trace.embed,
      target,
      logit: base.logits[target],
      attribution: gpt.attribution(trace, target),
      ablation,
    };
  }

  // Zeroes one head at a time at the inspected position only. Earlier positions
  // are untouched because the model is causal, so each probe is one step.
  private ablate(token: number, pos: number, probs: Float32Array, top: number): Ablation {
    const { gpt } = this;
    const { n_layer: L, n_head: H, d_model: d } = gpt.cfg;
    const saved = this.cache.k.map((k) => k.slice(pos * d, (pos + 1) * d));
    const savedV = this.cache.v.map((v) => v.slice(pos * d, (pos + 1) * d));
    const kl = new Float32Array(L * H);
    const dTop = new Float32Array(L * H);
    const mask = new Uint8Array(L * H);
    for (let i = 0; i < L * H; i++) {
      mask.fill(0);
      mask[i] = 1;
      gpt.mute = mask;
      this.cache.len = pos;
      const q = softmax(gpt.step(token, this.cache).logits);
      kl[i] = klNats(probs, q);
      dTop[i] = q[top] - probs[top];
    }
    gpt.mute = null;
    this.cache.k.forEach((k, l) => k.set(saved[l], pos * d));
    this.cache.v.forEach((v, l) => v.set(savedV[l], pos * d));
    this.cache.len = pos + 1;
    return { kl, dTop };
  }
}
