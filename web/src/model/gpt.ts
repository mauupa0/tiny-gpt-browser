import type { Fault, ModelConfig, Weights } from "./types";

export interface KVCache {
  k: Float32Array[];
  v: Float32Array[];
  len: number;
}

export interface StepResult {
  logits: Float32Array;
  attn: Float32Array;
}

// Everything one position computes on its way to the logits, copied out so the
// page can draw it. Produced only by stepTraced; the hot path never allocates it.
export interface BlockTrace {
  ln1: Float32Array;
  headNorm: Float32Array;
  headOut: Float32Array[];
  attnBias: Float32Array;
  xMid: Float32Array;
  mlp: Float32Array;
  mlpOut: Float32Array;
  xOut: Float32Array;
}

export interface Trace {
  tok: Float32Array;
  pos: Float32Array;
  embed: Float32Array;
  blocks: BlockTrace[];
  final: Float32Array;
  // Residual stream at each stage: embedding, then after attention and after the MLP of every block.
  stages: Float32Array[];
  // Logit lens: the stream at every stage pushed through the final LayerNorm and the tied output matrix.
  lens: Float32Array[];
  lnScale: number;
}

export interface TracedStep extends StepResult {
  trace: Trace;
}

// Signed push of every component on one output logit, in logit units.
// embed + sum(heads) + sum(mlp) + bias + constant equals the logit.
export interface Attribution {
  embed: number;
  heads: Float32Array;
  mlp: Float32Array;
  bias: number;
  constant: number;
}

const GELU_C = Math.sqrt(2 / Math.PI);
const LN_EPS = 1e-5;

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array): void {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const c = x[i] - mean;
    variance += c * c;
  }
  const inv = 1 / Math.sqrt(variance / n + LN_EPS);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// y = W x + b with W row-major [outDim, inDim], the layout of torch.nn.Linear.
function linear(w: Float32Array, b: Float32Array, x: Float32Array, out: Float32Array, outDim: number, inDim: number): void {
  for (let o = 0; o < outDim; o++) {
    let s = b[o];
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) s += w[row + i] * x[i];
    out[o] = s;
  }
}

export class GPT {
  readonly cfg: ModelConfig;
  readonly w: Weights;
  private readonly fault: Fault;
  private readonly d: number;
  private readonly hd: number;
  private readonly x: Float32Array;
  private readonly h: Float32Array;
  private readonly qkv: Float32Array;
  private readonly y: Float32Array;
  private readonly tmp: Float32Array;
  private readonly hidden: Float32Array;
  private readonly scores: Float32Array;
  // Heads whose output is zeroed (index layer * n_head + head). Null leaves the model intact.
  mute: Uint8Array | null = null;

  constructor(cfg: ModelConfig, weights: Weights, fault: Fault = "none") {
    this.cfg = cfg;
    this.w = weights;
    this.fault = fault;
    this.d = cfg.d_model;
    this.hd = cfg.d_model / cfg.n_head;
    this.x = new Float32Array(this.d);
    this.h = new Float32Array(this.d);
    this.qkv = new Float32Array(3 * this.d);
    this.y = new Float32Array(this.d);
    this.tmp = new Float32Array(this.d);
    this.hidden = new Float32Array(4 * this.d);
    this.scores = new Float32Array(cfg.ctx + 1);
  }

  createCache(): KVCache {
    const { n_layer, ctx } = this.cfg;
    return {
      k: Array.from({ length: n_layer }, () => new Float32Array(ctx * this.d)),
      v: Array.from({ length: n_layer }, () => new Float32Array(ctx * this.d)),
      len: 0,
    };
  }

  // Feeds one token at position cache.len. Returns next-token logits and this
  // position's attention probabilities laid out as [layer][head][0..pos].
  step(token: number, cache: KVCache): StepResult {
    return this.run(token, cache, null);
  }

  stepTraced(token: number, cache: KVCache): TracedStep {
    const trace: Trace = {
      tok: new Float32Array(0),
      pos: new Float32Array(0),
      embed: new Float32Array(0),
      blocks: [],
      final: new Float32Array(0),
      stages: [],
      lens: [],
      lnScale: 0,
    };
    const r = this.run(token, cache, trace);
    return { ...r, trace };
  }

  private run(token: number, cache: KVCache, tr: Trace | null): StepResult {
    const { d, hd, x, h, qkv, y, tmp, hidden, scores } = this;
    const { n_layer, n_head, ctx, vocab } = this.cfg;
    const pos = cache.len;
    if (pos >= ctx) throw new Error("Context window is full");
    const T = pos + 1;
    const attn = new Float32Array(n_layer * n_head * T);
    const te = token * d;
    const pe = pos * d;
    for (let i = 0; i < d; i++) x[i] = this.w.tok[te + i] + this.w.pos[pe + i];
    const scale = this.fault === "scale" ? 1 : 1 / Math.sqrt(hd);
    if (tr) {
      tr.tok = this.w.tok.slice(te, te + d);
      tr.pos = this.w.pos.slice(pe, pe + d);
      tr.embed = x.slice();
      tr.stages.push(x.slice());
    }

    for (let l = 0; l < n_layer; l++) {
      const B = this.w.blocks[l];
      layerNorm(x, B.ln1w, B.ln1b, h);
      linear(B.qkvw, B.qkvb, h, qkv, 3 * d, d);
      cache.k[l].set(qkv.subarray(d, 2 * d), pos * d);
      cache.v[l].set(qkv.subarray(2 * d, 3 * d), pos * d);
      const K = cache.k[l];
      const V = cache.v[l];
      // Fault "mask": the query also attends to the stale slot after it, which a
      // causal mask forbids. Used only by the negative-control tests.
      const span = this.fault === "mask" && T < ctx ? T + 1 : T;
      for (let hh = 0; hh < n_head; hh++) {
        const qo = hh * hd;
        let max = -Infinity;
        for (let t = 0; t < span; t++) {
          let s = 0;
          const ko = t * d + qo;
          for (let i = 0; i < hd; i++) s += qkv[qo + i] * K[ko + i];
          s *= scale;
          scores[t] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t < span; t++) {
          const e = Math.exp(scores[t] - max);
          scores[t] = e;
          sum += e;
        }
        const ao = (l * n_head + hh) * T;
        for (let i = 0; i < hd; i++) y[qo + i] = 0;
        for (let t = 0; t < span; t++) {
          const p = scores[t] / sum;
          if (t < T) attn[ao + t] = p;
          const vo = t * d + qo;
          for (let i = 0; i < hd; i++) y[qo + i] += p * V[vo + i];
        }
        if (this.mute && this.mute[l * n_head + hh]) for (let i = 0; i < hd; i++) y[qo + i] = 0;
      }
      const bt = tr ? this.headWrites(B.projw, y, d, n_head, hd) : null;
      linear(B.projw, B.projb, y, tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
      if (tr && bt) {
        tr.stages.push(x.slice());
        tr.blocks.push({
          ln1: h.slice(),
          headNorm: Float32Array.from(bt, (v) => Math.hypot(...v)),
          headOut: bt,
          attnBias: B.projb.slice(),
          xMid: x.slice(),
          mlp: new Float32Array(0),
          mlpOut: new Float32Array(0),
          xOut: new Float32Array(0),
        });
      }
      layerNorm(x, B.ln2w, B.ln2b, h);
      linear(B.fcw, B.fcb, h, hidden, 4 * d, d);
      for (let i = 0; i < 4 * d; i++) {
        const v = hidden[i];
        hidden[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
      }
      linear(B.outw, B.outb, hidden, tmp, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
      if (tr) {
        const bl = tr.blocks[l];
        bl.mlp = hidden.slice();
        bl.mlpOut = tmp.slice();
        bl.xOut = x.slice();
        tr.stages.push(x.slice());
      }
    }

    layerNorm(x, this.w.lnfw, this.w.lnfb, h);
    const logits = new Float32Array(vocab);
    for (let v = 0; v < vocab; v++) {
      let s = 0;
      const o = v * d;
      for (let i = 0; i < d; i++) s += this.w.tok[o + i] * h[i];
      logits[v] = s;
    }
    if (tr) {
      tr.final = h.slice();
      tr.lnScale = this.lnScale(x);
      tr.lens = tr.stages.map((st) => this.project(st));
    }
    cache.len = pos + 1;
    return { logits, attn };
  }

  // What each head writes into the residual stream: its slice of the output
  // projection applied to its own mixed values, bias excluded.
  private headWrites(projw: Float32Array, y: Float32Array, d: number, nHead: number, hd: number): Float32Array[] {
    const out: Float32Array[] = [];
    for (let hh = 0; hh < nHead; hh++) {
      const v = new Float32Array(d);
      const qo = hh * hd;
      for (let o = 0; o < d; o++) {
        let s = 0;
        const row = o * d + qo;
        for (let i = 0; i < hd; i++) s += projw[row + i] * y[qo + i];
        v[o] = s;
      }
      out.push(v);
    }
    return out;
  }

  private lnScale(x: Float32Array): number {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const c = x[i] - mean;
      variance += c * c;
    }
    return 1 / Math.sqrt(variance / n + LN_EPS);
  }

  // Logit lens: final LayerNorm followed by the tied output matrix.
  project(stream: Float32Array): Float32Array {
    const { d } = this;
    const { vocab } = this.cfg;
    const z = new Float32Array(d);
    layerNorm(stream, this.w.lnfw, this.w.lnfb, z);
    const out = new Float32Array(vocab);
    for (let v = 0; v < vocab; v++) {
      let s = 0;
      const o = v * d;
      for (let i = 0; i < d; i++) s += this.w.tok[o + i] * z[i];
      out[v] = s;
    }
    return out;
  }

  // Direct effect of one residual-stream contribution on the logit of `char`.
  // The final LayerNorm is frozen at this position's statistics, which makes the
  // sum over all contributions plus the constant term equal the real logit.
  attribution(trace: Trace, char: number): Attribution {
    const { d } = this;
    const { n_layer, n_head } = this.cfg;
    const tok = this.w.tok;
    const lw = this.w.lnfw;
    const lb = this.w.lnfb;
    const o = char * d;
    const push = (vec: Float32Array): number => {
      let mean = 0;
      for (let i = 0; i < d; i++) mean += vec[i];
      mean /= d;
      let s = 0;
      for (let i = 0; i < d; i++) s += tok[o + i] * lw[i] * (vec[i] - mean);
      return s * trace.lnScale;
    };
    const heads = new Float32Array(n_layer * n_head);
    const mlp = new Float32Array(n_layer);
    let constant = 0;
    for (let i = 0; i < d; i++) constant += tok[o + i] * lb[i];
    let bias = 0;
    for (let l = 0; l < n_layer; l++) {
      const b = trace.blocks[l];
      for (let hh = 0; hh < n_head; hh++) heads[l * n_head + hh] = push(b.headOut[hh]);
      mlp[l] = push(b.mlpOut);
      bias += push(b.attnBias);
    }
    return { embed: push(trace.embed), heads, mlp, bias, constant };
  }
}
