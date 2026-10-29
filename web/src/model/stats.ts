export function softmax(logits: Float32Array): Float32Array {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  const p = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    p[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) p[i] /= sum;
  return p;
}

export function argmax(a: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

// Shannon entropy in bits.
export function entropyBits(p: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i]);
  return h;
}

// KL(p || q) in nats. Both must be full softmax outputs (q > 0 wherever p > 0).
export function klNats(p: ArrayLike<number>, q: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / Math.max(q[i], 1e-30));
  return s;
}

export function l2(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

// Attention probabilities travel as bytes. Square-root coding keeps resolution
// near zero, where most cells of a sharp head sit. Absolute error is at most
// sqrt(p) / 255 + 1 / 255^2 (checked in the unit tests).
export function packAttn(p: number): number {
  return Math.round(Math.sqrt(Math.min(1, Math.max(0, p))) * 255);
}

export function unpackAttn(q: number): number {
  const v = q / 255;
  return v * v;
}
