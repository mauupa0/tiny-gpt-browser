export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Temperature scaling, then top-k truncation, then softmax. Tokens outside the
// top-k get probability exactly 0. Temperature 0 collapses to greedy.
export function distribution(logits: Float32Array, temperature: number, topK: number): Float32Array {
  const n = logits.length;
  const p = new Float32Array(n);
  if (temperature <= 1e-6) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    p[best] = 1;
    return p;
  }
  const k = Math.max(1, Math.min(topK, n));
  let cutoff = -Infinity;
  if (k < n) {
    const sorted = Array.from(logits).sort((a, b) => b - a);
    cutoff = sorted[k - 1];
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] >= cutoff && logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (logits[i] >= cutoff) {
      const e = Math.exp((logits[i] - max) / temperature);
      p[i] = e;
      sum += e;
    }
  }
  for (let i = 0; i < n; i++) p[i] /= sum;
  return p;
}

export function sampleFrom(p: Float32Array, rand: () => number): number {
  const r = rand();
  let acc = 0;
  let last = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      acc += p[i];
      last = i;
      if (r < acc) return i;
    }
  }
  return last;
}

export function topN(p: Float32Array, n: number): { id: number; p: number }[] {
  const idx = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, n);
  return idx.filter((i) => p[i] > 0).map((i) => ({ id: i, p: p[i] }));
}
