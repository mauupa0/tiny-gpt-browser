import type { Inspection } from "../model/inspect";

// Facts about one inspection, for the captions beside the figure. Every sentence the page
// shows is built from these numbers.

export function rankOf(probs: ArrayLike<number>, id: number): number {
  let rank = 1;
  for (let i = 0; i < probs.length; i++) if (probs[i] > probs[id]) rank++;
  return rank;
}

// Where a head looks hardest: the column with the largest weight in its last row.
export function peakOf(insp: Inspection, layer: number, head: number): { pos: number; back: number; p: number } {
  const T = insp.n;
  const base = (layer * insp.heads + head) * T;
  let pos = 0;
  for (let t = 1; t < T; t++) if (insp.last[base + t] > insp.last[base + pos]) pos = t;
  return { pos, back: T - 1 - pos, p: insp.last[base + pos] };
}

export function activeNeurons(insp: Inspection, layer: number, threshold = 0.1): number {
  const width = insp.mlp.length / insp.layers;
  let n = 0;
  for (let i = 0; i < width; i++) if (insp.mlp[layer * width + i] > threshold) n++;
  return n;
}

// The head of a block with the largest push, in either direction, on the explained letter.
export function topPush(insp: Inspection, layer: number): { head: number; push: number } {
  let head = 0;
  for (let h = 1; h < insp.heads; h++) {
    if (Math.abs(insp.attribution.heads[layer * insp.heads + h]) > Math.abs(insp.attribution.heads[layer * insp.heads + head])) head = h;
  }
  return { head, push: insp.attribution.heads[layer * insp.heads + head] };
}

export function pushScale(insp: Inspection): number {
  const a = insp.attribution;
  let m = Math.abs(a.embed);
  for (const v of a.heads) m = Math.max(m, Math.abs(v));
  for (const v of a.mlp) m = Math.max(m, Math.abs(v));
  return m || 1;
}

export function pushTotal(insp: Inspection): number {
  const a = insp.attribution;
  let s = a.embed + a.bias + a.constant;
  for (const v of a.heads) s += v;
  for (const v of a.mlp) s += v;
  return s;
}

export function ordinalBack(back: number): string {
  if (back === 0) return "on itself";
  if (back === 1) return "1 letter back";
  return `${back} letters back`;
}
