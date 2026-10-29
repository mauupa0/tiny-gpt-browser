import type { GPT } from "./gpt";

// Logits PyTorch produced for a fixed verse line, from the same fp16-rounded weights the page ships.
export interface ParityRef {
  text: string;
  ids: number[];
  logits: number[][];
}

export const PARITY_LIMIT = 1e-3;

export interface ParityResult {
  max: number;
  perPos: number[];
}

// Runs the verse through the model token by token with the KV cache and reports
// the largest absolute logit difference per position and overall.
export function parityCheck(gpt: GPT, ref: ParityRef): ParityResult {
  const cache = gpt.createCache();
  const perPos: number[] = [];
  ref.ids.forEach((id, t) => {
    const r = gpt.step(id, cache);
    let m = 0;
    for (let v = 0; v < r.logits.length; v++) m = Math.max(m, Math.abs(r.logits[v] - ref.logits[t][v]));
    perPos.push(m);
  });
  return { max: Math.max(...perPos), perPos };
}
