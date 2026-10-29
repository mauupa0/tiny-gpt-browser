import { pct } from "./format";

// What one attention head does on held-out text, measured by scripts/head-stats.ts.
export interface HeadStat {
  layer: number;
  head: number;
  offsets: number[];
  spaces: number;
  punct: number;
  same: number;
  dLoss: number;
}

export interface HeadsData {
  windows: number;
  queries: number;
  baseLoss: number;
  buckets: string[];
  heads: HeadStat[];
}

export interface HeadRole {
  key: "boundaries" | "previous" | "two" | "three" | "far" | "mid" | "self" | "mixed";
  label: string;
  detail: string;
}

// Rule-based reading of the measured attention profile. The thresholds are plain
// majorities, so the label is a statement about the numbers and nothing more.
export function classifyHead(h: HeadStat): HeadRole {
  const [o0, o1, o2, o3, o48, o932, o33] = h.offsets;
  if (h.spaces >= 0.5) return { key: "boundaries", label: "Word boundaries", detail: `${pct(h.spaces)} of its attention lands on spaces and line breaks` };
  if (o1 >= 0.85) return { key: "previous", label: "Previous letter", detail: `${pct(o1)} on the letter just before` };
  if (o0 >= 0.5) return { key: "self", label: "Itself", detail: `${pct(o0)} on the current letter` };
  if (o1 + o2 >= 0.7) return { key: "two", label: "Last two letters", detail: `${pct(o1 + o2)} on the two letters before` };
  if (o1 + o2 + o3 >= 0.6) return { key: "three", label: "Last three letters", detail: `${pct(o1 + o2 + o3)} on the three letters before` };
  if (o33 >= 0.4) return { key: "far", label: "Far back", detail: `${pct(o33)} on letters more than 32 back` };
  if (o48 + o932 >= 0.5) return { key: "mid", label: "Mid range", detail: `${pct(o48 + o932)} on letters 4 to 32 back` };
  return { key: "mixed", label: "Mixed", detail: "attention split between the last few letters and the last 32" };
}

export function importance(data: HeadsData): { top: HeadStat; ratio: number } {
  const sorted = [...data.heads].sort((a, b) => b.dLoss - a.dLoss);
  return { top: sorted[0], ratio: sorted[0].dLoss / sorted[1].dLoss };
}

export const headIndex = (layer: number, head: number, heads = 4): number => (layer - 1) * heads + (head - 1);
