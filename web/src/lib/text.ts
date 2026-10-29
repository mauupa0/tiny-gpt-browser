export function encodeText(chars: readonly string[], text: string): { ids: number[]; dropped: number } {
  const index = new Map(chars.map((c, i) => [c, i]));
  const ids: number[] = [];
  let dropped = 0;
  for (const ch of text.replace(/\r/g, "")) {
    const id = index.get(ch);
    if (id === undefined) dropped++;
    else ids.push(id);
  }
  return { ids, dropped };
}

export function decode(chars: readonly string[], ids: readonly number[]): string {
  return ids.map((i) => chars[i] ?? "").join("");
}

export type GlyphKind = "char" | "space" | "newline";

export function glyphKind(ch: string): GlyphKind {
  if (ch === " ") return "space";
  if (ch === "\n") return "newline";
  return "char";
}

// How a character is named for people who cannot see the glyph.
export function spoken(ch: string): string {
  if (ch === " ") return "space";
  if (ch === "\n") return "line break";
  return ch;
}

// Share of the n-grams in `ids` that already occurred earlier in the same text.
export function repeatShare(ids: readonly number[], n = 4): number {
  const total = ids.length - n + 1;
  if (total <= 0) return 0;
  const seen = new Set<string>();
  let repeated = 0;
  for (let i = 0; i < total; i++) {
    const g = ids.slice(i, i + n).join(",");
    if (seen.has(g)) repeated++;
    else seen.add(g);
  }
  return repeated / total;
}

// Lines of a poem as arrays of [text index, id] so every character keeps its place.
export function splitLines(ids: readonly number[], chars: readonly string[]): { index: number; id: number }[][] {
  const lines: { index: number; id: number }[][] = [[]];
  ids.forEach((id, index) => {
    lines[lines.length - 1].push({ index, id });
    if (chars[id] === "\n") lines.push([]);
  });
  return lines;
}

// Where to leave the figure when a poem is finished: the position just before the letter the model
// chose against the longest odds in the last stretch, so the candidates on screen include a long shot.
// Returns -1 when every recent letter was a safe pick (at least `limit` probability).
export function longShotIndex(
  ids: readonly number[],
  ps: readonly number[],
  chars: readonly string[],
  promptLen: number,
  window = 80,
  limit = 0.25,
): number {
  let best = -1;
  let bestP = limit;
  for (let j = Math.max(promptLen, 1, ids.length - window); j < ids.length; j++) {
    if (chars[ids[j]] === "\n") continue;
    if (ps[j] < bestP) {
      bestP = ps[j];
      best = j;
    }
  }
  return best < 0 ? -1 : best - 1;
}
