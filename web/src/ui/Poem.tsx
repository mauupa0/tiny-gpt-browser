import { memo, useCallback, type CSSProperties } from "react";
import { glyphKind, spoken, splitLines } from "../lib/text";

interface PoemProps {
  chars: readonly string[];
  ids: readonly number[];
  promptLen: number;
  // Probability the sampler gave each written character; prompt characters carry 1.
  ps: readonly number[];
  selected: number;
  running: boolean;
  onSelect?: (index: number) => void;
  label: string;
  size?: "hero" | "col";
}

interface CharProps {
  ch: string;
  index: number;
  prompt: boolean;
  surprise: number;
  selected: boolean;
  fresh: boolean;
}

const Char = memo(function Char({ ch, index, prompt, surprise, selected, fresh }: CharProps) {
  const kind = glyphKind(ch);
  const cls = ["c", prompt ? "prompt" : "gen", selected ? "sel" : "", fresh ? "fresh" : "", kind === "newline" ? "nl" : ""].filter(Boolean).join(" ");
  return (
    <span className={cls} data-i={index} style={prompt ? undefined : ({ "--s": surprise } as CSSProperties)}>
      {kind === "newline" ? "" : ch}
    </span>
  );
});

// Underline strength of a sampled character: none from 25% up, full for a long shot.
export const GAMBLE = 0.25;
export const surpriseOf = (p: number): number => (p >= GAMBLE ? 0 : (1 - p / GAMBLE) ** 1.5);

export const Poem = memo(function Poem({ chars, ids, promptLen, ps, selected, running, onSelect, label, size = "hero" }: PoemProps) {
  const all = splitLines(ids, chars);
  // A poem that ends on a line break has an empty last line; it would only be blank space under the verse.
  const lines = all.length > 1 && all[all.length - 1].length === 0 ? all.slice(0, -1) : all;
  const click = useCallback(
    (e: React.MouseEvent) => {
      if (!onSelect) return;
      const t = (e.target as HTMLElement).closest<HTMLElement>("[data-i]");
      if (t) onSelect(Number(t.dataset.i));
    },
    [onSelect],
  );
  const last = ids.length - 1;
  return (
    <div className={`poem poem-${size}`} role="region" aria-label={label} onClick={click}>
      {lines.map((line, li) => (
        <span className={line.length === 1 && glyphKind(chars[line[0].id] ?? "") === "newline" ? "l blank" : "l"} key={li}>
          {line.map(({ index, id }) => (
            <Char
              key={index}
              ch={chars[id] ?? ""}
              index={index}
              prompt={index < promptLen}
              surprise={index < promptLen || glyphKind(chars[id] ?? "") === "newline" ? 0 : surpriseOf(ps[index] ?? 1)}
              selected={index === selected}
              fresh={running && index === last && index >= promptLen}
            />
          ))}
          {running && li === lines.length - 1 && <span className="caret" aria-hidden="true" />}
        </span>
      ))}
      <span className="sr-only">
        {ids.length && selected >= 0 ? `Selected character: ${spoken(chars[ids[selected]] ?? "")}, position ${selected + 1} of ${ids.length}.` : ""}
      </span>
    </div>
  );
});
