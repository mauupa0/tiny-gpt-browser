import { memo, type CSSProperties } from "react";
import { glyphKind, spoken } from "../lib/text";
import { pct } from "../lib/format";
import type { Prob } from "../worker/protocol";

interface Props {
  probs: Prob[];
  chars: readonly string[];
  chosen: number | null;
  columns?: number;
  height?: number;
  label: string;
  stale?: boolean;
}

function Glyph({ ch }: { ch: string }) {
  const k = glyphKind(ch);
  if (k === "space") return <i className="gl-space" aria-hidden="true" />;
  if (k === "newline") return <i className="gl-break" aria-hidden="true" />;
  return <>{ch}</>;
}

export const Candidates = memo(function Candidates({ probs, chars, chosen, columns = 8, height = 60, label, stale = false }: Props) {
  const shown = probs.slice(0, columns);
  return (
    <figure className={stale ? "cands stale" : "cands"} aria-busy={stale} aria-label={label} style={{ "--h": `${height}px`, "--cols": columns } as CSSProperties}>
      <ol>
        {Array.from({ length: columns }, (_, i) => {
          const c = shown[i];
          if (!c) return <li key={i} className="empty" aria-hidden="true" />;
          const ch = chars[c.id] ?? "";
          return (
            <li key={i} className={c.id === chosen ? "on" : undefined}>
              <span className="bar" aria-hidden="true">
                <span style={{ transform: `scaleY(${Math.max(c.p, 0.004)})` }} />
              </span>
              <span className="g">
                <Glyph ch={ch} />
              </span>
              <span className="p num">{pct(c.p, c.p < 0.1 ? 1 : 0)}</span>
              <span className="sr-only">
                {spoken(ch)}, {pct(c.p, 1)}
              </span>
            </li>
          );
        })}
      </ol>
    </figure>
  );
});
