import { memo, useEffect, useMemo, useRef, useState, type PointerEvent as RPE } from "react";
import { pct, signed } from "../lib/format";
import { classifyHead, headIndex, importance, type HeadsData } from "../lib/heads";
import { ordinalBack } from "../lib/narrate";
import { RAMP_LUT } from "../lib/palette";
import { spoken } from "../lib/text";
import type { Inspection } from "../model/inspect";
import { unpackAttn } from "../model/stats";
import "../styles/atlas.css";

interface Props {
  insp: Inspection | null;
  chars: readonly string[];
  heads: HeadsData;
}

interface Hot {
  row: number;
  col: number;
  lh: number;
  x: number;
  y: number;
}

const quoted = (ch: string) => `“${spoken(ch)}”`;
const REACH_COLORS = [6, 5, 4, 3, 2, 1, 0].map((i) => `var(--ramp-${i})`);

const HeadMap = memo(function HeadMap({ q, n, lh, label, onHot, hot }: { q: Uint8Array; n: number; lh: number; label: string; onHot: (h: Hot | null) => void; hot: Hot | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const g = cv?.getContext("2d");
    if (!cv || !g) return;
    cv.width = n;
    cv.height = n;
    const img = g.createImageData(n, n);
    const px = new Uint32Array(img.data.buffer);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c <= r; c++) px[r * n + c] = RAMP_LUT[q[r * n + c]];
    }
    g.putImageData(img, 0, 0);
  }, [q, n]);

  const locate = (e: RPE<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * n)));
    const row = Math.min(n - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * n)));
    return { row, col: Math.min(col, row), lh, x: e.clientX, y: e.clientY };
  };

  return (
    <div className="map">
      <canvas
        ref={ref}
        role="img"
        aria-label={label}
        onPointerMove={(e) => onHot(locate(e))}
        onPointerDown={(e) => onHot(locate(e))}
        onPointerLeave={() => onHot(null)}
      />
      <i className="last-row" style={{ height: `${100 / n}%` }} aria-hidden="true" />
      {hot && (
        <>
          <i className="hair h" style={{ top: `${((hot.row + 0.5) / n) * 100}%` }} aria-hidden="true" />
          <i className="hair v" style={{ left: `${((hot.col + 0.5) / n) * 100}%` }} aria-hidden="true" />
        </>
      )}
    </div>
  );
});

export const Atlas = memo(function Atlas({ insp, chars, heads }: Props) {
  const [hot, setHot] = useState<Hot | null>(null);
  const { top, ratio } = useMemo(() => importance(heads), [heads]);
  const n = insp?.n ?? 0;
  const maps = useMemo(() => {
    if (!insp) return [];
    return Array.from({ length: insp.layers * insp.heads }, (_, lh) => insp.attn.subarray(lh * n * n, (lh + 1) * n * n));
  }, [insp, n]);

  const readout = (() => {
    if (!insp || !hot) return null;
    const qv = maps[hot.lh][hot.row * n + hot.col];
    const l = Math.floor(hot.lh / insp.heads) + 1;
    const h = (hot.lh % insp.heads) + 1;
    return `Head ${l}.${h}: ${quoted(chars[insp.ids[hot.row]] ?? "")} reads ${quoted(chars[insp.ids[hot.col]] ?? "")}, ${ordinalBack(hot.row - hot.col)}: ${pct(unpackAttn(qv), 1)}`;
  })();

  return (
    <div className="atlas">
      <div className="reach-legend small" aria-hidden="true">
        <span>How far back a head looks, over {heads.windows} held-out windows:</span>
        {["0", "1", "2", "3", "4-8", "9-32", "33+"].map((b, i) => (
          <span key={b} className="lg">
            <i style={{ background: REACH_COLORS[i] }} />
            {b}
          </span>
        ))}
      </div>
      <div className="atlas-grid" onPointerLeave={() => setHot(null)}>
        {Array.from({ length: 16 }, (_, lh) => {
          const l = Math.floor(lh / 4) + 1;
          const h = (lh % 4) + 1;
          const stat = heads.heads[headIndex(l, h)];
          const role = classifyHead(stat);
          const best = stat === top;
          return (
            <figure className={`head${best ? " best" : ""}`} key={lh}>
              <figcaption>
                <b className="num">{l}.{h}</b> <span>{role.label}</span>
              </figcaption>
              {insp ? (
                <HeadMap
                  q={maps[lh]}
                  n={n}
                  lh={lh}
                  hot={hot}
                  onHot={setHot}
                  label={`Attention of head ${l}.${h} for every letter of the window. ${role.detail}.`}
                />
              ) : (
                <div className="map empty" />
              )}
              <div className="fp" role="img" aria-label={`Attention distance: ${stat.offsets.map((v, i) => `${heads.buckets[i]} letters back ${pct(v)}`).join(", ")}`}>
                {stat.offsets.map((v, i) => (
                  <i key={i} style={{ flexGrow: Math.max(v, 0.004), background: REACH_COLORS[i] }} />
                ))}
              </div>
              <p className="hloss small">
                zeroed: <span className="num">{signed(stat.dLoss, 3)}</span> nats per letter
              </p>
            </figure>
          );
        })}
      </div>
      <p className="readout small" aria-live="off">
        {readout ?? "Hover a map. Rows are the letter doing the reading, columns the letter it reads; the same cell lights up in all 16 heads."}
      </p>
      <p className="finding">
        Zeroing head {top.layer}.{top.head} ({classifyHead(top).label.toLowerCase()}) raises the loss by {top.dLoss.toFixed(3)} nats per letter, {ratio.toFixed(1)} times the damage of the next most important head.
      </p>
      {hot && readout && (
        <div className="tip" style={{ left: hot.x, top: hot.y }}>
          {readout}
        </div>
      )}
    </div>
  );
});
