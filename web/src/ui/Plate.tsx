import { memo, useCallback, useLayoutEffect, useMemo, useState, type CSSProperties, type KeyboardEvent } from "react";
import { int, pct, signed } from "../lib/format";
import { classifyHead, headIndex, type HeadsData } from "../lib/heads";
import { activeNeurons, ordinalBack, peakOf, pushScale, pushTotal, rankOf, topPush } from "../lib/narrate";
import { DIV_CSS, RAMP_CSS, divIndex, rampIndex } from "../lib/palette";
import { glyphKind, spoken } from "../lib/text";
import type { Inspection } from "../model/inspect";
import type { Explain } from "../state/store";
import { HeatCanvas } from "./HeatCanvas";
import "../styles/plate.css";

interface Props {
  insp: Inspection | null;
  busy: boolean;
  phase: "loading" | "ready" | "error";
  chars: readonly string[];
  heads: HeadsData;
  idx: number;
  textLen: number;
  nextId: number | null;
  explain: Explain;
  onExplain: (m: Explain) => void;
  onSelect: (idx: number) => void;
}

function Glyph({ ch }: { ch: string }) {
  const k = glyphKind(ch);
  if (k === "space") return <i className="gl-space" aria-hidden="true" />;
  if (k === "newline") return <i className="gl-break" aria-hidden="true" />;
  return <>{ch}</>;
}

const quoted = (ch: string) => `“${spoken(ch)}”`;

export const Plate = memo(function Plate({ insp, busy, phase, chars, heads, idx, textLen, nextId, explain, onExplain, onSelect }: Props) {
  const [scoreEl, setScoreEl] = useState<HTMLDivElement | null>(null);
  const [cols, setCols] = useState(40);
  const [hover, setHover] = useState<{ col: number; text: string; x: number; y: number } | null>(null);

  useLayoutEffect(() => {
    const el = scoreEl;
    if (!el) return;
    const read = () => setCols(Math.max(14, Math.min(56, Math.floor(el.clientWidth / 12.5))));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [scoreEl]);

  const move = useCallback(
    (delta: number, absolute?: number) => onSelect(absolute !== undefined ? absolute : idx + delta),
    [idx, onSelect],
  );
  const onKey = (e: KeyboardEvent) => {
    const keys: Record<string, () => void> = {
      ArrowLeft: () => move(-1),
      ArrowRight: () => move(1),
      PageUp: () => move(-10),
      PageDown: () => move(10),
      Home: () => move(0, 0),
      End: () => move(0, textLen - 1),
    };
    const fn = keys[e.key];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };

  const data = useMemo(() => (insp ? derive(insp, cols) : null), [insp, cols]);

  if (!insp || !data) {
    return (
      <figure className="plate plate-empty" aria-busy={phase === "loading" || busy}>
        <p role="status">
          {phase === "error" ? "The model did not load, so there is nothing to open." : "The plate opens as soon as the model has read the first line."}
        </p>
        <div className="skeleton" aria-hidden="true" />
      </figure>
    );
  }

  const { n, layers } = insp;
  const target = insp.target;
  const targetCh = chars[target] ?? "";
  const pickCh = chars[insp.top] ?? "";
  const hasNext = nextId !== null;
  const scale = pushScale(insp);
  const total = pushTotal(insp);
  const finalStage = insp.lens[insp.lens.length - 1];
  const rank = rankOf(insp.probs, target);

  const setTip = (col: number, text: string, x: number, y: number) => setHover({ col, text, x, y });

  return (
    <figure className="plate" aria-busy={busy} aria-label="Anatomy of one prediction, from the letter to the next-letter probabilities">
      <div className="plate-tools">
        <div className="seg" role="group" aria-label="What to explain">
          <button type="button" aria-pressed={explain === "pick"} onClick={() => onExplain("pick")}>
            The model&apos;s pick
          </button>
          <button type="button" aria-pressed={explain === "actual"} disabled={!hasNext} onClick={() => onExplain("actual")}>
            The next letter in the poem
          </button>
        </div>
        <p className="ctx-note small">
          {n > cols ? `Last ${cols} of ${n} letters the model reads.` : `${n} ${n === 1 ? "letter" : "letters"} read.`} Click a letter or use the arrow keys.
        </p>
        <div className="stepper">
          <button type="button" className="btn ghost" onClick={() => move(-1)} disabled={idx <= 0}>
            Earlier letter
          </button>
          <button type="button" className="btn ghost" onClick={() => move(1)} disabled={idx >= textLen - 1}>
            Later letter
          </button>
        </div>
      </div>
      <div className="plate-head">
        <div className="ph-grid">
          <p className="ph-left small">The model is reading</p>
          <div className="ph-score" ref={setScoreEl}>
            <div
              className="ctx"
              role="slider"
              tabIndex={0}
              aria-label="Letter to inspect"
              aria-valuemin={0}
              aria-valuemax={Math.max(0, textLen - 1)}
              aria-valuenow={idx}
              aria-valuetext={`letter ${spoken(chars[insp.ids[n - 1]] ?? "")}, ${idx + 1} of ${textLen}`}
              onKeyDown={onKey}
              style={{ "--cols": cols } as CSSProperties}
            >
              {Array.from({ length: cols }, (_, c) => {
                const p = n - cols + c;
                if (p < 0) return <span key={c} className="cc none" aria-hidden="true" />;
                const ch = chars[insp.ids[p]] ?? "";
                const back = n - 1 - p;
                return (
                  <span
                    key={c}
                    className={`cc${back === 0 ? " now" : ""}${hover?.col === c ? " hot" : ""}`}
                    onClick={() => onSelect(idx - back)}
                    aria-hidden="true"
                  >
                    <Glyph ch={ch} />
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <Band
        className="b-embed"
        title="Letter and position"
        meta={`residual stream norm ${insp.norms[0].toFixed(1)}`}
        note={
          <>
            The letter {quoted(chars[insp.ids[n - 1]] ?? "")} and its place in the window ({n - 1}) are each looked up as {int(insp.embed.length)} numbers and added. Read back through the output table, that
            sum already says {quoted(chars[topOf(insp.lens[0])] ?? "")} {pct(insp.lens[0][topOf(insp.lens[0])])}: the input, copied.
          </>
        }
        lens={<Lens probs={insp.lens[0]} chars={chars} title="Guess so far" target={target} />}
      >
        <div className="embed-row">
          {[
            { v: data.embed.tok, label: `letter ${quoted(chars[insp.ids[n - 1]] ?? "")}` },
            { v: data.embed.pos, label: `position ${n - 1}` },
            { v: data.embed.sum, label: "together" },
          ].map((m, i) => (
            <div className="mat" key={i}>
              <HeatCanvas cols={24} rows={8} values={m.v} css={DIV_CSS} cellH={6} label={`${m.label}: ${insp.embed.length} numbers drawn as a grid`} />
              <span className="small">{m.label}</span>
              {i < 2 && <span className="op" aria-hidden="true">{i === 0 ? "+" : "="}</span>}
            </div>
          ))}
        </div>
      </Band>

      {Array.from({ length: layers }, (_, l) => {
        const tp = topPush(insp, l);
        const hid = (h: number) => `${l + 1}.${h + 1}`;
        const pk = peakOf(insp, l, tp.head);
        const roleTop = classifyHead(heads.heads[headIndex(l + 1, tp.head + 1)]);
        const stream = `${insp.norms[2 * l].toFixed(1)} to ${insp.norms[2 * l + 2].toFixed(1)}`;
        return (
          <Band
            key={l}
            className="b-block"
            title={`Block ${l + 1}`}
            meta={`residual stream norm ${stream}`}
            note={
              <>
                Head {hid(tp.head)} moves {quoted(targetCh)} the most in this block, {tp.push >= 0 ? "up" : "down"} by {Math.abs(tp.push).toFixed(2)}. It puts {pct(pk.p)} of its attention on {quoted(chars[insp.ids[pk.pos]] ?? "")}, {pk.back === 0 ? "the letter being inspected" : ordinalBack(pk.back)}. Across held-out text it is a
                “{roleTop.label.toLowerCase()}” head. {activeNeurons(insp, l)} of {int(insp.mlp.length / layers)} MLP neurons fire above 0.1.
              </>
            }
            lens={
              <>
                <Lens probs={insp.lens[1 + 2 * l]} chars={chars} title="Guess after attention" target={target} />
                <Lens probs={insp.lens[2 + 2 * l]} chars={chars} title="Guess after the MLP" target={target} />
              </>
            }
          >
            {Array.from({ length: insp.heads }, (_, h) => {
              const role = classifyHead(heads.heads[headIndex(l + 1, h + 1)]);
              const peak = data.peaks[l * insp.heads + h];
              return (
                <div className="row" key={h}>
                  <div className="hl" title={role.detail}>
                    <b className="num">{hid(h)}</b>
                    <span>{role.label}</span>
                  </div>
                  <div className="strip">
                    <HeatCanvas
                      cols={cols}
                      rows={1}
                      firstCol={Math.max(0, cols - n)}
                      values={data.strips[l * insp.heads + h]}
                      css={RAMP_CSS}
                      cellH={14}
                      marks={[{ col: peak.col, row: 0 }]}
                      label={`Head ${hid(h)} attention over the letters before the inspected one. Strongest: ${spoken(chars[insp.ids[peak.pos]] ?? "")}, ${ordinalBack(n - 1 - peak.pos)}, ${pct(peak.p)}.`}
                      onHover={(c) => {
                        if (!c) return setHover(null);
                        const p = n - cols + c.col;
                        if (p < 0) return setHover(null);
                        const v = insp.last[(l * insp.heads + h) * n + p];
                        setTip(c.col, `${hid(h)} reads ${quoted(chars[insp.ids[p]] ?? "")}, ${ordinalBack(n - 1 - p)}: ${pct(v, v < 0.1 ? 1 : 0)}`, c.x, c.y);
                      }}
                    />
                  </div>
                  <PushBar v={insp.attribution.heads[l * insp.heads + h]} scale={scale} />
                </div>
              );
            })}
            <div className="row mlp">
              <div className="hl">
                <b className="num">MLP</b>
                <span>{int(insp.mlp.length / layers)} neurons</span>
              </div>
              <div className="strip">
                <HeatCanvas
                  cols={96}
                  rows={8}
                  values={data.mlp[l]}
                  css={RAMP_CSS}
                  cellH={4}
                  label={`MLP of block ${l + 1}: ${int(insp.mlp.length / layers)} neuron activations, ${activeNeurons(insp, l)} above 0.1`}
                />
              </div>
              <PushBar v={insp.attribution.mlp[l]} scale={scale} />
            </div>
          </Band>
        );
      })}

      <Band
        className="b-out"
        title="Next letter"
        meta="final LayerNorm, then the embedding table again"
        note={
          <>
            {hasNext && explain === "actual" ? (
              <>
                The poem has {quoted(targetCh)} next. The model ranked it {rank} of {chars.length} at {pct(insp.probs[target], 1)}; its own pick was {quoted(pickCh)} at {pct(insp.probs[insp.top], 1)}.
              </>
            ) : (
              <>
                The model&apos;s pick is {quoted(pickCh)} at {pct(insp.probs[insp.top], 1)}.
              </>
            )}{" "}
            Every push on the logit of {quoted(targetCh)} adds up to {insp.logit.toFixed(3)}; the model computed {insp.logit.toFixed(3)}
            {Math.abs(total - insp.logit) < 1e-3 ? ", so nothing is missing." : "."}
          </>
        }
        lens={<Lens probs={finalStage} chars={chars} title="After block 4" target={target} rows={5} />}
      >
        <div className="out-grid">
          <ol className="dist" aria-label="Next-letter probabilities">
            {data.top.map((t) => (
              <li key={t.id} className={t.id === target ? "on" : undefined}>
                <span className="g">
                  <Glyph ch={chars[t.id] ?? ""} />
                </span>
                <span className="b" aria-hidden="true">
                  <span style={{ transform: `scaleX(${Math.max(t.p, 0.003)})` }} />
                </span>
                <span className="num">{pct(t.p, t.p < 0.1 ? 1 : 0)}</span>
              </li>
            ))}
          </ol>
          <dl className="sum" aria-label={`Contributions to the logit of ${spoken(targetCh)}`}>
            <div>
              <dt>Letter and position</dt>
              <dd className="num">{signed(insp.attribution.embed)}</dd>
            </div>
            <div>
              <dt>16 heads</dt>
              <dd className="num">{signed(insp.attribution.heads.reduce((a, b) => a + b, 0))}</dd>
            </div>
            <div>
              <dt>4 MLPs</dt>
              <dd className="num">{signed(insp.attribution.mlp.reduce((a, b) => a + b, 0))}</dd>
            </div>
            <div>
              <dt>Biases</dt>
              <dd className="num">{signed(insp.attribution.bias + insp.attribution.constant)}</dd>
            </div>
            <div className="tot">
              <dt>Logit of {quoted(targetCh)}</dt>
              <dd className="num">{insp.logit.toFixed(3)}</dd>
            </div>
          </dl>
        </div>
      </Band>

      {hover && (
        <div className="tip" style={{ left: hover.x, top: hover.y }} role="status">
          {hover.text}
        </div>
      )}
    </figure>
  );
});

function topOf(p: ArrayLike<number>): number {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}

interface Derived {
  embed: { tok: Uint8Array; pos: Uint8Array; sum: Uint8Array };
  strips: Uint8Array[];
  peaks: { col: number; pos: number; p: number }[];
  mlp: Uint8Array[];
  top: { id: number; p: number }[];
}

function derive(insp: Inspection, cols: number): Derived {
  const { n, layers, heads: H } = insp;
  const signedScale = Math.max(...Array.from(insp.tok, Math.abs), ...Array.from(insp.pos, Math.abs), ...Array.from(insp.embed, Math.abs)) || 1;
  const toDiv = (v: Float32Array) => Uint8Array.from(v, (x) => divIndex(x / signedScale));
  const strips: Uint8Array[] = [];
  const peaks: { col: number; pos: number; p: number }[] = [];
  for (let lh = 0; lh < layers * H; lh++) {
    const row = new Uint8Array(cols);
    let best = -1;
    let bestP = -1;
    for (let c = 0; c < cols; c++) {
      const p = n - cols + c;
      if (p < 0) continue;
      const v = insp.last[lh * n + p];
      row[c] = rampIndex(v);
      if (v > bestP) {
        bestP = v;
        best = c;
      }
    }
    strips.push(row);
    peaks.push({ col: best, pos: n - cols + best, p: bestP });
  }
  const width = insp.mlp.length / layers;
  const mlp: Uint8Array[] = [];
  for (let l = 0; l < layers; l++) {
    const a = insp.mlp.subarray(l * width, (l + 1) * width);
    let max = 0;
    for (const v of a) max = Math.max(max, v);
    mlp.push(Uint8Array.from(a, (v) => rampIndex(Math.max(0, v) / (max || 1))));
  }
  const top = Array.from(insp.probs.keys())
    .sort((a, b) => insp.probs[b] - insp.probs[a])
    .slice(0, 8)
    .map((id) => ({ id, p: insp.probs[id] }));
  return { embed: { tok: toDiv(insp.tok), pos: toDiv(insp.pos), sum: toDiv(insp.embed) }, strips, peaks, mlp, top };
}

function Band({ className, title, meta, note, lens, children }: { className: string; title: string; meta: string; note: React.ReactNode; lens: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className={`band ${className}`}>
      <div className="notes">
        <h3>{title}</h3>
        <p className="meta small">{meta}</p>
        <p className="note">{note}</p>
      </div>
      <div className="body">{children}</div>
      <div className="lens">{lens}</div>
    </section>
  );
}

function PushBar({ v, scale }: { v: number; scale: number }) {
  const w = Math.min(1, Math.abs(v) / scale);
  return (
    <div className="push" title={`Direct push on the explained letter's logit: ${signed(v)}`}>
      <span className="axis" aria-hidden="true">
        <span className={v < 0 ? "neg" : "pos"} style={{ transform: `scaleX(${w})` }} />
      </span>
      <span className="num">{signed(v)}</span>
    </div>
  );
}

function Lens({ probs, chars, title, target, rows = 3 }: { probs: Float32Array; chars: readonly string[]; title: string; target: number; rows?: number }) {
  const top = useMemo(
    () =>
      Array.from(probs.keys())
        .sort((a, b) => probs[b] - probs[a])
        .slice(0, rows),
    [probs, rows],
  );
  return (
    <div className="lens-list">
      <p className="small">{title}</p>
      <ol>
        {top.map((id) => (
          <li key={id} className={id === target ? "on" : undefined}>
            <span className="g">
              <Glyph ch={chars[id] ?? ""} />
            </span>
            <span className="b" aria-hidden="true">
              <span style={{ transform: `scaleX(${Math.max(probs[id], 0.003)})` }} />
            </span>
            <span className="num">{pct(probs[id], probs[id] < 0.1 ? 1 : 0)}</span>
            <span className="sr-only">{spoken(chars[id] ?? "")}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
