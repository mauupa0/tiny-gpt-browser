import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, type CSSProperties } from "react";
import parity from "../generated/parity.json";
import stats from "../generated/stats.json";
import { duration, int, mb, sci } from "../lib/format";
import { rankOf } from "../lib/narrate";
import { encodeText, spoken } from "../lib/text";
import { distribution, topN } from "../model/sample";
import { entropyBits } from "../model/stats";
import type { ModelState } from "../state/store";
import { Candidates } from "./Candidates";
import { Poem } from "./Poem";
import "../styles/hero.css";

export const PRESETS = ["Litwo! Ojczyzno moja!", "Tadeusz wszedł do pokoju", "Dzień dobry, panie Sędzio"];
const PACE_MS = 42;

export interface HeroSettings {
  temperature: number;
  topK: number;
  fast: boolean;
}

interface Props {
  s: ModelState;
  reduced: boolean;
  settings: HeroSettings;
  onSettings: (s: HeroSettings) => void;
  prompt: string;
  onPrompt: (p: string) => void;
  onWrite: (prompt: string | null) => void;
  onStop: () => void;
  // The poem has reached its length: keep this many characters and stop the worker.
  onFinish: (n: number) => void;
  onSelect: (i: number) => void;
  onFollow: () => void;
}

export const paceFor = (fast: boolean): number => (fast ? 0 : PACE_MS);

function Loading({ s }: { s: ModelState }) {
  const pct = Math.min(1, s.loaded / Math.max(1, s.total));
  if (s.phase === "error") {
    return (
      <p className="status err" role="alert">
        The model could not load: {s.error}. <button className="textbtn" onClick={() => location.reload()}>Reload the page</button>
      </p>
    );
  }
  if (s.phase === "loading") {
    return (
      <p className="status" role="status">
        <span className="meter" aria-hidden="true">
          <span style={{ transform: `scaleX(${pct})` }} />
        </span>
        Loading weights, {Math.round(pct * 100)}% of {mb(s.total > 1 ? s.total : stats.fp16Bytes)} MB
      </p>
    );
  }
  return (
    <p className="status" role="status">
      {mb(s.meta?.bytes ?? 0)} MB of weights loaded in {duration(s.meta?.loadMs ?? 0)}
    </p>
  );
}

export const Hero = memo(function Hero({ s, reduced, settings, onSettings, prompt, onPrompt, onWrite, onStop, onFinish, onSelect, onFollow }: Props) {
  const ready = s.phase === "ready";
  const run = s.runs.hero;
  const lane = run.lanes[0];
  const chars = s.meta?.chars ?? [];
  const wrote = lane.ids.length;
  const tps = lane.stats.computeMs > 0 ? (lane.stats.tokens / lane.stats.computeMs) * 1000 : 0;
  const started = useRef(false);
  const root = useRef<HTMLElement>(null);
  const armed = useRef(false);
  const write = useCallback(
    (p: string | null) => {
      // Only a fresh poem ends by itself on the page's rule; "Keep writing" is the visitor asking for more.
      armed.current = p !== null;
      onWrite(p);
    },
    [onWrite],
  );
  const typed = useMemo(() => (chars.length ? encodeText(chars, prompt) : { ids: [], dropped: 0 }), [chars, prompt]);
  const unreadable = chars.length > 0 && typed.ids.length === 0;

  useEffect(() => {
    if (!ready || started.current) return;
    started.current = true;
    if (!reduced) write(prompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // A fresh poem ends on the first line break once it fills --fill rows of the --rows the layout
  // reserved (hero.css). A verse that would spill past --rows is cut back to the previous break.
  useLayoutEffect(() => {
    if (!armed.current || !run.running || s.text.length <= s.promptLen) return;
    const poem = root.current?.querySelector<HTMLElement>(".poem-hero");
    if (!poem) return;
    const cs = getComputedStyle(poem);
    const target = Number(cs.getPropertyValue("--rows"));
    const fill = Number(cs.getPropertyValue("--fill")) || target;
    const lh = parseFloat(cs.lineHeight);
    if (!target || !lh) return;
    const top = poem.getBoundingClientRect().top;
    const rowsTo = (el: Element | null) => (el ? (el.getBoundingClientRect().bottom - top) / lh : 0);
    const done = (n: number) => {
      armed.current = false;
      onFinish(n);
    };
    let prev = -1;
    for (const b of poem.querySelectorAll<HTMLElement>(".c.nl")) {
      const i = Number(b.dataset.i);
      if (i < s.promptLen) continue;
      // The line break sits on the last row of its verse, so that verse's bottom is the row count so far.
      const rows = rowsTo(b.parentElement);
      if (rows > target + 0.05) return done(prev >= 0 ? prev + 1 : i + 1);
      if (rows >= fill - 0.05) return done(i + 1);
      prev = i;
    }
    const verses = poem.querySelectorAll(".l");
    if (prev >= 0 && rowsTo(verses[verses.length - 1] ?? null) > target + 0.05) done(prev + 1);
  }, [s.text.length, s.promptLen, run.running, onFinish]);

  const idle = useMemo(() => {
    if (!s.inspection || run.running) return null;
    const p = distribution(s.inspection.logits, settings.temperature, settings.topK);
    return { top: topN(p, 8), entropy: entropyBits(p) };
  }, [s.inspection, run.running, settings.temperature, settings.topK]);

  const probs = run.running ? lane.top : idle ? idle.top : lane.top;
  // The figure is a moment behind the selection while the worker answers; say so instead of mislabelling.
  const insp = s.inspection;
  const stale = !run.running && (!insp || insp.ids[insp.n - 1] !== s.text[s.inspectIdx] || insp.n !== Math.min(s.inspectIdx + 1, s.meta?.ctx ?? 128));
  const entropy = run.running ? lane.entropy[lane.entropy.length - 1] : idle?.entropy;
  const subject = s.text[s.inspectIdx] !== undefined ? spoken(chars[s.text[s.inspectIdx]] ?? "") : "";
  const nextId = s.text[s.inspectIdx + 1];
  const nextCh = nextId === undefined ? "" : spoken(chars[nextId] ?? "");
  let nextNote = "";
  if (!run.running && !stale && insp && nextId !== undefined) {
    nextNote = probs.some((p) => p.id === nextId)
      ? " The orange one is the letter the poem has next."
      : ` The poem has “${nextCh}” next, which the model ranked ${rankOf(insp.probs, nextId)} of ${chars.length}.`;
  }
  const label = run.running
    ? "Candidates the model weighed for the letter it just wrote"
    : `Candidates for the letter after the selected one, “${subject}”`;

  return (
    <header className="hero wrap" id="writing" ref={root}>
      <div className="hero-side">
        <h1>Tiny GPT</h1>
        <p className="lead">
          A transformer with <b>{int(stats.params)}</b> parameters, trained from scratch on <i>Pan Tadeusz</i>. It writes the verse beside this text one letter at a time, in your browser, with
          inference code I wrote without a machine learning library.
        </p>
        <p className="lead">
          Its logits match PyTorch to within <span className="num">{sci(parity.float16.worstLogitDiff)}</span>. Click any letter of the poem to see what the model did to produce it.
        </p>
        <Loading s={s} />
        <a className="cue" href="#inside">
          Look inside the model
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 2.5v10M3.5 8.5 8 13l4.5-4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </a>
      </div>

      <div className="hero-main">
        {s.text.length > 0 ? (
          <Poem
            chars={chars}
            ids={s.text}
            promptLen={s.promptLen}
            ps={s.heroPs}
            selected={s.inspectIdx}
            running={run.running}
            onSelect={onSelect}
            label="Verse written by the model"
            size="hero"
          />
        ) : (
          <div className="poem poem-hero" role="region" aria-label="Verse written by the model">
            <span className="l">
              <span className="c prompt">{prompt}</span>
            </span>
          </div>
        )}

        <p className="legend small">
          <i className="swatch" aria-hidden="true" /> Underlined letters were long shots: the model picked them with under 25% probability.
        </p>
        <div className="cand-row">
          <Candidates probs={probs} chars={chars} chosen={run.running ? lane.chosen : stale ? null : (s.text[s.inspectIdx + 1] ?? null)} label={label} stale={stale} />
          <p className="cand-note small" aria-live="off">
            {run.running ? "The model weighs these for the next letter" : subject ? `After “${subject}” the model weighs these` : "The model weighs these for the next letter"}
            {entropy !== undefined && Number.isFinite(entropy) ? (
              <>
                , with <span className="num">{entropy.toFixed(1)}</span> bits of uncertainty
              </>
            ) : null}
            .{nextNote}
            {!s.followTail && !run.running && s.text.length > 0 && (
              <>
                {" "}
                <button className="textbtn" onClick={onFollow}>
                  Select the last letter
                </button>
              </>
            )}
          </p>
        </div>

        <div className="hero-actions">
          {run.running ? (
            <button className="btn" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button className="btn" onClick={() => write(prompt)} disabled={!ready || unreadable}>
              {wrote > 0 ? "Write again" : "Write"}
            </button>
          )}
          <button className="btn ghost" onClick={() => write(null)} disabled={!ready || run.running || wrote === 0}>
            Keep writing
          </button>
          <p className="tps" aria-live="off">
            <span className="num">{tps ? int(tps) : "-"}</span> characters per second
            <span className="small"> (model time only, on this device)</span>
          </p>
        </div>

        <div className="hero-controls">
          <div className="field">
            <label htmlFor="prompt">Opening line</label>
            <input
              id="prompt"
              type="text"
              value={prompt}
              maxLength={160}
              spellCheck={false}
              autoComplete="off"
              disabled={run.running}
              onChange={(e) => onPrompt(e.target.value)}
            />
            <p className="presets">
              {PRESETS.map((p) => (
                <button key={p} type="button" className="textbtn" disabled={run.running} onClick={() => onPrompt(p)}>
                  {p}
                </button>
              ))}
            </p>
            {typed.dropped > 0 && (
              <p className="small" role="status">
                {unreadable
                  ? `None of these characters is in the model's ${chars.length}-letter alphabet, so there is nothing to continue.`
                  : `${typed.dropped} ${typed.dropped === 1 ? "character is" : "characters are"} outside the model's ${chars.length}-letter alphabet and will be skipped.`}
              </p>
            )}
          </div>
          <Range
            id="temp"
            label="Temperature"
            value={settings.temperature}
            min={0}
            max={1.5}
            step={0.05}
            format={(v) => (v === 0 ? "0, always the top letter" : v.toFixed(2))}
            ends={["steady", "adventurous"]}
            onChange={(v) => onSettings({ ...settings, temperature: v })}
          />
          <Range
            id="topk"
            label="Top-k"
            value={settings.topK}
            min={1}
            max={chars.length || 71}
            step={1}
            format={(v) => (v >= (chars.length || 71) ? "all letters" : `${v} best letters`)}
            ends={["1", "all"]}
            onChange={(v) => onSettings({ ...settings, topK: v })}
          />
          <label className="switch">
            <input type="checkbox" checked={settings.fast} onChange={(e) => onSettings({ ...settings, fast: e.target.checked })} />
            <i aria-hidden="true" />
            Full speed, no reading pace
          </label>
        </div>
      </div>
    </header>
  );
});

interface RangeProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  ends: [string, string];
  onChange: (v: number) => void;
}

function Range({ id, label, value, min, max, step, format, ends, onChange }: RangeProps) {
  const [a, b] = ends;
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="range">
      <label htmlFor={id}>
        {label}
        <output htmlFor={id}>{format(value)}</output>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} style={{ "--pct": `${pct}%` } as CSSProperties} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="ends" aria-hidden="true">
        <span>{a}</span>
        <span>{b}</span>
      </span>
    </div>
  );
}
