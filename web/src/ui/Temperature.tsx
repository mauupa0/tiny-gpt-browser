import { memo, useEffect, useRef } from "react";
import { repeatShare } from "../lib/text";
import { pct } from "../lib/format";
import type { LaneSpec } from "../worker/protocol";
import type { ModelState } from "../state/store";
import { useSeen } from "../state/inview";
import { Poem } from "./Poem";
import "../styles/sections.css";

export const TEMPS = [0.3, 0.8, 1.4];

interface Props {
  s: ModelState;
  topK: number;
  paceMs: number;
  reduced: boolean;
  onWrite: (lanes: LaneSpec[]) => void;
  onStop: () => void;
}

function Spark({ values, max = 6 }: { values: number[]; max?: number }) {
  const w = 240;
  const h = 40;
  const tail = values.slice(-80);
  const pts = tail.map((v, i) => `${(i / Math.max(1, 79)) * w},${h - Math.min(1, v / max) * (h - 4) - 2}`).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" y1={h - 1} x2={w} y2={h - 1} stroke="var(--line-2)" />
      {tail.length > 1 && <polyline points={pts} fill="none" stroke="var(--ramp-4)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

export const Temperature = memo(function Temperature({ s, topK, paceMs, reduced, onWrite, onStop }: Props) {
  const run = s.runs.temp;
  const chars = s.meta?.chars ?? [];
  const ref = useRef<HTMLDivElement>(null);
  const seen = useSeen(ref);
  const started = useRef(false);
  const ready = s.phase === "ready";
  const heroBusy = s.runs.hero.running;

  const lanes: LaneSpec[] = TEMPS.map((temperature) => ({ temperature, fault: "none", mute: [] }));
  useEffect(() => {
    if (!seen || !ready || started.current || reduced || heroBusy || s.promptLen === 0) return;
    started.current = true;
    onWrite(lanes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seen, ready, heroBusy, reduced, s.promptLen]);

  const prompt = s.text.slice(0, s.promptLen);
  const written = run.lanes.some((l) => l.ids.length > 0);

  return (
    <div ref={ref} className="temp">
      <div className="temp-bar">
        {run.running ? (
          <button className="btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="btn" disabled={!ready || s.promptLen === 0} onClick={() => onWrite(lanes)}>
            {written ? "Write three again" : "Write three"}
          </button>
        )}
        <p className="small">
          Same opening line, same dice, top-k {topK >= chars.length ? "off" : topK}. Only the temperature differs.
        </p>
      </div>
      <div className="temp-cols">
        {TEMPS.map((t, i) => {
          const lane = run.lanes[i];
          const ids = lane ? [...prompt, ...lane.ids] : prompt;
          const ps = lane ? [...prompt.map(() => 1), ...lane.ps] : [];
          const mean = lane && lane.entropy.length ? lane.entropy.reduce((a, b) => a + b, 0) / lane.entropy.length : null;
          const rep = lane && lane.ids.length >= 20 ? repeatShare(lane.ids, 4) : null;
          return (
            <section className="tcol" key={t} aria-label={`Temperature ${t}`}>
              <h3>
                <span className="num">T = {t.toFixed(1)}</span>
              </h3>
              {chars.length > 0 && ids.length > 0 ? (
                <Poem chars={chars} ids={ids} promptLen={prompt.length} ps={ps} selected={-1} running={run.running} label={`Verse written at temperature ${t}`} size="col" />
              ) : (
                <div className="poem poem-col empty-poem" />
              )}
              <Spark values={lane?.entropy ?? []} />
              <dl className="tstats">
                <div>
                  <dt>Uncertainty per letter</dt>
                  <dd className="num">{mean === null ? "-" : `${mean.toFixed(2)} bits`}</dd>
                </div>
                <div>
                  <dt>Repeated 4-letter runs</dt>
                  <dd className="num">{rep === null ? "-" : pct(rep)}</dd>
                </div>
              </dl>
            </section>
          );
        })}
      </div>
      <p className="small temp-note">The line under each poem is the model&apos;s uncertainty, in bits, for each letter it chose. The pace of writing is {paceMs === 0 ? "full speed" : "slowed to reading speed"}.</p>
    </div>
  );
});
