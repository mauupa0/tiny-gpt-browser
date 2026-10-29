import { memo, useEffect, useRef, useState } from "react";
import faults from "../generated/faults.json";
import headsData from "../generated/heads.json";
import { repeatShare } from "../lib/text";
import { pct } from "../lib/format";
import { RAMP_CSS, rampIndex } from "../lib/palette";
import type { Fault } from "../model/types";
import { useSeen, useVisible } from "../state/inview";
import type { ModelState } from "../state/store";
import type { LaneSpec } from "../worker/protocol";
import { ParityPlot } from "./ParityPlot";
import { Poem } from "./Poem";
import "../styles/sections.css";

export type Break = "scale" | "mask" | "mute";

interface Props {
  s: ModelState;
  temperature: number;
  reduced: boolean;
  onWrite: (lanes: LaneSpec[]) => void;
  onStop: () => void;
  onAblation: (on: boolean) => void;
  onParity: (f: Fault) => void;
}

const OPTIONS: { key: Break; label: string; blurb: string }[] = [
  { key: "scale", label: "Drop the 1/√d factor", blurb: "Attention scores are not divided by the square root of the head size. Scores grow, softmax saturates, heads stop mixing." },
  { key: "mask", label: "Let attention see the future", blurb: "A query may also read the slot after it. The causal mask is what stops a language model from peeking." },
  { key: "mute", label: "Silence one head", blurb: "Not a bug, an operation: one head's output is set to zero everywhere. The grid shows how much that head matters at the letter inspected above." },
];

export const BreakIt = memo(function BreakIt({ s, temperature, reduced, onWrite, onStop, onAblation, onParity }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const visible = useVisible(ref, "200px");
  const [kind, setKind] = useState<Break>("scale");
  const [head, setHead] = useState(7);
  const [asked, setAsked] = useState(false);
  const run = s.runs.brk;
  const ready = s.phase === "ready";
  const chars = s.meta?.chars ?? [];
  const heads = s.meta?.heads ?? 4;

  useEffect(() => {
    onAblation(visible && kind === "mute");
  }, [visible, kind, onAblation]);

  const seen = useSeen(ref);
  const started = useRef(false);
  const heroBusy = s.runs.hero.running;
  useEffect(() => {
    if (!seen || !ready || started.current || heroBusy || s.promptLen === 0) return;
    started.current = true;
    setAsked(true);
    (["none", "scale", "mask"] as Fault[]).forEach(onParity);
    if (!reduced) onWrite([{ temperature, fault: "none", mute: [] }, { temperature, fault: "scale", mute: [] }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seen, ready, heroBusy, s.promptLen]);

  const broken: LaneSpec =
    kind === "mute" ? { temperature, fault: "none", mute: [head] } : { temperature, fault: kind, mute: [] };
  const lanes: LaneSpec[] = [{ temperature, fault: "none", mute: [] }, broken];
  const prompt = s.text.slice(0, s.promptLen);
  const written = run.lanes.some((l) => l.ids.length > 0);

  const live = s.parity;
  const missing = (["none", "scale", "mask"] as Fault[]).some((f) => !live[f]);
  const busy = asked && missing && !s.error;
  const runParity = () => {
    setAsked(true);
    (["none", "scale", "mask"] as Fault[]).forEach(onParity);
  };

  const brokenLoss = kind === "scale" ? faults.scale : kind === "mask" ? faults.mask : faults.none + headsData.heads[head].dLoss;
  const losses = [faults.none, brokenLoss];
  const kl = s.inspection?.ablation?.kl;
  const klMax = kl ? Math.max(...Array.from(kl), 1e-6) : 1;

  return (
    <div ref={ref} className="brk">
      <div className="brk-bar">
        <div className="seg" role="radiogroup" aria-label="What to break">
          {OPTIONS.map((o) => (
            <button key={o.key} role="radio" aria-checked={kind === o.key} onClick={() => setKind(o.key)}>
              {o.label}
            </button>
          ))}
        </div>
        {run.running ? (
          <button className="btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="btn" disabled={!ready || s.promptLen === 0} onClick={() => onWrite(lanes)}>
            {written ? "Write both again" : "Write both"}
          </button>
        )}
      </div>
      <p className="lead brk-blurb">{OPTIONS.find((o) => o.key === kind)?.blurb}</p>

      {kind === "mute" && (
        <div className="muter">
          <p className="small">Pick the head to silence. Colour is how far the prediction at the inspected letter moves when only that head is zeroed there (KL divergence, nats).</p>
          <div className="mgrid" role="radiogroup" aria-label="Head to silence">
            {Array.from({ length: 16 }, (_, i) => {
              const v = kl ? kl[i] : null;
              return (
                <button
                  key={i}
                  role="radio"
                  aria-checked={head === i}
                  className={head === i ? "on" : undefined}
                  onClick={() => setHead(i)}
                  style={v !== null ? { background: RAMP_CSS[rampIndex(Math.sqrt(v / klMax))] } : undefined}
                  title={v !== null ? `Head ${Math.floor(i / heads) + 1}.${(i % heads) + 1}: ${v.toFixed(3)} nats` : undefined}
                >
                  <span className="num">
                    {Math.floor(i / heads) + 1}.{(i % heads) + 1}
                  </span>
                  <span className="num sm">{v !== null ? v.toFixed(2) : "-"}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="brk-cols">
        {[
          { name: "Intact model", note: "same dice, same opening, same temperature" },
          { name: OPTIONS.find((o) => o.key === kind)!.label, note: kind === "mute" ? `head ${Math.floor(head / heads) + 1}.${(head % heads) + 1} zeroed` : "broken on purpose" },
        ].map((c, i) => {
          const lane = run.lanes[i];
          const ids = lane ? [...prompt, ...lane.ids] : prompt;
          const ps = lane ? [...prompt.map(() => 1), ...lane.ps] : [];
          const mean = lane && lane.entropy.length ? lane.entropy.reduce((a, b) => a + b, 0) / lane.entropy.length : null;
          const rep = lane && lane.ids.length >= 20 ? repeatShare(lane.ids, 4) : null;
          return (
            <section className="tcol" key={i} aria-label={c.name}>
              <h3>{c.name}</h3>
              <p className="small">{c.note}</p>
              {chars.length > 0 && ids.length > 0 ? (
                <Poem chars={chars} ids={ids} promptLen={prompt.length} ps={ps} selected={-1} running={run.running} label={`Verse written by: ${c.name}`} size="col" />
              ) : (
                <div className="poem poem-col empty-poem" />
              )}
              <dl className="tstats">
                <div>
                  <dt>Loss on held-out text, {faults.windows} windows</dt>
                  <dd className="num">
                    {losses[i].toFixed(3)} nats{i === 1 ? ` (${losses[1] >= losses[0] ? "+" : "-"}${Math.abs(losses[1] - losses[0]).toFixed(3)})` : ""}
                  </dd>
                </div>
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
      {kind === "mute" && kl && <p className="small">Silenced head {Math.floor(head / heads) + 1}.{(head % heads) + 1} moves the next-letter distribution by {kl[head].toFixed(3)} nats at the inspected letter.</p>}

      {asked && missing && s.error && (
        <p className="small" role="alert">
          The check could not run: {s.error}
        </p>
      )}
      <ParityPlot live={live} busy={busy} onRun={runParity} />
    </div>
  );
});
