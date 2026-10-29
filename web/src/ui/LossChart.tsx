import { memo, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent as RPE } from "react";
import { int } from "../lib/format";

export interface CurvePoint {
  step: number;
  train: number;
  val: number;
  sec: number;
}

interface Props {
  curve: CurvePoint[];
  vocab: number;
}

const PAD = { l: 42, r: 58, t: 14, b: 30 };
const H = 280;
const Y_MIN = 1.2;
const Y_MAX = 4.6;

export const LossChart = memo(function LossChart({ curve, vocab }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(640);
  const [i, setI] = useState(curve.length - 1);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const read = () => setW(Math.max(280, Math.floor(el.clientWidth)));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const maxStep = curve[curve.length - 1].step;
  const x = (step: number) => PAD.l + (step / maxStep) * (w - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD.t - PAD.b);
  const line = (key: "train" | "val") => curve.map((p) => `${x(p.step).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  const random = Math.log(vocab);
  const p = curve[i];
  const ticksY = [2, 3, 4];
  const ticksX = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * maxStep));

  const pick = (e: RPE<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const step = ((e.clientX - rect.left - PAD.l) / (rect.width - PAD.l - PAD.r)) * maxStep;
    let best = 0;
    curve.forEach((c, k) => {
      if (Math.abs(c.step - step) < Math.abs(curve[best].step - step)) best = k;
    });
    setI(best);
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft") setI((v) => Math.max(0, v - 1));
    else if (e.key === "ArrowRight") setI((v) => Math.min(curve.length - 1, v + 1));
    else if (e.key === "Home") setI(0);
    else if (e.key === "End") setI(curve.length - 1);
    else return;
    e.preventDefault();
  };

  return (
    <figure className="loss">
      <p className="loss-read" aria-live="polite">
        <span className="num">step {int(p.step)}</span> <span className="small">after {Math.round(p.sec / 60)} min of CPU time</span>
        <span className="lv">
          <i className="sw val" /> validation <b className="num">{p.val.toFixed(3)}</b>
        </span>
        <span className="lv">
          <i className="sw trn" /> training <b className="num">{p.train.toFixed(3)}</b>
        </span>
      </p>
      <div ref={box} className="loss-box">
        <svg
          width={w}
          height={H}
          viewBox={`0 0 ${w} ${H}`}
          role="slider"
          tabIndex={0}
          aria-label="Training curve, step"
          aria-valuemin={0}
          aria-valuemax={maxStep}
          aria-valuenow={p.step}
          aria-valuetext={`step ${p.step}, validation loss ${p.val}, training loss ${p.train}`}
          onPointerMove={pick}
          onPointerDown={pick}
          onKeyDown={key}
        >
          {ticksY.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={w - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
              <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" className="ax">
                {t}
              </text>
            </g>
          ))}
          <line x1={PAD.l} x2={w - PAD.r} y1={y(random)} y2={y(random)} stroke="var(--ink-3)" strokeDasharray="4 4" />
          <text x={w - PAD.r} y={y(random) - 6} textAnchor="end" className="ax">
            guessing at random: ln {vocab} = {random.toFixed(2)}
          </text>
          {ticksX.map((t) => (
            <text key={t} x={x(t)} y={H - 8} textAnchor={t === 0 ? "start" : t === maxStep ? "end" : "middle"} className="ax">
              {int(t)}
            </text>
          ))}
          <polyline points={line("train")} fill="none" stroke="var(--ramp-3)" strokeWidth="2" strokeLinejoin="round" />
          <polyline points={line("val")} fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinejoin="round" />
          <text x={x(maxStep) + 8} y={y(curve[curve.length - 1].val) + 4} className="ax end val">
            {curve[curve.length - 1].val.toFixed(2)}
          </text>
          <text x={x(maxStep) + 8} y={y(curve[curve.length - 1].train) + 4} className="ax end">
            {curve[curve.length - 1].train.toFixed(2)}
          </text>
          <line x1={x(p.step)} x2={x(p.step)} y1={PAD.t} y2={H - PAD.b} stroke="var(--marker)" strokeWidth="1" />
          <circle cx={x(p.step)} cy={y(p.val)} r="4.5" fill="var(--ink)" stroke="var(--marker)" strokeWidth="2" />
          <circle cx={x(p.step)} cy={y(p.train)} r="4.5" fill="var(--ramp-3)" stroke="var(--marker)" strokeWidth="2" />
        </svg>
      </div>
      <figcaption className="small">Loss is the average surprise per letter, in nats. Hover, tap or use the arrow keys to read any checkpoint.</figcaption>
    </figure>
  );
});
