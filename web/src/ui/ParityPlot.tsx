import { memo, type CSSProperties } from "react";
import parity from "../generated/parity.json";
import { sci } from "../lib/format";
import { logPos } from "../lib/logaxis";
import { PARITY_LIMIT } from "../model/parity";
import type { Fault } from "../model/types";
import type { ParityOut } from "../state/store";

interface Props {
  live: Partial<Record<Fault, ParityOut>>;
  busy: boolean;
  onRun: () => void;
}

interface Row {
  key: string;
  label: string;
  value: number | null;
  source: "test" | "live" | "none";
}

const TICKS = [1e-6, 1e-4, 1e-2, 1];

export const ParityPlot = memo(function ParityPlot({ live, busy, onRun }: Props) {
  const pick = (f: Fault, fallback: number | null): { value: number | null; source: Row["source"] } =>
    live[f] ? { value: live[f]!.max, source: "live" } : fallback !== null ? { value: fallback, source: "test" } : { value: null, source: "none" };
  const rows: Row[] = [
    { key: "f32", label: "fp32 weights", value: parity.float32.worstLogitDiff, source: "test" },
    { key: "f16-test", label: "fp16 weights, test run", value: parity.float16.worstLogitDiff, source: "test" },
    { key: "none", label: "This page, intact", ...pick("none", null) },
    { key: "scale", label: "Without the 1/√d factor", ...pick("scale", parity.fault_scale.worstLogitDiff) },
    { key: "mask", label: "Attention sees the future", ...pick("mask", parity.fault_mask.worstLogitDiff) },
  ];
  const limit = logPos(PARITY_LIMIT) * 100;
  return (
    <figure className="parity">
      <figcaption>
        <b>Distance from PyTorch</b>, largest difference between any logit of the two implementations over one verse line (50 positions, 71 logits each). Log scale.
      </figcaption>
      <div className="pp" style={{ "--limit": `${limit}%` } as CSSProperties}>
        <div className="pp-axis" aria-hidden="true">
          {TICKS.map((t) => (
            <span key={t} style={{ left: `${logPos(t) * 100}%` }} className="num">
              {t.toExponential(0).replace("e+0", "e0")}
            </span>
          ))}
          <span className="lim num" style={{ left: `${limit}%` }}>
            limit 1e-3
          </span>
        </div>
        <ul>
          {rows.map((r) => {
            const fail = r.value !== null && r.value >= PARITY_LIMIT;
            return (
              <li key={r.key} className={fail ? "fail" : r.value === null ? "none" : "ok"}>
                <span className="pl">
                  {r.label}
                  <span className="small"> {r.source === "live" ? "measured now" : r.source === "test" ? "from the test suite" : "not measured yet"}</span>
                </span>
                <span className="track" aria-hidden="true">
                  {r.value !== null && <i className="dot" style={{ left: `${logPos(r.value) * 100}%` }} />}
                </span>
                <span className="pv num">{r.value === null ? "-" : sci(r.value)}</span>
                <span className="verdict small">{r.value === null ? "" : fail ? "fails the limit" : "within the limit"}</span>
              </li>
            );
          })}
        </ul>
      </div>
      <button className="btn ghost" onClick={onRun} disabled={busy}>
        {busy ? "Running the check" : "Run the check on this page now"}
      </button>
    </figure>
  );
});
