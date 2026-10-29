import { memo } from "react";
import bench from "../generated/bench.json";
import parity from "../generated/parity.json";
import stats from "../generated/stats.json";
import tests from "../generated/tests.json";
import { int, mb, pct, sci } from "../lib/format";
import { paramLedger } from "../lib/params";
import type { Bench } from "../state/store";
import { LossChart } from "./LossChart";
import "../styles/sections.css";

interface Props {
  ready: boolean;
  bench: Bench | null;
  benchBusy: boolean;
  onBench: () => void;
}

const CASES: { key: "single" | "short" | "verse" | "full_ctx"; label: string }[] = [
  { key: "single", label: "One letter" },
  { key: "short", label: "A short prompt" },
  { key: "verse", label: "One verse line" },
  { key: "full_ctx", label: "A full 128-letter window" },
];

const CHECKS: { file: string; what: string }[] = [
  { file: "parity.test.ts", what: "TypeScript against PyTorch logits at every position of four inputs, and every attention map of one verse line, for fp32 and shipped fp16 weights. Two deliberately broken variants must fail the same check." },
  { file: "inspect.test.ts", what: "The traced step equals the plain step. The logit lens at the last stage is the real prediction. Every part adds up to the residual stream and to the logit. A reused cache gives what a fresh one gives, and a cache that lies is caught." },
  { file: "lanes.test.ts", what: "Writers with the same settings write the same text, another seed or a broken model does not, a cold writer repeats itself more than a hot one." },
  { file: "lib.test.ts", what: "CSS colours equal canvas colours and text contrast clears 4.5, head roles follow from the measured profiles, the parameter count equals the one PyTorch reported, captions agree with the raw arrays, and the fault losses quoted on the page are re-measured." },
  { file: "units.test.ts", what: "fp16 decoding, top-k and temperature, seeded sampling, the sliding context window, attention rows as probability distributions." },
];

export const Proof = memo(function Proof({ ready, bench: live, benchBusy, onBench }: Props) {
  const ledger = paramLedger({ vocab: stats.vocab, ctx: stats.ctx, d: 192, layers: 4 });
  const parts = [
    { key: "MLP", v: ledger.mlp, c: "var(--ramp-5)" },
    { key: "Attention", v: ledger.attention, c: "var(--ramp-4)" },
    { key: "Position table", v: ledger.positionEmbedding, c: "var(--ramp-3)" },
    { key: "Letter table, also the output layer", v: ledger.tokenEmbedding, c: "var(--ramp-2)" },
    { key: "LayerNorm", v: ledger.layerNorm, c: "var(--ramp-1)" },
  ];
  const minutes = Math.round(stats.seconds / 60);
  const ppl = Math.exp(stats.bestVal);
  const widening = stats.curve.every((c, i, a) => i === 0 || c.val - c.train >= a[i - 1].val - a[i - 1].train);

  return (
    <div className="proof">
      <section className="pblock">
        <h3>Training</h3>
        <div className="pgrid">
          <div className="ptext">
            <p>
              {int(stats.steps)} steps of {stats.batch} windows of {stats.ctx} letters, AdamW with a cosine schedule, dropout 0.2, on the CPU of a Ryzen 5 7600X in {minutes} minutes. PyTorch is used for training and for nothing else.
            </p>
            <p>
              At step 0 the loss is {stats.curve[0].val.toFixed(2)}, which is guessing at random over {stats.vocab} letters (ln {stats.vocab} = {Math.log(stats.vocab).toFixed(2)}). It ends at {stats.bestVal.toFixed(3)} on text it never trained on, as unsure as
              choosing between {ppl.toFixed(1)} equally likely letters.
            </p>
            <p>
              Training loss ends at {stats.bestTrain.toFixed(3)}, so the model has started to memorise: the gap between the two lines is {(stats.bestVal - stats.bestTrain).toFixed(2)} nats{widening ? " and it widens at every checkpoint" : ""}.
            </p>
          </div>
          <LossChart curve={stats.curve} vocab={stats.vocab} />
        </div>
      </section>

      <section className="pblock">
        <h3>Where {int(ledger.total)} parameters sit</h3>
        <div className="ledger" role="img" aria-label={parts.map((p) => `${p.key} ${pct(p.v / ledger.total, 1)}`).join(", ")}>
          {parts.map((p) => (
            <i key={p.key} style={{ flexGrow: p.v, background: p.c }} />
          ))}
        </div>
        <ul className="ledger-list">
          {parts.map((p) => (
            <li key={p.key}>
              <i style={{ background: p.c }} aria-hidden="true" />
              <span>{p.key}</span>
              <span className="num">{int(p.v)}</span>
              <span className="num dim">{pct(p.v / ledger.total, 1)}</span>
            </li>
          ))}
        </ul>
        <p className="small">Counted from the shape of the network: width 192, 4 blocks, 4 heads, 71 letters, 128 positions. The sum is checked against PyTorch&apos;s own count in the tests.</p>
      </section>

      <section className="pblock">
        <h3>
          <span className="num">
            {tests.passed} of {tests.total}
          </span>{" "}
          tests pass
        </h3>
        <dl className="checks">
          {CHECKS.map((c) => {
            const f = tests.files.find((x) => x.file === c.file);
            return (
              <div key={c.file}>
                <dt>
                  <span className="num">{c.file}</span> <span className="small">{f ? `${f.passed} tests` : ""}</span>
                </dt>
                <dd>{c.what}</dd>
              </div>
            );
          })}
        </dl>
        <div className="table-wrap">
          <table>
            <caption>Largest absolute logit difference to PyTorch, limit 1e-3</caption>
            <thead>
              <tr>
                <th scope="col">Input</th>
                <th scope="col" className="r">
                  Letters
                </th>
                <th scope="col" className="r">
                  fp32
                </th>
                <th scope="col" className="r">
                  fp16 shipped
                </th>
              </tr>
            </thead>
            <tbody>
              {CASES.map(({ key, label }) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  <td className="num r">{parity.float32.cases[key].tokens}</td>
                  <td className="num r">{sci(parity.float32.cases[key].maxLogit)}</td>
                  <td className="num r">{sci(parity.float16.cases[key].maxLogit)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="pblock">
        <h3>Speed</h3>
        <div className="speed">
          <button className="btn" disabled={!ready || benchBusy} onClick={onBench}>
            {benchBusy ? "Measuring" : "Measure on this device"}
          </button>
          <p aria-live="polite">
            {live ? (
              <>
                <b className="num big">{int(live.tokPerSec)}</b> letters per second on this device ({live.tokens} letters in {Math.round(live.ms)} ms, model and sampling, no reading pace).
              </>
            ) : (
              <span className="small">Writes {400} letters at full speed inside the worker and times it.</span>
            )}
          </p>
        </div>
        <p className="small">
          Reference run: {bench.tokPerSec} letters per second, {bench.device}. The weights are {mb(stats.fp16Bytes)} MB as fp16 and the model reads one position at a time from a key-value cache, so a new letter costs one pass instead of a pass over the
          whole window.
        </p>
      </section>
    </div>
  );
});
