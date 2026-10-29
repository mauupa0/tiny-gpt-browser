/// <reference lib="webworker" />
import { GPT } from "../model/gpt";
import { Inspector } from "../model/inspect";
import { Lanes, type LaneSpec } from "../model/lanes";
import { parityCheck, type ParityRef } from "../model/parity";
import type { Fault, Manifest, Weights } from "../model/types";
import { parseWeights } from "../model/weights";
import type { FromWorker, ToWorker } from "./protocol";

const scope = self as unknown as DedicatedWorkerGlobalScope;
const send = (m: FromWorker, transfer: Transferable[] = []) => scope.postMessage(m, transfer);

let manifest: Manifest | null = null;
let weights: Weights | null = null;
let base = "";
let inspector: Inspector | null = null;
let parityRef: ParityRef | null = null;
let running = 0;
let live: { lanes: Lanes; paceMs: number } | null = null;
const gpts = new Map<string, GPT>();

function gptFor(spec: Pick<LaneSpec, "fault" | "mute">): GPT {
  if (!manifest || !weights) throw new Error("Model is not loaded");
  const key = `${spec.fault}|${[...spec.mute].sort((a, b) => a - b).join(",")}`;
  let g = gpts.get(key);
  if (!g) {
    g = new GPT(manifest.config, weights, spec.fault);
    if (spec.mute.length) {
      g.mute = new Uint8Array(manifest.config.n_layer * manifest.config.n_head);
      for (const i of spec.mute) g.mute[i] = 1;
    }
    gpts.set(key, g);
  }
  return g;
}

async function load(origin: string): Promise<void> {
  const t0 = performance.now();
  base = origin;
  const m = (await (await fetch(`${origin}manifest.json`)).json()) as Manifest;
  const res = await fetch(`${origin}weights.f16.bin`);
  if (!res.ok || !res.body) throw new Error(`Weights request failed (${res.status})`);
  const reader = res.body.getReader();
  const buf = new Uint8Array(m.bytes);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > m.bytes) throw new Error("Weights file is larger than its manifest says");
    buf.set(value, loaded);
    loaded += value.length;
    send({ type: "progress", loaded, total: m.bytes });
  }
  if (loaded !== m.bytes) throw new Error(`Weights truncated: ${loaded} of ${m.bytes} bytes`);
  manifest = m;
  weights = parseWeights(m, buf.buffer);
  inspector = new Inspector(gptFor({ fault: "none", mute: [] }));
  send({
    type: "ready",
    chars: m.chars,
    ctx: m.config.ctx,
    layers: m.config.n_layer,
    heads: m.config.n_head,
    dModel: m.config.d_model,
    bytes: m.bytes,
    loadMs: Math.round(performance.now() - t0),
  });
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function run(m: Extract<ToWorker, { type: "run" }>): Promise<void> {
  const myRun = m.runId;
  running = myRun;
  if (m.prompt.length === 0) {
    send({ type: "done", runId: myRun, reason: "stop", stats: m.lanes.map(() => ({ tokens: 0, computeMs: 0 })) });
    return;
  }
  const lanes = new Lanes(gptFor, m.prompt, m.lanes, m.seed, m.topK);
  live = { lanes, paceMs: m.paceMs };
  let yielded = performance.now();
  let n = 0;
  while (running === myRun && n < m.maxNew) {
    const t0 = performance.now();
    const toks = lanes.step();
    n++;
    // Stop on the first line break once the run is long enough, so a poem ends at the end of a verse.
    const atBreak = m.endAtBreak > 0 && n >= m.endAtBreak && manifest?.chars[toks[0].id] === "\n";
    const st = lanes.stats();
    toks.forEach((t) => send({ type: "token", runId: myRun, lane: t.lane, id: t.id, top: t.top, p: t.p, entropy: t.entropy, stats: st[t.lane] }));
    if (atBreak && running === myRun) break;
    const pace = live?.paceMs ?? 0;
    if (pace > 0) await sleep(Math.max(0, pace - (performance.now() - t0)));
    else if (performance.now() - yielded > 12) {
      await sleep(0);
      yielded = performance.now();
    }
  }
  const finished = running === myRun;
  if (finished) running = 0;
  if (live?.lanes === lanes) live = null;
  send({ type: "done", runId: myRun, reason: finished ? "max" : "stop", stats: lanes.stats() });
}

function inspect(m: Extract<ToWorker, { type: "inspect" }>): void {
  if (!inspector) return;
  const insp = inspector.inspect(m.ids, { target: m.target, ablate: m.ablate });
  send({ type: "inspected", reqId: m.reqId, insp }, [insp.attn.buffer as ArrayBuffer]);
}

async function parity(m: Extract<ToWorker, { type: "parity" }>): Promise<void> {
  if (!parityRef) parityRef = (await (await fetch(`${base}parity-verse.json`)).json()) as ParityRef;
  const r = parityCheck(gptFor({ fault: m.fault as Fault, mute: [] }), parityRef);
  send({ type: "parity", reqId: m.reqId, fault: m.fault, max: r.max, perPos: r.perPos });
}

function bench(m: Extract<ToWorker, { type: "bench" }>): void {
  const lanes = new Lanes(gptFor, [1], [{ temperature: 0.8, fault: "none", mute: [] }], 7, 20);
  const t0 = performance.now();
  for (let i = 0; i < m.tokens; i++) lanes.step();
  const ms = performance.now() - t0;
  send({ type: "bench", reqId: m.reqId, tokPerSec: (m.tokens / ms) * 1000, tokens: m.tokens, ms });
}

const fail = (err: unknown) => send({ type: "error", message: err instanceof Error ? err.message : String(err) });

scope.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "load":
      load(m.base).catch(fail);
      break;
    case "run":
      run(m).catch(fail);
      break;
    case "params":
      if (running === m.runId && live) {
        live.lanes.setParams(m.temperatures, m.topK);
        live.paceMs = m.paceMs;
      }
      break;
    case "stop":
      if (running === m.runId) running = 0;
      break;
    case "inspect":
      try {
        inspect(m);
      } catch (err) {
        fail(err);
      }
      break;
    case "parity":
      parity(m).catch(fail);
      break;
    case "bench":
      try {
        bench(m);
      } catch (err) {
        fail(err);
      }
      break;
  }
};
