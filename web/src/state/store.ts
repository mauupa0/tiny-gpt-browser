import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import type { Inspection } from "../model/inspect";
import type { Fault } from "../model/types";
import { encodeText, longShotIndex } from "../lib/text";
import type { FromWorker, LaneSpec, Prob, Stats, ToWorker } from "../worker/protocol";

export interface Meta {
  chars: string[];
  ctx: number;
  layers: number;
  heads: number;
  dModel: number;
  bytes: number;
  loadMs: number;
}

export interface LaneView {
  ids: number[];
  // Probability the lane's sampler gave each character it wrote, same index as ids.
  ps: number[];
  entropy: number[];
  top: Prob[];
  chosen: number | null;
  stats: Stats;
}

export type Channel = "hero" | "temp" | "brk";

export interface RunView {
  runId: number;
  running: boolean;
  lanes: LaneView[];
  seed: number;
}

export interface ParityOut {
  fault: Fault;
  max: number;
  perPos: number[];
}

export interface Bench {
  tokPerSec: number;
  tokens: number;
  ms: number;
}

export type Explain = "pick" | "actual";

export interface ModelState {
  phase: "loading" | "ready" | "error";
  loaded: number;
  total: number;
  error: string;
  meta: Meta | null;
  // The verse on the page: prompt first, then what the model wrote.
  text: number[];
  promptLen: number;
  dropped: number;
  heroPs: number[];
  runs: Record<Channel, RunView>;
  inspectIdx: number;
  followTail: boolean;
  explain: Explain;
  inspection: Inspection | null;
  inspecting: boolean;
  parity: Partial<Record<Fault, ParityOut>>;
  parityBusy: boolean;
  bench: Bench | null;
  benchBusy: boolean;
}

const emptyLane = (): LaneView => ({ ids: [], ps: [], entropy: [], top: [], chosen: null, stats: { tokens: 0, computeMs: 0 } });
const emptyRun = (): RunView => ({ runId: 0, running: false, lanes: [emptyLane()], seed: 0 });

const initial: ModelState = {
  phase: "loading",
  loaded: 0,
  total: 1,
  error: "",
  meta: null,
  text: [],
  promptLen: 0,
  dropped: 0,
  heroPs: [],
  runs: { hero: emptyRun(), temp: emptyRun(), brk: emptyRun() },
  inspectIdx: 0,
  followTail: true,
  explain: "pick",
  inspection: null,
  inspecting: false,
  parity: {},
  parityBusy: false,
  bench: null,
  benchBusy: false,
};

type Action =
  | { type: "batch"; msgs: FromWorker[] }
  | { type: "setText"; ids: number[]; dropped: number }
  | { type: "cut"; runId: number; n: number }
  | { type: "begin"; channel: Channel; runId: number; seed: number; lanes: number; keep: boolean; promptIds: number[]; promptLen?: number }
  | { type: "select"; idx: number }
  | { type: "follow" }
  | { type: "explain"; mode: Explain }
  | { type: "inspecting"; busy: boolean }
  | { type: "parityBusy"; busy: boolean }
  | { type: "benchBusy"; busy: boolean };

function channelOf(s: ModelState, runId: number): Channel | null {
  for (const c of ["hero", "temp", "brk"] as const) if (s.runs[c].runId === runId) return c;
  return null;
}

function apply(s: ModelState, m: FromWorker): ModelState {
  switch (m.type) {
    case "progress":
      return { ...s, loaded: m.loaded, total: m.total };
    case "ready":
      return {
        ...s,
        phase: "ready",
        loaded: m.bytes,
        total: m.bytes,
        meta: { chars: m.chars, ctx: m.ctx, layers: m.layers, heads: m.heads, dModel: m.dModel, bytes: m.bytes, loadMs: m.loadMs },
      };
    case "error":
      return { ...s, phase: s.phase === "ready" ? "ready" : "error", error: m.message, inspecting: false, parityBusy: false, benchBusy: false };
    case "token": {
      const ch = channelOf(s, m.runId);
      if (!ch) return s;
      const run = s.runs[ch];
      const lanes = run.lanes.map((l, i) =>
        i !== m.lane
          ? l
          : { ids: [...l.ids, m.id], ps: [...l.ps, m.p], entropy: [...l.entropy, m.entropy], top: m.top, chosen: m.id, stats: m.stats },
      );
      const runs = { ...s.runs, [ch]: { ...run, lanes } };
      if (ch === "hero" && m.lane === 0) return { ...s, runs, text: [...s.text, m.id], heroPs: [...s.heroPs, m.p], inspectIdx: s.followTail ? s.text.length : s.inspectIdx };
      return { ...s, runs };
    }
    case "done": {
      const ch = channelOf(s, m.runId);
      if (!ch) return s;
      const run = s.runs[ch];
      const lanes = run.lanes.map((l, i) => (m.stats[i]?.tokens ? { ...l, stats: m.stats[i] } : l));
      const runs = { ...s.runs, [ch]: { ...run, running: false, lanes } };
      // A poem that ends on a line break would leave the figure explaining a line break.
      const last = s.text.length - 1;
      if (ch !== "hero" || !s.followTail || !s.meta || last < 1) return { ...s, runs };
      const gamble = longShotIndex(s.text, s.heroPs, s.meta.chars, s.promptLen);
      const endsOnBreak = s.meta.chars[s.text[last]] === "\n";
      const moved = gamble >= 0 ? gamble : endsOnBreak ? last - 1 : s.inspectIdx;
      return { ...s, runs, inspectIdx: moved, followTail: moved === last };
    }
    case "inspected":
      return { ...s, inspection: m.insp, inspecting: false };
    case "parity":
      return { ...s, parity: { ...s.parity, [m.fault]: { fault: m.fault, max: m.max, perPos: m.perPos } }, parityBusy: false };
    case "bench":
      return { ...s, bench: { tokPerSec: m.tokPerSec, tokens: m.tokens, ms: m.ms }, benchBusy: false };
  }
}

function reducer(s: ModelState, a: Action): ModelState {
  switch (a.type) {
    case "batch":
      return a.msgs.reduce(apply, s);
    case "cut": {
      // The poem is long enough: drop whatever the worker wrote past the line break the page chose.
      if (s.runs.hero.runId !== a.runId || s.text.length <= a.n) return s;
      const lanes = s.runs.hero.lanes.map((l, i) =>
        i !== 0 ? l : { ...l, ids: l.ids.slice(0, a.n - s.promptLen), ps: l.ps.slice(0, a.n - s.promptLen), entropy: l.entropy.slice(0, a.n - s.promptLen) },
      );
      return {
        ...s,
        text: s.text.slice(0, a.n),
        heroPs: s.heroPs.slice(0, a.n),
        runs: { ...s.runs, hero: { ...s.runs.hero, lanes } },
        inspectIdx: Math.min(s.inspectIdx, a.n - 1),
      };
    }
    case "setText":
      return {
        ...s,
        text: a.ids,
        promptLen: a.ids.length,
        dropped: a.dropped,
        heroPs: a.ids.map(() => 1),
        inspectIdx: Math.max(0, a.ids.length - 1),
        followTail: true,
        runs: { ...s.runs, hero: { ...s.runs.hero, lanes: [emptyLane()] } },
      };
    case "begin": {
      const lanes = Array.from({ length: a.lanes }, emptyLane);
      const runs = {
        ...s.runs,
        [a.channel]: { runId: a.runId, running: true, lanes, seed: a.seed },
      };
      for (const c of ["hero", "temp", "brk"] as const) if (c !== a.channel) runs[c] = { ...runs[c], running: false };
      if (a.channel !== "hero") return { ...s, runs };
      return {
        ...s,
        runs,
        text: a.promptIds,
        promptLen: a.keep ? s.promptLen : a.promptIds.length,
        heroPs: a.keep ? s.heroPs : a.promptIds.map(() => 1),
        inspectIdx: Math.max(0, a.promptIds.length - 1),
        followTail: true,
      };
    }
    case "select":
      return { ...s, followTail: false, inspectIdx: Math.min(Math.max(0, a.idx), Math.max(0, s.text.length - 1)) };
    case "follow":
      return { ...s, followTail: true, inspectIdx: Math.max(0, s.text.length - 1) };
    case "explain":
      return { ...s, explain: a.mode };
    case "inspecting":
      return { ...s, inspecting: a.busy };
    case "parityBusy":
      return { ...s, parityBusy: a.busy };
    case "benchBusy":
      return { ...s, benchBusy: a.busy };
  }
}

export interface WriteOptions {
  prompt: string | null;
  temperature: number;
  topK: number;
  paceMs: number;
  maxNew: number;
  endAtBreak: number;
}

export function useModel(base: string) {
  const [state, dispatch] = useReducer(reducer, initial);
  const worker = useRef<Worker | null>(null);
  const queue = useRef<FromWorker[]>([]);
  const frame = useRef(0);
  const live = useRef(state);
  live.current = state;
  const nextRun = useRef(1);
  const nextReq = useRef(1);
  const inflight = useRef(false);
  const wantAblation = useRef(false);
  const pending = useRef(false);
  const lastReq = useRef(0);
  // After the page ends a poem itself, tokens the worker had already produced are dropped on arrival.
  const cutAt = useRef<number | null>(null);

  const post = useCallback((m: ToWorker, transfer: Transferable[] = []) => worker.current?.postMessage(m, transfer), []);

  const requestInspect = useCallback(() => {
    const s = live.current;
    if (s.phase !== "ready" || s.text.length === 0) return;
    if (inflight.current) {
      pending.current = true;
      return;
    }
    const idx = Math.min(s.inspectIdx, s.text.length - 1);
    const ids = s.text.slice(0, idx + 1);
    const next = s.text[idx + 1];
    const target = s.explain === "actual" && next !== undefined ? next : null;
    inflight.current = true;
    pending.current = false;
    dispatch({ type: "inspecting", busy: true });
    const reqId = nextReq.current++;
    lastReq.current = reqId;
    post({ type: "inspect", reqId, ids, target, ablate: wantAblation.current });
    // A lost reply must not freeze the figure: ask again if nothing came back.
    window.setTimeout(() => {
      if (inflight.current && lastReq.current === reqId) {
        inflight.current = false;
        requestInspect();
      }
    }, 4000);
  }, [post]);

  useEffect(() => {
    const w = new Worker(new URL("../worker/inference.worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    const flush = () => {
      frame.current = 0;
      const msgs = queue.current;
      queue.current = [];
      if (msgs.length) dispatch({ type: "batch", msgs });
    };
    const enqueue = (m: FromWorker) => {
      queue.current.push(m);
      if (!frame.current) frame.current = requestAnimationFrame(flush);
    };
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "token" && m.runId === cutAt.current && m.lane === 0) return;
      if (m.type === "inspected") {
        inflight.current = false;
        enqueue(m);
        if (pending.current) queueMicrotask(requestInspect);
        return;
      }
      if (m.type === "error") inflight.current = false;
      enqueue(m);
    };
    w.onerror = (e) => enqueue({ type: "error", message: e.message || "Worker failed" });
    const timer = window.setTimeout(() => w.postMessage({ type: "load", base } satisfies ToWorker), 150);
    return () => {
      window.clearTimeout(timer);
      cancelAnimationFrame(frame.current);
      w.terminate();
    };
  }, [base, requestInspect]);

  // Re-inspect whenever the inspected position, the text length (while writing) or the explain mode changes.
  const wasReady = state.phase === "ready";
  useEffect(() => {
    if (wasReady) requestInspect();
  }, [wasReady, state.inspectIdx, state.text.length, state.explain, requestInspect]);

  const setText = useCallback(
    (prompt: string) => {
      const chars = live.current.meta?.chars;
      if (!chars) return;
      const { ids, dropped } = encodeText(chars, prompt);
      if (ids.length === 0) return;
      dispatch({ type: "setText", ids, dropped });
    },
    [],
  );

  const beginRun = useCallback(
    (channel: Channel, promptIds: number[], lanes: LaneSpec[], opt: { topK: number; paceMs: number; maxNew: number; endAtBreak: number }, keep = false) => {
      const runId = nextRun.current++;
      cutAt.current = null;
      const seed = (Math.random() * 2 ** 31) | 0;
      for (const c of ["hero", "temp", "brk"] as const) {
        const r = live.current.runs[c];
        if (r.running) post({ type: "stop", runId: r.runId });
      }
      dispatch({ type: "begin", channel, runId, seed, lanes: lanes.length, keep, promptIds });
      post({ type: "run", runId, prompt: promptIds, lanes, topK: opt.topK, paceMs: opt.paceMs, maxNew: opt.maxNew, endAtBreak: opt.endAtBreak, seed });
      return runId;
    },
    [post],
  );

  const writeHero = useCallback(
    (o: WriteOptions) => {
      const s = live.current;
      const chars = s.meta?.chars;
      if (!chars) return;
      let ids: number[];
      let keep = false;
      if (o.prompt === null) {
        ids = s.text;
        keep = true;
      } else {
        ids = encodeText(chars, o.prompt).ids;
      }
      if (ids.length === 0) return;
      beginRun("hero", ids, [{ temperature: o.temperature, fault: "none", mute: [] }], { topK: o.topK, paceMs: o.paceMs, maxNew: o.maxNew, endAtBreak: o.endAtBreak }, keep);
    },
    [beginRun],
  );

  const writeLanes = useCallback(
    (channel: "temp" | "brk", lanes: LaneSpec[], o: { topK: number; paceMs: number; maxNew: number; endAtBreak?: number }) => {
      const s = live.current;
      const ids = s.text.slice(0, s.promptLen);
      if (ids.length === 0) return;
      beginRun(channel, ids, lanes, { ...o, endAtBreak: o.endAtBreak ?? 0 });
    },
    [beginRun],
  );

  const finishHero = useCallback(
    (n: number) => {
      const r = live.current.runs.hero;
      if (!r.running) return;
      cutAt.current = r.runId;
      queue.current = queue.current.filter((m) => !(m.type === "token" && m.runId === r.runId && m.lane === 0));
      dispatch({ type: "cut", runId: r.runId, n });
      post({ type: "stop", runId: r.runId });
    },
    [post],
  );

  const stop = useCallback(
    (channel: Channel) => {
      const r = live.current.runs[channel];
      if (r.running) post({ type: "stop", runId: r.runId });
    },
    [post],
  );

  const setParams = useCallback(
    (channel: Channel, temperatures: number[], topK: number, paceMs: number) => {
      const r = live.current.runs[channel];
      if (r.running) post({ type: "params", runId: r.runId, temperatures, topK, paceMs });
    },
    [post],
  );

  const select = useCallback((idx: number) => dispatch({ type: "select", idx }), []);
  const follow = useCallback(() => dispatch({ type: "follow" }), []);
  const setExplain = useCallback((mode: Explain) => dispatch({ type: "explain", mode }), []);

  const setAblation = useCallback(
    (on: boolean) => {
      if (wantAblation.current === on) return;
      wantAblation.current = on;
      if (on) requestInspect();
    },
    [requestInspect],
  );

  const runParity = useCallback(
    (fault: Fault) => {
      dispatch({ type: "parityBusy", busy: true });
      post({ type: "parity", reqId: nextReq.current++, fault });
    },
    [post],
  );

  const runBench = useCallback(() => {
    dispatch({ type: "benchBusy", busy: true });
    post({ type: "bench", reqId: nextReq.current++, tokens: 400 });
  }, [post]);

  return useMemo(
    () => ({ state, setText, writeHero, writeLanes, finishHero, stop, setParams, select, follow, setExplain, setAblation, runParity, runBench }),
    [state, setText, writeHero, writeLanes, finishHero, stop, setParams, select, follow, setExplain, setAblation, runParity, runBench],
  );
}
