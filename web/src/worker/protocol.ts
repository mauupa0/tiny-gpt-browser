import type { Inspection } from "../model/inspect";
import type { LaneSpec } from "../model/lanes";
import type { Fault } from "../model/types";

export type { LaneSpec };

export interface Prob {
  id: number;
  p: number;
}

export interface Stats {
  tokens: number;
  computeMs: number;
}

export type ToWorker =
  | { type: "load"; base: string }
  | { type: "run"; runId: number; prompt: number[]; lanes: LaneSpec[]; topK: number; paceMs: number; maxNew: number; endAtBreak: number; seed: number }
  | { type: "params"; runId: number; temperatures: number[]; topK: number; paceMs: number }
  | { type: "stop"; runId: number }
  | { type: "inspect"; reqId: number; ids: number[]; target: number | null; ablate: boolean }
  | { type: "parity"; reqId: number; fault: Fault }
  | { type: "bench"; reqId: number; tokens: number };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; chars: string[]; ctx: number; layers: number; heads: number; dModel: number; bytes: number; loadMs: number }
  | { type: "error"; message: string }
  | { type: "token"; runId: number; lane: number; id: number; top: Prob[]; p: number; entropy: number; stats: Stats }
  | { type: "done"; runId: number; reason: "max" | "stop"; stats: Stats[] }
  | { type: "inspected"; reqId: number; insp: Inspection }
  | { type: "parity"; reqId: number; fault: Fault; max: number; perPos: number[] }
  | { type: "bench"; reqId: number; tokPerSec: number; tokens: number; ms: number };
