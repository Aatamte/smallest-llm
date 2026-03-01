import { atom } from "jotai";
import type { StepMetrics, LayerStat, ActivationStat, Generation, TrainingStatus, LogEntry } from "../../types/metrics";
import { persistAtom } from "../persist";

export type ConnectionStatus = "connected" | "reconnecting" | "disconnected";

const MAX_STEPS = 50_000;

// ── Core atoms ──────────────────────────────────────────

export const stepsAtom = atom<StepMetrics[]>([]);
export const statusAtom = atom<TrainingStatus>("idle");
export const connectionStatusAtom = atom<ConnectionStatus>("disconnected");
export const activeRunIdAtom = persistAtom<number | null>("sllm:activeRunId", null);
export const experimentNameAtom = atom("smallest-llm");
export const maxStepsAtom = atom(10000);
export const startTimeAtom = persistAtom<number>("sllm:startTime", Date.now());
export const hydratingAtom = atom(false);
export const layerStatsAtom = atom<LayerStat[]>([]);
export const activationStatsAtom = atom<ActivationStat[]>([]);
export const generationsAtom = atom<Generation[]>([]);
export const logsAtom = atom<LogEntry[]>([]);
export const tokensPerStepAtom = atom(0);

// ── Incremental tracking atoms ──────────────────────────
// These avoid O(n) scans over the full steps array.

const _bestValLossAtom = atom(Infinity);
const _lastValLossAtom = atom(0);

export interface RunInfo {
  id: number;
  name: string;
  status: string;
  created_at: string;
}
export const availableRunsAtom = atom<RunInfo[]>([]);

export interface CheckpointInfo {
  id: number;
  run_id: number;
  step: number;
  path: string;
  is_best: number;
}
export const availableCheckpointsAtom = atom<CheckpointInfo[]>([]);
export const activeCheckpointIdAtom = persistAtom<number | null>("sllm:activeCheckpointId", null);

export interface EvalResultInfo {
  id: number;
  run_id: number;
  step: number;
  task: string;
  metrics: Record<string, number>;
  metadata: Record<string, unknown>;
  model_name: string | null;
  created_at: string;
}
export const evalsAtom = atom<EvalResultInfo[]>([]);

// ── Derived atoms (computed from stepsAtom) ─────────────

export const currentStepAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].step : 0;
});

export const currentTrainLossAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].trainLoss : 0;
});

export const currentValLossAtom = atom((get) => get(_lastValLossAtom));

export const bestValLossAtom = atom((get) => get(_bestValLossAtom));

export const currentLRAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].lr : 0;
});

export const currentGradNormAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].gradNorm : 0;
});

export const tokensSeenAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].tokensSeen : 0;
});

export const tokensPerSecAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? steps[steps.length - 1].tokensPerSec : 0;
});

export const bpcAtom = atom((get) => {
  const steps = get(stepsAtom);
  return steps.length > 0 ? (steps[steps.length - 1].bpc ?? 0) : 0;
});

// ── Write atoms (actions) ───────────────────────────────

export const pushStepAtom = atom(null, (get, set, step: StepMetrics) => {
  const prev = get(stepsAtom);
  const next = prev.length >= MAX_STEPS
    ? [...prev.slice(prev.length - MAX_STEPS + 1), step]
    : [...prev, step];
  set(stepsAtom, next);
  set(statusAtom, "training");

  if (step.valLoss !== undefined) {
    set(_lastValLossAtom, step.valLoss);
    if (step.valLoss < get(_bestValLossAtom)) {
      set(_bestValLossAtom, step.valLoss);
    }
  }
});

export const hydrateStepsAtom = atom(null, (_get, set, steps: StepMetrics[]) => {
  const capped = steps.length > MAX_STEPS ? steps.slice(steps.length - MAX_STEPS) : steps;
  set(stepsAtom, capped);
  set(statusAtom, "training");

  let bestVal = Infinity;
  let lastVal = 0;
  for (const s of capped) {
    if (s.valLoss !== undefined) {
      lastVal = s.valLoss;
      if (s.valLoss < bestVal) bestVal = s.valLoss;
    }
  }
  set(_bestValLossAtom, bestVal);
  set(_lastValLossAtom, lastVal);
});

export const addGenerationAtom = atom(null, (get, set, gen: Generation) => {
  set(generationsAtom, [...get(generationsAtom).slice(-9), gen]);
});

export const addLogAtom = atom(null, (get, set, entry: LogEntry) => {
  const logs = get(logsAtom);
  set(logsAtom, [...logs.slice(-(500 - 1)), entry]);
});

export const clearLogsAtom = atom(null, (_get, set) => {
  set(logsAtom, []);
});

export const resetAtom = atom(null, (_get, set) => {
  set(stepsAtom, []);
  set(statusAtom, "idle");
  set(hydratingAtom, false);
  set(layerStatsAtom, []);
  set(activationStatsAtom, []);
  set(generationsAtom, []);
  set(logsAtom, []);
  set(evalsAtom, []);
  set(startTimeAtom, Date.now());
  set(connectionStatusAtom, "disconnected");
  set(activeRunIdAtom, null);
  set(tokensPerStepAtom, 0);
  set(_bestValLossAtom, Infinity);
  set(_lastValLossAtom, 0);
});
