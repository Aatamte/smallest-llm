import { useAtomValue } from "jotai";
import { getTableVersionAtom } from "./atoms";
import {
  getMetricSeries,
  getLatestMetric,
  getCurrentStep,
  getRunState,
  getStatus,
  getRuns,
  getRunModelName,
  getRunMaxSteps,
  getRunStartTime,
  getCheckpoints,
  getGenerations,
  getLogs,
  getEvalSeries,
  getEvalTaskMetrics,
  getEvals,
  getTextState,
  getCompareRunSeries,
  getMetricKeys,
} from "./queries";

// ── Building block ──────────────────────────────
export function useTableQuery<T>(tables: string | string[], queryFn: () => T): T {
  const names = Array.isArray(tables) ? tables : [tables];
  for (const name of names) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useAtomValue(getTableVersionAtom(name));
  }
  return queryFn();
}

// ── Metrics ─────────────────────────────────────
export function useMetricSeries(key: string, runId?: number | null) {
  return useTableQuery("metrics", () => getMetricSeries(key, runId));
}

export function useLatestMetric(key: string, runId: number | null) {
  return useTableQuery("metrics", () => getLatestMetric(runId, key));
}

export function useCurrentStep(runId?: number | null) {
  return useTableQuery("metrics", () => getCurrentStep(runId));
}

export function useCompareRunSeries(runId: number, key: string) {
  return useTableQuery("metrics", () => getCompareRunSeries(runId, key));
}

export function useMetricKeys(runId?: number | null) {
  return useTableQuery("metrics", () => getMetricKeys(runId));
}

// ── Run state (merged into runs table) ──────────
export function useRunState(runId: number | null) {
  return useTableQuery("runs", () => getRunState(runId));
}

export function useStatus(runId: number | null) {
  return useTableQuery("runs", () => getStatus(runId));
}

export function useTextState(runId?: number | null) {
  return useTableQuery("runs", () => getTextState(runId));
}

// ── Runs ────────────────────────────────────────
export function useRuns() {
  return useTableQuery("runs", () => getRuns());
}

export function useRunModelName(runId: number | null) {
  return useTableQuery("runs", () => getRunModelName(runId));
}

export function useRunMaxSteps(runId: number | null) {
  return useTableQuery("runs", () => getRunMaxSteps(runId));
}

export function useRunStartTime(runId: number | null) {
  return useTableQuery("runs", () => getRunStartTime(runId));
}

// ── Other tables ────────────────────────────────
export function useCheckpoints(runId: number | null) {
  return useTableQuery("checkpoints", () => getCheckpoints(runId));
}

export function useGenerations(runId: number | null) {
  return useTableQuery("generations", () => getGenerations(runId));
}

export function useLogs(runId: number | null) {
  return useTableQuery("logs", () => getLogs(runId));
}

// ── Evals (stored in metrics table) ─────────────
export function useEvalSeries(task: string, metric: string) {
  return useTableQuery("metrics", () => getEvalSeries(task, metric));
}

export function useEvalTaskMetrics() {
  return useTableQuery("metrics", () => getEvalTaskMetrics());
}

export function useEvals() {
  return useTableQuery("metrics", () => getEvals());
}
