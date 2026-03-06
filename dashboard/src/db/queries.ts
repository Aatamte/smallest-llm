import { db } from "../lib/db";

/** Map frontend camelCase metric keys to backend slash-separated keys. */
const METRIC_KEY_MAP: Record<string, string> = {
  trainLoss: "train/loss",
  valLoss: "val/loss",
  lr: "train/lr",
  gradNorm: "train/grad_norm",
  tokensPerSec: "train/tokens_per_sec",
  tokensSeen: "train/tokens_seen",
  bpc: "train/bpc",
  stepTime: "train/step_time",
};

/** Get the current (max) step from the metrics table. */
export function getCurrentStep(): number {
  try {
    const rows = db.query<{ step: number }>(
      "SELECT MAX(step) as step FROM metrics"
    );
    return rows[0]?.step ?? 0;
  } catch {
    return 0;
  }
}

/** Get a time series of (step, value) for a given metric key. */
export function getMetricSeries(
  key: string
): { step: number; value: number }[] {
  const dbKey = METRIC_KEY_MAP[key] ?? key;
  try {
    return db.query<{ step: number; value: number }>(
      "SELECT step, value FROM metrics WHERE key = ? ORDER BY step",
      [dbKey]
    );
  } catch {
    return [];
  }
}

/** Get eval time series for a task/metric pair. Evals not yet in sync layer. */
export function getEvalSeries(
  _task: string,
  _metric: string
): { step: number; value: number }[] {
  return [];
}

/** Discover which eval tasks have data. Evals not yet in sync layer. */
export function getEvalTaskMetrics(): Map<string, string> {
  return new Map();
}

/** Get all eval records. Evals not yet in sync layer. */
export function getEvals(): {
  task: string;
  metric_name: string;
  step: number;
  value: number;
}[] {
  return [];
}
