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

// ── Run state queries ──────────────────────────────────

export function getStatus(runId: number | null): string {
  if (runId == null) return "idle";
  try {
    const rows = db.query<{ status: string }>(
      "SELECT status FROM run_state WHERE run_id = ?", [runId]
    );
    return rows[0]?.status ?? "idle";
  } catch { return "idle"; }
}

export function getMaxSteps(runId: number | null): number {
  if (runId == null) return 0;
  try {
    const rows = db.query<{ max_steps: number }>(
      "SELECT max_steps FROM run_state WHERE run_id = ?", [runId]
    );
    return rows[0]?.max_steps ?? 0;
  } catch { return 0; }
}

export function getStartTime(runId: number | null): string {
  if (runId == null) return "";
  try {
    const rows = db.query<{ start_time: string }>(
      "SELECT start_time FROM run_state WHERE run_id = ?", [runId]
    );
    return rows[0]?.start_time ?? "";
  } catch { return ""; }
}

export function getTokensPerStep(runId: number | null): number {
  if (runId == null) return 0;
  try {
    const rows = db.query<{ tokens_per_step: number }>(
      "SELECT tokens_per_step FROM run_state WHERE run_id = ?", [runId]
    );
    return rows[0]?.tokens_per_step ?? 0;
  } catch { return 0; }
}

export function getTextState(runId?: number | null): string {
  try {
    if (runId != null) {
      const rows = db.query<{ text_state: string }>(
        "SELECT text_state FROM run_state WHERE run_id = ?", [runId]
      );
      return rows[0]?.text_state ?? "";
    }
    const rows = db.query<{ text_state: string }>(
      "SELECT text_state FROM run_state ORDER BY run_id DESC LIMIT 1"
    );
    return rows[0]?.text_state ?? "";
  } catch { return ""; }
}

export interface RunState {
  status: string;
  text_state: string;
  stage_index: number;
  stage_name: string;
  total_stages: number;
  dataset: string;
  stage_type: string;
  max_steps: number;
  start_time: string;
  tokens_per_step: number;
}

export function getRunState(runId: number | null): RunState | null {
  if (runId == null) return null;
  try {
    const rows = db.query<RunState>(
      "SELECT status, text_state, stage_index, stage_name, total_stages, dataset, stage_type, max_steps, start_time, tokens_per_step FROM run_state WHERE run_id = ?",
      [runId]
    );
    return rows[0] ?? null;
  } catch { return null; }
}

export function getRunModelName(runId: number | null): string {
  if (runId == null) return "";
  try {
    const rows = db.query<{ config: string }>(
      "SELECT config FROM runs WHERE id = ?", [runId]
    );
    if (!rows[0]?.config) return "";
    const cfg = JSON.parse(rows[0].config);
    return cfg?.model?.name ?? "";
  } catch { return ""; }
}

// ── Metrics queries ────────────────────────────────────

export function getCurrentStep(runId?: number | null): number {
  try {
    if (runId != null) {
      const rows = db.query<{ step: number }>(
        "SELECT MAX(step) as step FROM metrics WHERE run_id = ?", [runId]
      );
      return rows[0]?.step ?? 0;
    }
    const rows = db.query<{ step: number }>(
      "SELECT MAX(step) as step FROM metrics"
    );
    return rows[0]?.step ?? 0;
  } catch { return 0; }
}

export function getLatestMetric(runId: number | null, key: string): number {
  if (runId == null) return 0;
  const dbKey = METRIC_KEY_MAP[key] ?? key;
  try {
    const rows = db.query<{ value: number }>(
      "SELECT value FROM metrics WHERE run_id = ? AND key = ? ORDER BY step DESC LIMIT 1",
      [runId, dbKey]
    );
    return rows[0]?.value ?? 0;
  } catch { return 0; }
}

export function getMetricSeries(key: string, runId?: number | null): { step: number; value: number }[] {
  const dbKey = METRIC_KEY_MAP[key] ?? key;
  try {
    if (runId != null) {
      return db.query<{ step: number; value: number }>(
        "SELECT step, value FROM metrics WHERE run_id = ? AND key = ? ORDER BY step",
        [runId, dbKey]
      );
    }
    return db.query<{ step: number; value: number }>(
      "SELECT step, value FROM metrics WHERE key = ? ORDER BY step",
      [dbKey]
    );
  } catch { return []; }
}

export function getTrainLossSeries(runId: number | null): { step: number; value: number }[] {
  if (runId == null) return [];
  return getMetricSeries("trainLoss", runId);
}

export function getValLossSeries(runId: number | null): { step: number; value: number }[] {
  if (runId == null) return [];
  return getMetricSeries("valLoss", runId);
}

export function getCompareRunSeries(runId: number, key: string): { step: number; value: number }[] {
  return getMetricSeries(key, runId);
}

// ── Runs & checkpoints ─────────────────────────────────

export function getRuns(): { id: number; name: string; status: string; created_at: string }[] {
  try {
    return db.query<{ id: number; name: string; status: string; created_at: string }>(
      "SELECT id, name, status, created_at FROM runs ORDER BY id DESC"
    );
  } catch { return []; }
}

export function getCheckpoints(runId: number | null): { id: number; run_id: number; step: number; path: string; is_best: number }[] {
  if (runId == null) return [];
  try {
    return db.query<{ id: number; run_id: number; step: number; path: string; is_best: number }>(
      "SELECT id, run_id, step, path, is_best FROM checkpoints WHERE run_id = ? ORDER BY step",
      [runId]
    );
  } catch { return []; }
}

// ── Layer stats & activation stats ─────────────────────

export function getLayerStats(runId: number | null): { name: string; gradNorm: number; weightNorm: number; updateRatio: number }[] {
  if (runId == null) return [];
  try {
    const rows = db.query<{ layer: string; grad_norm: number; weight_norm: number; update_ratio: number }>(
      `SELECT layer, grad_norm, weight_norm, update_ratio FROM layer_stats
       WHERE run_id = ? AND step = (SELECT MAX(step) FROM layer_stats WHERE run_id = ?)
       ORDER BY layer`,
      [runId, runId]
    );
    return rows.map((r) => ({
      name: r.layer,
      gradNorm: r.grad_norm ?? 0,
      weightNorm: r.weight_norm ?? 0,
      updateRatio: r.update_ratio ?? 0,
    }));
  } catch { return []; }
}

export function getActivationStats(runId: number | null): { name: string; mean: number; std: number; max: number; min: number; pctZero: number }[] {
  if (runId == null) return [];
  try {
    const rows = db.query<{ layer: string; mean: number; std: number; max_val: number; min_val: number; pct_zero: number }>(
      `SELECT layer, mean, std, max_val, min_val, pct_zero FROM activation_stats
       WHERE run_id = ? AND step = (SELECT MAX(step) FROM activation_stats WHERE run_id = ?)
       ORDER BY layer`,
      [runId, runId]
    );
    return rows.map((r) => ({
      name: r.layer,
      mean: r.mean ?? 0,
      std: r.std ?? 0,
      max: r.max_val ?? 0,
      min: r.min_val ?? 0,
      pctZero: r.pct_zero ?? 0,
    }));
  } catch { return []; }
}

// ── Generations ────────────────────────────────────────

export function getGenerations(runId: number | null): { step: number; prompt: string; output: string }[] {
  if (runId == null) return [];
  try {
    return db.query<{ step: number; prompt: string; output: string }>(
      "SELECT step, prompt, output FROM generations WHERE run_id = ? ORDER BY id DESC LIMIT 10",
      [runId]
    );
  } catch { return []; }
}

// ── Logs ───────────────────────────────────────────────

export function getLogs(runId: number | null): { id: number; level: string; message: string; created_at: string }[] {
  if (runId == null) return [];
  try {
    return db.query<{ id: number; level: string; message: string; created_at: string }>(
      "SELECT id, level, message, created_at FROM logs WHERE run_id = ? ORDER BY id DESC LIMIT 500",
      [runId]
    );
  } catch { return []; }
}

// ── Eval queries ───────────────────────────────────────

export function getEvalSeries(task: string, metric: string): { step: number; value: number }[] {
  try {
    return db.query<{ step: number; value: number }>(
      "SELECT step, value FROM metrics WHERE key = ? ORDER BY step",
      [`eval/${task}/${metric}`]
    );
  } catch { return []; }
}

export function getEvalTaskMetrics(): Map<string, string> {
  try {
    const rows = db.query<{ key: string }>(
      "SELECT DISTINCT key FROM metrics WHERE key LIKE 'eval/%'"
    );
    const map = new Map<string, string>();
    for (const row of rows) {
      const parts = row.key.split("/");
      if (parts.length >= 3) {
        const task = parts[1];
        const metric = parts.slice(2).join("/");
        if (!map.has(task)) map.set(task, metric);
      }
    }
    return map;
  } catch { return new Map(); }
}

export function getEvals(): { task: string; metric_name: string; step: number; value: number }[] {
  try {
    const rows = db.query<{ key: string; step: number; value: number }>(
      "SELECT key, step, value FROM metrics WHERE key LIKE 'eval/%' ORDER BY step"
    );
    return rows.map((r) => {
      const parts = r.key.split("/");
      return {
        task: parts[1] ?? "",
        metric_name: parts.slice(2).join("/"),
        step: r.step,
        value: r.value,
      };
    });
  } catch { return []; }
}
