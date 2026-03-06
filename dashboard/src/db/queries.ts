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

// ── Run state queries (merged into runs table) ───────

export function getStatus(runId: number | null): string {
  if (runId == null) return "idle";
  try {
    const rows = db.query<{ live_status: string }>(
      "SELECT live_status FROM runs WHERE id = ?", [runId]
    );
    return rows[0]?.live_status ?? "idle";
  } catch { return "idle"; }
}

export function getTextState(runId?: number | null): string {
  try {
    if (runId != null) {
      const rows = db.query<{ text_state: string }>(
        "SELECT text_state FROM runs WHERE id = ?", [runId]
      );
      return rows[0]?.text_state ?? "";
    }
    const rows = db.query<{ text_state: string }>(
      "SELECT text_state FROM runs ORDER BY id DESC LIMIT 1"
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
}

export function getRunState(runId: number | null): RunState | null {
  if (runId == null) return null;
  try {
    const rows = db.query<RunState>(
      "SELECT live_status as status, text_state, stage_index, stage_name, total_stages, dataset, stage_type FROM runs WHERE id = ?",
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

export function getRunMaxSteps(runId: number | null): number {
  if (runId == null) return 0;
  try {
    const rows = db.query<{ config: string }>(
      "SELECT config FROM runs WHERE id = ?", [runId]
    );
    if (!rows[0]?.config) return 0;
    const cfg = JSON.parse(rows[0].config);
    return cfg?.training?.max_steps ?? 0;
  } catch { return 0; }
}

export function getRunStartTime(runId: number | null): string {
  if (runId == null) return "";
  try {
    const rows = db.query<{ created_at: string }>(
      "SELECT created_at FROM runs WHERE id = ?", [runId]
    );
    return rows[0]?.created_at ?? "";
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

export function getMetricKeys(runId?: number | null): string[] {
  try {
    if (runId != null) {
      const rows = db.query<{ key: string }>(
        "SELECT DISTINCT key FROM metrics WHERE run_id = ? ORDER BY key",
        [runId]
      );
      return rows.map((r) => r.key);
    }
    const rows = db.query<{ key: string }>(
      "SELECT DISTINCT key FROM metrics ORDER BY key"
    );
    return rows.map((r) => r.key);
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
