import type { TrainingState } from "../types/metrics";

export const API_BASE = `http://${window.location.hostname}:8000/api`;

export async function startRun(config?: Record<string, unknown>): Promise<{ status: string; run_id: number }> {
  const res = await fetch(`${API_BASE}/runs/start`, {
    method: "POST",
    headers: config ? { "Content-Type": "application/json" } : {},
    body: config ? JSON.stringify(config) : undefined,
  });
  if (!res.ok) throw new Error(`Failed to start run: ${res.status}`);
  return res.json();
}

export async function stopRun(runId: number): Promise<{ status: string; run_id: number }> {
  const res = await fetch(`${API_BASE}/runs/${runId}/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to stop run: ${res.status}`);
  return res.json();
}

export async function deleteRun(runId: number): Promise<{ status: string; run_id: number }> {
  const res = await fetch(`${API_BASE}/runs/${runId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete run: ${res.status}`);
  return res.json();
}

export async function bulkDeleteRuns(runIds: number[]): Promise<{ status: string; run_ids: number[] }> {
  const res = await fetch(`${API_BASE}/runs/bulk-delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds }),
  });
  if (!res.ok) throw new Error(`Failed to bulk delete runs: ${res.status}`);
  return res.json();
}

export async function fetchConfig(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(`Failed to fetch config: ${res.status}`);
  return res.json();
}

export async function fetchPresets(): Promise<{ name: string; label: string }[]> {
  const res = await fetch(`${API_BASE}/presets`);
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.status}`);
  return res.json();
}

export async function fetchPreset(name: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/presets/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Failed to fetch preset: ${res.status}`);
  return res.json();
}

export async function fetchTrainingState(runId: number): Promise<TrainingState> {
  const res = await fetch(`${API_BASE}/runs/${runId}/state`);
  if (!res.ok) throw new Error(`Failed to fetch state: ${res.status}`);
  return res.json();
}

export async function fetchRuns(): Promise<{ id: number; name: string; status: string; created_at: string }[]> {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error(`Failed to fetch runs: ${res.status}`);
  return res.json();
}

export async function fetchMetrics(
  runId: number,
  key?: string,
): Promise<{ step: number; key: string; value: number }[]> {
  const url = key
    ? `${API_BASE}/runs/${runId}/metrics?key=${encodeURIComponent(key)}`
    : `${API_BASE}/runs/${runId}/metrics`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
  return res.json();
}

// ── Run detail ──────────────────────────────────────────

export interface RunDetail {
  id: number;
  name: string;
  status: string;
  config: {
    model?: { name: string; extra_args: Record<string, unknown> };
    data?: Record<string, unknown>;
    training?: Record<string, unknown>;
    [key: string]: unknown;
  };
  env: Record<string, unknown>;
  created_at: string;
}

export async function fetchRunDetail(runId: number): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to fetch run: ${res.status}`);
  return res.json();
}

// ── Checkpoints ─────────────────────────────────────────

export interface Checkpoint {
  id: number;
  run_id: number;
  step: number;
  path: string;
  metrics: Record<string, number>;
  is_best: number;
  created_at: string;
}

export async function fetchCheckpoints(runId: number): Promise<Checkpoint[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/checkpoints`);
  if (!res.ok) throw new Error(`Failed to fetch checkpoints: ${res.status}`);
  return res.json();
}

// ── Weights ─────────────────────────────────────────────

export interface WeightLayer {
  name: string;
  shape: number[];
  data: number[][];
}

export async function fetchWeights(runId: number, step: number): Promise<WeightLayer[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/checkpoints/${step}/weights`);
  if (!res.ok) throw new Error(`Failed to fetch weights: ${res.status}`);
  const json = await res.json();
  return json.layers;
}

// ── Evals ───────────────────────────────────────────────

export interface EvalResult {
  id: number;
  run_id: number;
  step: number;
  task: string;
  metrics: Record<string, number>;
  metadata: Record<string, unknown>;
  model_name: string | null;
  created_at: string;
}

export async function fetchEvals(runId: number): Promise<EvalResult[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/evals`);
  if (!res.ok) throw new Error(`Failed to fetch evals: ${res.status}`);
  return res.json();
}

export async function fetchAllEvals(modelName?: string): Promise<EvalResult[]> {
  const url = modelName
    ? `${API_BASE}/evals?model_name=${encodeURIComponent(modelName)}`
    : `${API_BASE}/evals`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch evals: ${res.status}`);
  return res.json();
}

export async function fetchEvalModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/evals/models`);
  if (!res.ok) throw new Error(`Failed to fetch eval models: ${res.status}`);
  return res.json();
}

// ── HF Model Evaluation ─────────────────────────────────

export interface AvailableModel {
  name: string;
  hf_id: string;
}

export interface EvalStatus {
  status: "idle" | "running" | "error" | "stopped";
  model_name: string | null;
  task: string | null;
  task_index: number;
  task_count: number;
  current_sample: number;
  total_samples: number;
  started_at: number | null;
  error: string | null;
}

export async function fetchAvailableModels(): Promise<AvailableModel[]> {
  const res = await fetch(`${API_BASE}/evals/available-models`);
  if (!res.ok) throw new Error(`Failed to fetch available models: ${res.status}`);
  return res.json();
}

export type RunEvalRequest =
  | { source?: "hf"; model_name: string; tasks: string[]; max_samples?: number }
  | { source: "checkpoint"; run_id: number; step: number; tasks: string[]; max_samples?: number };

export async function runEval(
  request: RunEvalRequest,
): Promise<{ status: string; model_name: string; tasks: string[] }> {
  const res = await fetch(`${API_BASE}/evals/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to start eval: ${res.status}`);
  }
  return res.json();
}

export async function stopEval(): Promise<EvalStatus> {
  const res = await fetch(`${API_BASE}/evals/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to stop eval: ${res.status}`);
  return res.json();
}

export async function fetchEvalStatus(): Promise<EvalStatus> {
  const res = await fetch(`${API_BASE}/evals/status`);
  if (!res.ok) throw new Error(`Failed to fetch eval status: ${res.status}`);
  return res.json();
}

// ── Chat / Generation ───────────────────────────────────

export interface ChatStatus {
  loaded: boolean;
  source: "checkpoint" | "hf" | null;
  name: string | null;
}

export async function fetchChatStatus(): Promise<ChatStatus> {
  const res = await fetch(`${API_BASE}/chat/status`);
  if (!res.ok) throw new Error(`Failed to fetch chat status: ${res.status}`);
  return res.json();
}

export async function loadChatModel(
  opts: { source: "hf"; model_name: string } | { source: "checkpoint"; run_id: number; step: number },
): Promise<{ status: string; name: string }> {
  const res = await fetch(`${API_BASE}/chat/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to load model: ${res.status}`);
  }
  return res.json();
}

export async function generateChat(
  prompt: string,
  params?: { max_tokens?: number; temperature?: number; top_k?: number },
): Promise<{ text: string; prompt: string }> {
  const res = await fetch(`${API_BASE}/chat/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...params }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to generate: ${res.status}`);
  }
  return res.json();
}

export async function unloadChatModel(): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/unload`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to unload model: ${res.status}`);
}
