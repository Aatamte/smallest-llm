import { atom } from "jotai";
import { persistAtom } from "../persist";

export type ConnectionStatus = "connected" | "reconnecting" | "disconnected";

// ── UI state atoms (not synced from DB) ─────────────────

export const connectionStatusAtom = atom<ConnectionStatus>("disconnected");
export const activeRunIdAtom = persistAtom<number | null>("sllm:activeRunId", null);
export const activeCheckpointIdAtom = persistAtom<number | null>("sllm:activeCheckpointId", null);

// ── Types re-exported for components ────────────────────

export interface RunInfo {
  id: number;
  name: string;
  status: string;
  created_at: string;
}

export interface CheckpointInfo {
  id: number;
  run_id: number;
  step: number;
  path: string;
  is_best: number;
}

// ── Compare feature (REST-fetched, user-selected) ───────

export interface CompareRunData {
  name: string;
  series: Record<string, { step: number; value: number }[]>;
  evals: { step: number; task: string; metrics: Record<string, number> }[];
}
export const compareRunIdsAtom = persistAtom<number[]>("sllm:compareRunIds", []);
export const compareRunsDataAtom = atom<Record<number, CompareRunData>>({});
