import { useEffect, useCallback, useRef, useState } from "react";
import {
  fetchAllEvals,
  fetchAvailableModels,
  fetchEvalStatus,
  fetchRuns,
  fetchCheckpoints,
  runEval,
  stopEval,
  type AvailableModel,
  type EvalStatus,
  type RunEvalRequest,
} from "../api/client";
import { EvalPage } from "../components/EvalPage";
import type { RunInfo } from "../storage";

// model_name → task → metrics
export type GroupedEvals = Record<string, Record<string, Record<string, number>>>;

export interface CheckpointOption {
  runId: number;
  runName: string;
  step: number;
  label: string;
}

export function EvalContainer() {
  const [allEvals, setAllEvals] = useState<GroupedEvals>({});
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [checkpointOptions, setCheckpointOptions] = useState<CheckpointOption[]>([]);
  const [evalStatus, setEvalStatus] = useState<EvalStatus>({
    status: "idle",
    model_name: null,
    task: null,
    task_index: 0,
    task_count: 0,
    current_sample: 0,
    total_samples: 0,
    started_at: null,
    error: null,
  });

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load all evals grouped by model_name → task → metrics
  const loadAllEvals = useCallback(() => {
    fetchAllEvals()
      .then((evals) => {
        const grouped: GroupedEvals = {};
        for (const e of evals) {
          const name = e.model_name;
          if (!name) continue;
          if (!grouped[name]) grouped[name] = {};
          // Keep latest per task (by created_at or id)
          grouped[name][e.task] = e.metrics;
        }
        setAllEvals(grouped);
      })
      .catch((e) => console.warn("Failed to fetch evals:", e));
  }, []);

  useEffect(() => {
    loadAllEvals();
  }, [loadAllEvals]);

  // Sync eval status on mount (picks up already-running evals)
  useEffect(() => {
    fetchEvalStatus()
      .then(setEvalStatus)
      .catch((e) => console.warn("Failed to fetch eval status:", e));
  }, []);

  // Load available HF models
  useEffect(() => {
    fetchAvailableModels()
      .then(setAvailableModels)
      .catch((e) => console.warn("Failed to fetch available models:", e));
  }, []);

  // Load checkpoint options (runs + their checkpoints)
  useEffect(() => {
    fetchRuns()
      .then(async (runs: RunInfo[]) => {
        const options: CheckpointOption[] = [];
        for (const run of runs) {
          try {
            const checkpoints = await fetchCheckpoints(run.id);
            for (const cp of checkpoints) {
              options.push({
                runId: run.id,
                runName: run.name,
                step: cp.step,
                label: `${run.name} (step ${cp.step})`,
              });
            }
          } catch {
            // skip runs without checkpoints
          }
        }
        setCheckpointOptions(options);
      })
      .catch((e) => console.warn("Failed to fetch checkpoints:", e));
  }, []);

  // Poll eval status while running
  useEffect(() => {
    if (evalStatus.status === "running") {
      pollRef.current = setInterval(() => {
        fetchEvalStatus().then((s) => {
          setEvalStatus(s);
          if (s.status !== "running") {
            loadAllEvals();
          }
        });
      }, 2000);
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [evalStatus.status, loadAllEvals]);

  const handleRunEval = useCallback(
    async (request: RunEvalRequest) => {
      try {
        const result = await runEval(request);
        setEvalStatus({
          status: "running", model_name: result.model_name, task: null,
          task_index: 0, task_count: 0, current_sample: 0, total_samples: 0,
          started_at: Date.now() / 1000, error: null,
        });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setEvalStatus({
          status: "error", model_name: null, task: null,
          task_index: 0, task_count: 0, current_sample: 0, total_samples: 0,
          started_at: null, error: msg,
        });
      }
    },
    [],
  );

  const handleStopEval = useCallback(async () => {
    try {
      const status = await stopEval();
      setEvalStatus(status);
    } catch (e) {
      console.warn("Failed to stop eval:", e);
    }
  }, []);

  return (
    <EvalPage
      allEvals={allEvals}
      availableModels={availableModels}
      checkpointOptions={checkpointOptions}
      evalStatus={evalStatus}
      onRunEval={handleRunEval}
      onStopEval={handleStopEval}
    />
  );
}
