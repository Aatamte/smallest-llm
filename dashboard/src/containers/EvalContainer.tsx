import { useEffect, useCallback, useRef, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import { activeRunIdAtom, evalsAtom, type EvalResultInfo } from "../storage";
import {
  fetchEvals,
  fetchAvailableModels,
  fetchAllEvals,
  fetchEvalStatus,
  runEval,
  type AvailableModel,
  type EvalStatus,
} from "../api/client";
import { EvalPage } from "../components/EvalPage";

function latestByTask(evals: EvalResultInfo[]): Record<string, EvalResultInfo> {
  const latest: Record<string, EvalResultInfo> = {};
  for (const e of evals) {
    if (!latest[e.task] || e.step > latest[e.task].step) {
      latest[e.task] = e;
    }
  }
  return latest;
}

export function EvalContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const [evals, setEvals] = useAtom(evalsAtom);

  // HF eval state
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [evalStatus, setEvalStatus] = useState<EvalStatus>({
    status: "idle",
    model_name: null,
    task: null,
    error: null,
  });
  const [baselineEvals, setBaselineEvals] = useState<
    Record<string, Record<string, Record<string, number>>>
  >({});

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch run-specific evals
  useEffect(() => {
    if (runId != null) {
      fetchEvals(runId).then(setEvals).catch((e) => console.warn("Failed to fetch evals:", e));
    } else {
      setEvals([]);
    }
  }, [runId, setEvals]);

  // Fetch available models on mount
  useEffect(() => {
    fetchAvailableModels()
      .then(setAvailableModels)
      .catch((e) => console.warn("Failed to fetch available models:", e));
  }, []);

  // Load baseline evals from DB (all model evals not tied to a run)
  const loadBaselines = useCallback(() => {
    fetchAllEvals()
      .then((allEvals) => {
        const baselines: Record<string, Record<string, Record<string, number>>> = {};
        for (const e of allEvals) {
          const modelName = e.model_name;
          if (!modelName) continue;
          if (!baselines[modelName]) baselines[modelName] = {};
          baselines[modelName][e.task] = e.metrics;
        }
        setBaselineEvals(baselines);
      })
      .catch((e) => console.warn("Failed to fetch baselines:", e));
  }, []);

  useEffect(() => {
    loadBaselines();
  }, [loadBaselines]);

  // Poll eval status while running
  useEffect(() => {
    if (evalStatus.status === "running") {
      pollRef.current = setInterval(() => {
        fetchEvalStatus().then((s) => {
          setEvalStatus(s);
          if (s.status !== "running") {
            // Eval finished — reload baselines
            loadBaselines();
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
  }, [evalStatus.status, loadBaselines]);

  const handleRunEval = useCallback(
    async (modelName: string, tasks: string[]) => {
      try {
        await runEval(modelName, tasks);
        setEvalStatus({ status: "running", model_name: modelName, task: null, error: null });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setEvalStatus({ status: "error", model_name: null, task: null, error: msg });
      }
    },
    [],
  );

  const latest = latestByTask(evals);

  return (
    <EvalPage
      latestPerplexity={latest["perplexity"]}
      latestBlimp={latest["blimp"]}
      evals={evals}
      availableModels={availableModels}
      evalStatus={evalStatus}
      baselineEvals={baselineEvals}
      onRunEval={handleRunEval}
    />
  );
}
