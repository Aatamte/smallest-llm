import { useCallback, useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import {
  activeRunIdAtom,
  activeCheckpointIdAtom,
} from "../storage";
import { useQuery } from "../db/hooks";
import { getCheckpoints } from "../db/queries";
import {
  fetchRunDetail,
  fetchWeights,
  type RunDetail,
  type WeightLayer,
} from "../api/client";
import { ModelsPage } from "../components/ModelsPage";

export function ModelsContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [restCheckpoints, setRestCheckpoints] = useState<{ id: number; run_id: number; step: number; path: string; metrics: Record<string, number>; is_best: number; created_at: string }[]>([]);
  const [weights, setWeights] = useState<WeightLayer[]>([]);
  const [loadingWeights, setLoadingWeights] = useState(false);

  const checkpoints = useQuery(useCallback(() => getCheckpoints(runId), [runId]));
  const [activeCheckpointId, setActiveCheckpointId] = useAtom(activeCheckpointIdAtom);

  const selectedStep =
    checkpoints.find((c) => c.id === activeCheckpointId)?.step ?? null;

  useEffect(() => {
    if (runId == null) {
      setRun(null);
      setRestCheckpoints([]);
      setWeights([]);
      return;
    }
    fetchRunDetail(runId).then(setRun).catch((e) => console.warn("Failed to fetch run detail:", e));
  }, [runId]);

  useEffect(() => {
    if (selectedStep == null || runId == null) {
      setWeights([]);
      return;
    }
    setLoadingWeights(true);
    fetchWeights(runId, selectedStep)
      .then(setWeights)
      .catch((e) => console.warn("Failed to fetch weights:", e))
      .finally(() => setLoadingWeights(false));
  }, [selectedStep, runId]);

  function handleSelectStep(step: number | null) {
    if (step == null) {
      setActiveCheckpointId(null);
    } else {
      const cp = checkpoints.find((c) => c.step === step);
      setActiveCheckpointId(cp?.id ?? null);
    }
  }

  return (
    <ModelsPage
      runId={runId}
      run={run}
      checkpoints={restCheckpoints.length > 0 ? restCheckpoints : checkpoints.map((c) => ({
        ...c,
        metrics: {},
        created_at: "",
      }))}
      selectedStep={selectedStep}
      onSelectStep={handleSelectStep}
      weights={weights}
      loadingWeights={loadingWeights}
    />
  );
}
