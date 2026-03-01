import { useEffect, useState } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import {
  fetchRunDetail,
  fetchCheckpoints,
  fetchWeights,
  type RunDetail,
  type Checkpoint,
  type WeightLayer,
} from "../api/client";
import { ModelsPage } from "../components/ModelsPage";

export function ModelsContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [weights, setWeights] = useState<WeightLayer[]>([]);
  const [loadingWeights, setLoadingWeights] = useState(false);

  useEffect(() => {
    if (runId == null) {
      setRun(null);
      setCheckpoints([]);
      setSelectedStep(null);
      setWeights([]);
      return;
    }
    fetchRunDetail(runId).then(setRun).catch((e) => console.warn("Failed to fetch run detail:", e));
    fetchCheckpoints(runId).then(setCheckpoints).catch((e) => console.warn("Failed to fetch checkpoints:", e));
  }, [runId]);

  function handleSelectStep(step: number | null) {
    setSelectedStep(step);
    setWeights([]);
    if (step != null && runId != null) {
      setLoadingWeights(true);
      fetchWeights(runId, step)
        .then(setWeights)
        .catch((e) => console.warn("Failed to fetch weights:", e))
        .finally(() => setLoadingWeights(false));
    }
  }

  return (
    <ModelsPage
      runId={runId}
      run={run}
      checkpoints={checkpoints}
      selectedStep={selectedStep}
      onSelectStep={handleSelectStep}
      weights={weights}
      loadingWeights={loadingWeights}
    />
  );
}
