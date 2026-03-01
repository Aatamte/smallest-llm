import { useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import {
  activeRunIdAtom,
  availableCheckpointsAtom,
  activeCheckpointIdAtom,
} from "../storage";
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
  const [weights, setWeights] = useState<WeightLayer[]>([]);
  const [loadingWeights, setLoadingWeights] = useState(false);

  const availableCheckpoints = useAtomValue(availableCheckpointsAtom);
  const [activeCheckpointId, setActiveCheckpointId] = useAtom(activeCheckpointIdAtom);

  // Derive the selected step from the active checkpoint id
  const selectedStep =
    availableCheckpoints.find((c) => c.id === activeCheckpointId)?.step ?? null;

  useEffect(() => {
    if (runId == null) {
      setRun(null);
      setCheckpoints([]);
      setWeights([]);
      return;
    }
    fetchRunDetail(runId).then(setRun).catch((e) => console.warn("Failed to fetch run detail:", e));
    fetchCheckpoints(runId).then(setCheckpoints).catch((e) => console.warn("Failed to fetch checkpoints:", e));
  }, [runId]);

  // Fetch weights when the selected checkpoint changes
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
      // Find the checkpoint id for this step
      const cp = availableCheckpoints.find((c) => c.step === step);
      setActiveCheckpointId(cp?.id ?? null);
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
