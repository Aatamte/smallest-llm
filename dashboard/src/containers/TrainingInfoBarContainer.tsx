import { useEffect, useState } from "react";
import { useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useRuns, useRunState, useRunModelName, useRunMaxSteps, useRunStartTime, useCurrentStep, useLatestMetric } from "../db/hooks";
import { stopRun } from "../api/client";
import { TrainingInfoBar } from "../components/TrainingInfoBar";

export function TrainingInfoBarContainer() {
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const [stopping, setStopping] = useState(false);
  const runs = useRuns();

  const rs = useRunState(activeRunId);
  const modelName = useRunModelName(activeRunId);
  const maxSteps = useRunMaxSteps(activeRunId);
  const startTimeStr = useRunStartTime(activeRunId);
  const step = useCurrentStep(activeRunId);
  const tokensPerSec = useLatestMetric("tokensPerSec", activeRunId);
  const trainLoss = useLatestMetric("trainLoss", activeRunId);
  const bpc = useLatestMetric("bpc", activeRunId);

  const status = rs?.status ?? "idle";
  const startTime = startTimeStr ? new Date(startTimeStr).getTime() : Date.now();
  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - startTime) / 1000));

  useEffect(() => {
    setElapsed(Math.floor((Date.now() - startTime) / 1000));
    if (status !== "training") return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [startTime, status]);

  async function handleStop() {
    if (activeRunId == null) return;
    setStopping(true);
    try {
      await stopRun(activeRunId);
    } catch {
      // ignored
    } finally {
      setStopping(false);
    }
  }

  function handleRunChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const val = e.target.value;
    setActiveRunId(val === "" ? null : Number(val));
  }

  return (
    <TrainingInfoBar
      runs={runs}
      activeRunId={activeRunId}
      onRunChange={handleRunChange}
      status={status}
      modelName={modelName}
      stageIndex={rs?.stage_index ?? 0}
      stageName={rs?.stage_name ?? ""}
      totalStages={rs?.total_stages ?? 0}
      dataset={rs?.dataset ?? ""}
      step={step}
      maxSteps={maxSteps}
      elapsedMin={isNaN(elapsed) ? 0 : Math.floor(elapsed / 60)}
      elapsedSec={isNaN(elapsed) ? 0 : elapsed % 60}
      tokensPerSec={tokensPerSec}
      trainLoss={trainLoss}
      bpc={bpc}
      textState={rs?.text_state ?? ""}
      stopping={stopping}
      onStop={handleStop}
    />
  );
}
