import { useEffect, useState } from "react";
import { useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useRuns, useRunState, useRunModelName, useRunMaxSteps, useRunMaxFlops, useRunEvalInterval, useRunStartTime, useCurrentStep, useLatestMetric } from "../db/hooks";
import { stopRun } from "../api/client";
import { TrainingInfoBar } from "../components/TrainingInfoBar";

export function TrainingInfoBarContainer() {
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const [stopping, setStopping] = useState(false);
  const runs = useRuns();

  const rs = useRunState(activeRunId);
  const modelName = useRunModelName(activeRunId);
  const maxSteps = useRunMaxSteps(activeRunId);
  const maxFlops = useRunMaxFlops(activeRunId);
  const startTimeStr = useRunStartTime(activeRunId);
  const step = useCurrentStep(activeRunId);
  const tokensPerSec = useLatestMetric("tokensPerSec", activeRunId);
  const trainLoss = useLatestMetric("trainLoss", activeRunId);
  const bpc = useLatestMetric("bpc", activeRunId);
  const flopsPct = useLatestMetric("flopsPct", activeRunId);
  const evalInterval = useRunEvalInterval(activeRunId);

  const status = rs?.status ?? "idle";
  const nextEvalStep = evalInterval > 0 && step > 0
    ? Math.ceil(step / evalInterval) * evalInterval
    : null;
  const useFlopsBudget = (maxFlops ?? 0) > 0;
  const pct = useFlopsBudget
    ? Math.min(flopsPct ?? 0, 100)
    : (maxSteps ?? 0) > 0 ? Math.min(((step ?? 0) / maxSteps) * 100, 100) : 0;
  const startTime = startTimeStr
    ? new Date(startTimeStr.endsWith("Z") ? startTimeStr : startTimeStr + "Z").getTime()
    : Date.now();
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
      maxFlops={maxFlops}
      flopsPct={flopsPct}
      elapsedMin={isNaN(elapsed) ? 0 : Math.floor(elapsed / 60)}
      elapsedSec={isNaN(elapsed) ? 0 : elapsed % 60}
      tokensPerSec={tokensPerSec}
      trainLoss={trainLoss}
      bpc={bpc}
      etaSeconds={status === "training" && pct > 0 ? Math.round(elapsed * (100 - pct) / pct) : null}
      nextEvalStep={nextEvalStep}
      textState={rs?.text_state ?? ""}
      stopping={stopping}
      onStop={handleStop}
    />
  );
}
