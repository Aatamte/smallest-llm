import { useCallback, useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getRuns, getRunState, getRunModelName, getCurrentStep, getLatestMetric } from "../db/queries";
import { stopRun } from "../api/client";
import { TrainingInfoBar } from "../components/TrainingInfoBar";

export function TrainingInfoBarContainer() {
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const [stopping, setStopping] = useState(false);
  const runs = useQuery(useCallback(() => getRuns(), []));

  const rs = useQuery(useCallback(() => getRunState(activeRunId), [activeRunId]));
  const modelName = useQuery(useCallback(() => getRunModelName(activeRunId), [activeRunId]));
  const step = useQuery(useCallback(() => getCurrentStep(activeRunId), [activeRunId]));
  const tokensPerSec = useQuery(useCallback(() => getLatestMetric(activeRunId, "tokensPerSec"), [activeRunId]));
  const trainLoss = useQuery(useCallback(() => getLatestMetric(activeRunId, "trainLoss"), [activeRunId]));
  const bpc = useQuery(useCallback(() => getLatestMetric(activeRunId, "bpc"), [activeRunId]));

  const status = rs?.status ?? "idle";
  const startTime = rs?.start_time ? new Date(rs.start_time).getTime() : Date.now();
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
      maxSteps={rs?.max_steps ?? 0}
      elapsedMin={isNaN(elapsed) ? 0 : Math.floor(elapsed / 60)}
      elapsedSec={isNaN(elapsed) ? 0 : elapsed % 60}
      tokensPerSec={tokensPerSec}
      trainLoss={trainLoss}
      bpc={bpc}
      stopping={stopping}
      onStop={handleStop}
    />
  );
}
