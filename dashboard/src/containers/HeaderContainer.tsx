import { useEffect, useState } from "react";
import { useAtomValue, useAtom, useSetAtom } from "jotai";
import {
  statusAtom,
  currentStepAtom,
  maxStepsAtom,
  startTimeAtom,
  connectionStatusAtom,
  activeRunIdAtom,
  availableRunsAtom,
  availableCheckpointsAtom,
  activeCheckpointIdAtom,
  resetAtom,
} from "../storage";
import {
  sidebarTabAtom,
  navigateTo,
  type SidebarTab,
} from "../storage/atoms/uiAtoms";
import { fetchRuns, fetchCheckpoints, stopRun } from "../api/client";
import { Header } from "../components/Header";

export function HeaderContainer() {
  const status = useAtomValue(statusAtom);
  const step = useAtomValue(currentStepAtom);
  const maxSteps = useAtomValue(maxStepsAtom);
  const startTime = useAtomValue(startTimeAtom);
  const connectionStatus = useAtomValue(connectionStatusAtom);
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const [runs, setRuns] = useAtom(availableRunsAtom);
  const [checkpoints, setCheckpoints] = useAtom(availableCheckpointsAtom);
  const [activeCheckpointId, setActiveCheckpointId] = useAtom(activeCheckpointIdAtom);
  const reset = useSetAtom(resetAtom);
  const [tab, setTab] = useAtom(sidebarTabAtom);
  const [stopping, setStopping] = useState(false);

  useEffect(() => {
    fetchRuns().then(setRuns).catch((e) => console.warn("Failed to fetch runs:", e));
  }, [setRuns]);

  useEffect(() => {
    if (activeRunId != null) {
      fetchCheckpoints(activeRunId)
        .then((cps) =>
          setCheckpoints(
            cps.map((c) => ({
              id: c.id,
              run_id: c.run_id,
              step: c.step,
              path: c.path,
              is_best: c.is_best,
            })),
          ),
        )
        .catch(() => setCheckpoints([]));
    } else {
      setCheckpoints([]);
    }
  }, [activeRunId, setCheckpoints]);

  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - startTime) / 1000));

  useEffect(() => {
    setElapsed(Math.floor((Date.now() - startTime) / 1000));
    if (status !== "training") return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [startTime, status]);

  function handleTabSwitch(newTab: SidebarTab) {
    setTab(newTab);
    navigateTo(newTab === "train" ? "metrics" : "models");
  }

  function handleRunChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const val = e.target.value;
    reset();
    if (val !== "") setActiveRunId(Number(val));
  }

  function handleCheckpointChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const val = e.target.value;
    setActiveCheckpointId(val === "" ? null : Number(val));
  }

  async function handleStop() {
    if (activeRunId == null) return;
    setStopping(true);
    try {
      await stopRun(activeRunId);
    } catch {
      // ignored — WS status update will reflect actual state
    } finally {
      setStopping(false);
    }
  }

  return (
    <Header
      tab={tab}
      onTabSwitch={handleTabSwitch}
      status={status}
      connectionStatus={connectionStatus}
      runs={runs}
      activeRunId={activeRunId}
      onRunChange={handleRunChange}
      step={step}
      maxSteps={maxSteps}
      elapsedMin={Math.floor(elapsed / 60)}
      elapsedSec={elapsed % 60}
      checkpoints={checkpoints}
      activeCheckpointId={activeCheckpointId}
      onCheckpointChange={handleCheckpointChange}
      stopping={stopping}
      onStop={handleStop}
    />
  );
}
