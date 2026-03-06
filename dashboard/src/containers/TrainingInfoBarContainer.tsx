import { useCallback, useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import {
  statusAtom,
  textStateAtom,
  maxStepsAtom,
  startTimeAtom,
  activeRunIdAtom,
  availableCheckpointsAtom,
  activeCheckpointIdAtom,
} from "../storage";
import { fetchCheckpoints, stopRun } from "../api/client";
import { TrainingInfoBar } from "../components/TrainingInfoBar";
import { useQuery } from "../db/hooks";
import { getCurrentStep } from "../db/queries";

export function TrainingInfoBarContainer() {
  const status = useAtomValue(statusAtom);
  const textState = useAtomValue(textStateAtom);
  const step = useQuery(useCallback(() => getCurrentStep(), []));
  const maxSteps = useAtomValue(maxStepsAtom);
  const startTime = useAtomValue(startTimeAtom);
  const activeRunId = useAtomValue(activeRunIdAtom);
  const [checkpoints, setCheckpoints] = useAtom(availableCheckpointsAtom);
  const [activeCheckpointId, setActiveCheckpointId] = useAtom(activeCheckpointIdAtom);
  const [stopping, setStopping] = useState(false);

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
    <TrainingInfoBar
      status={status}
      textState={textState}
      step={step}
      maxSteps={maxSteps}
      elapsedMin={isNaN(elapsed) ? 0 : Math.floor(elapsed / 60)}
      elapsedSec={isNaN(elapsed) ? 0 : elapsed % 60}
      checkpoints={checkpoints}
      activeCheckpointId={activeCheckpointId}
      onCheckpointChange={handleCheckpointChange}
      stopping={stopping}
      onStop={handleStop}
    />
  );
}
