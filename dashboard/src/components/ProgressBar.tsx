import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getCurrentStep, getMaxSteps, getLatestMetric, getTokensPerStep } from "../db/queries";

export function ProgressBar() {
  const runId = useAtomValue(activeRunIdAtom);

  const step = useQuery(useCallback(() => getCurrentStep(runId), [runId]));
  const maxSteps = useQuery(useCallback(() => getMaxSteps(runId), [runId]));
  const tokensSeen = useQuery(useCallback(() => getLatestMetric(runId, "tokensSeen"), [runId]));
  const tokensPerSec = useQuery(useCallback(() => getLatestMetric(runId, "tokensPerSec"), [runId]));
  const tokensPerStep = useQuery(useCallback(() => getTokensPerStep(runId), [runId]));

  const progress = maxSteps > 0 ? (step / maxSteps) * 100 : 0;

  const stepsLeft = maxSteps - step;
  const avgStepTime = tokensPerSec > 0 && tokensPerStep > 0 ? tokensPerStep / tokensPerSec : 0;
  const etaSeconds = Math.round(stepsLeft * avgStepTime);
  const etaMin = Math.floor(etaSeconds / 60);
  const etaSec = etaSeconds % 60;

  return (
    <footer className="progress-footer">
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="progress-info">
        <span>{progress.toFixed(1)}%</span>
        <span>ETA: {etaMin}m {etaSec.toString().padStart(2, "0")}s</span>
        <span>Tokens: {(tokensSeen / 1e6).toFixed(1)}M</span>
      </div>
    </footer>
  );
}
