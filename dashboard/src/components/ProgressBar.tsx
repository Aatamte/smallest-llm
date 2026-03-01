import { useAtomValue } from "jotai";
import { currentStepAtom, maxStepsAtom, tokensSeenAtom, tokensPerSecAtom } from "../storage";

export function ProgressBar() {
  const step = useAtomValue(currentStepAtom);
  const maxSteps = useAtomValue(maxStepsAtom);
  const tokensSeen = useAtomValue(tokensSeenAtom);
  const tokensPerSec = useAtomValue(tokensPerSecAtom);

  const progress = maxSteps > 0 ? (step / maxSteps) * 100 : 0;

  const stepsLeft = maxSteps - step;
  const avgStepTime = tokensPerSec > 0 ? (64 * 256) / tokensPerSec : 0;
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
