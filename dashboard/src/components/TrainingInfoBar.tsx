import type { CheckpointInfo } from "../storage";

export interface TrainingInfoBarProps {
  status: string;
  textState: string;
  step: number;
  maxSteps: number;
  elapsedMin: number;
  elapsedSec: number;
  checkpoints: CheckpointInfo[];
  activeCheckpointId: number | null;
  onCheckpointChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  stopping: boolean;
  onStop: () => void;
}

export function TrainingInfoBar({
  status,
  textState,
  step,
  maxSteps,
  elapsedMin,
  elapsedSec,
  checkpoints,
  activeCheckpointId,
  onCheckpointChange,
  stopping,
  onStop,
}: TrainingInfoBarProps) {
  return (
    <div className="training-info-bar">
      <div className="training-info-left">
        {textState && (
          <span className="header-text-state">{textState}</span>
        )}
      </div>
      <div className="training-info-right">
        <span className="header-stat">
          Step <strong>{step.toLocaleString()}</strong> /{" "}
          {maxSteps.toLocaleString()}
        </span>
        <span className="header-stat">
          {elapsedMin}m {elapsedSec.toString().padStart(2, "0")}s
        </span>
        {status === "training" && (
          <button
            className="header-stop-btn"
            onClick={onStop}
            disabled={stopping}
          >
            {stopping ? "Stopping..." : "Stop"}
          </button>
        )}
        {checkpoints.length > 0 && (
          <select
            className="run-selector"
            value={activeCheckpointId ?? ""}
            onChange={onCheckpointChange}
          >
            <option value="">No checkpoint</option>
            {checkpoints.map((cp) => (
              <option key={cp.id} value={cp.id}>
                step {cp.step}
                {cp.is_best ? " (best)" : ""}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
