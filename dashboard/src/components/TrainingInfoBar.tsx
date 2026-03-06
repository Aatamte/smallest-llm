const STATUS_COLORS: Record<string, string> = {
  training: "#22c55e",
  paused: "#eab308",
  complete: "#3b82f6",
  idle: "#6b7280",
};

export interface RunOption {
  id: number;
  name: string;
}

export interface TrainingInfoBarProps {
  runs: RunOption[];
  activeRunId: number | null;
  onRunChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  status: string;
  modelName: string;
  stageIndex: number;
  stageName: string;
  totalStages: number;
  dataset: string;
  step: number;
  maxSteps: number;
  elapsedMin: number;
  elapsedSec: number;
  tokensPerSec: number;
  trainLoss: number;
  bpc: number;
  stopping: boolean;
  onStop: () => void;
}

const fmtNum = (n: number) => {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
};

export function TrainingInfoBar({
  runs,
  activeRunId,
  onRunChange,
  status,
  modelName,
  stageIndex,
  stageName,
  totalStages,
  dataset,
  step,
  maxSteps,
  elapsedMin,
  elapsedSec,
  tokensPerSec,
  trainLoss,
  bpc,
  stopping,
  onStop,
}: TrainingInfoBarProps) {
  const pct = maxSteps > 0 ? Math.min((step / maxSteps) * 100, 100) : 0;

  return (
    <div className="train-info-bar">
      {/* Row 1: run selector + metadata pills */}
      <div className="train-info-row">
        <select
          className="run-selector"
          value={activeRunId ?? ""}
          onChange={onRunChange}
        >
          <option value="">No run</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              #{r.id} {r.name}
            </option>
          ))}
        </select>
        <span
          className="train-info-status"
          style={{ backgroundColor: STATUS_COLORS[status] ?? "#6b7280" }}
        >
          {status}
        </span>
        {modelName && <span className="train-info-pill">{modelName}</span>}
        {totalStages > 0 && (
          <span className="train-info-pill">
            Stage {stageIndex + 1}/{totalStages}: {stageName}
          </span>
        )}
        {dataset && <span className="train-info-pill">{dataset}</span>}
        <div style={{ flex: 1 }} />
        <span className="train-info-stat">
          {elapsedMin}m {elapsedSec.toString().padStart(2, "0")}s
        </span>
        {status === "training" && (
          <button
            className="train-info-stop"
            onClick={onStop}
            disabled={stopping}
          >
            {stopping ? "Stopping..." : "Stop"}
          </button>
        )}
      </div>

      {/* Row 2: progress bar + metrics */}
      <div className="train-info-row">
        <div className="train-info-progress">
          <div className="train-info-progress-track">
            <div
              className="train-info-progress-fill"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
        <span className="train-info-stat">
          <strong>{step.toLocaleString()}</strong> / {maxSteps.toLocaleString()}
        </span>
        <span className="train-info-stat">{pct.toFixed(1)}%</span>
        {tokensPerSec > 0 && (
          <span className="train-info-stat">{fmtNum(tokensPerSec)} tok/s</span>
        )}
        {trainLoss > 0 && (
          <span className="train-info-stat">loss {trainLoss.toFixed(3)}</span>
        )}
        {bpc > 0 && (
          <span className="train-info-stat">bpc {bpc.toFixed(3)}</span>
        )}
      </div>
    </div>
  );
}
