import type { ConnectionStatus, RunInfo, CheckpointInfo } from "../storage";
import type { SidebarTab } from "../storage/atoms/uiAtoms";

const STATUS_COLORS: Record<string, string> = {
  training: "#22c55e",
  paused: "#eab308",
  complete: "#3b82f6",
  idle: "#6b7280",
};

const CONNECTION_COLORS: Record<ConnectionStatus, string> = {
  connected: "#22c55e",
  reconnecting: "#eab308",
  disconnected: "#ef4444",
};

const CONNECTION_LABELS: Record<ConnectionStatus, string> = {
  connected: "Connected",
  reconnecting: "Reconnecting...",
  disconnected: "Disconnected",
};

export interface HeaderProps {
  tab: SidebarTab;
  onTabSwitch: (tab: SidebarTab) => void;
  status: string;
  connectionStatus: ConnectionStatus;
  // Train mode
  runs: RunInfo[];
  activeRunId: number | null;
  onRunChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  step: number;
  maxSteps: number;
  elapsedMin: number;
  elapsedSec: number;
  // Inspect mode
  checkpoints: CheckpointInfo[];
  activeCheckpointId: number | null;
  onCheckpointChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  stopping: boolean;
  onStop: () => void;
}

export function Header({
  tab,
  onTabSwitch,
  status,
  connectionStatus,
  runs,
  activeRunId,
  onRunChange,
  step,
  maxSteps,
  elapsedMin,
  elapsedSec,
  checkpoints,
  activeCheckpointId,
  onCheckpointChange,
  stopping,
  onStop,
}: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-tabs">
          <button
            className={`header-tab ${tab === "train" ? "active" : ""}`}
            onClick={() => onTabSwitch("train")}
          >
            Train
          </button>
          <button
            className={`header-tab ${tab === "inspect" ? "active" : ""}`}
            onClick={() => onTabSwitch("inspect")}
          >
            Inspect
          </button>
        </div>

        {tab === "train" ? (
          <>
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
          </>
        ) : (
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
      <div className="header-right">
        <span className="connection-status">
          <span
            className="connection-dot"
            style={{ backgroundColor: CONNECTION_COLORS[connectionStatus] }}
          />
          {CONNECTION_LABELS[connectionStatus]}
        </span>
        <span
          className="status-badge"
          style={{ backgroundColor: STATUS_COLORS[status] }}
        >
          {status}
        </span>
      </div>
    </header>
  );
}
