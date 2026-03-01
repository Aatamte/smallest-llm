import { useState } from "react";
import type { AvailableModel, EvalStatus, RunEvalRequest } from "../api/client";
import type { GroupedEvals, CheckpointOption } from "../containers/EvalContainer";

// ── Eval Progress ───────────────────────────────────────

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function EvalProgress({ evalStatus }: { evalStatus: EvalStatus }) {
  const { model_name, task, task_index, task_count, current_sample, total_samples, started_at } = evalStatus;
  const pct = total_samples > 0 ? (current_sample / total_samples) * 100 : 0;

  let elapsed = "";
  let eta = "";
  if (started_at) {
    const elapsedSecs = (Date.now() / 1000) - started_at;
    elapsed = formatDuration(elapsedSecs);

    if (current_sample > 0 && total_samples > 0 && current_sample < total_samples) {
      const rate = current_sample / elapsedSecs;
      const remaining = (total_samples - current_sample) / rate;
      eta = formatDuration(remaining);
    }
  }

  return (
    <div className="eval-progress">
      <div className="eval-progress-header">
        <span className="eval-progress-model">{model_name}</span>
        {task && (
          <span className="eval-progress-task">
            Task {task_index + 1}/{task_count}: {task}
          </span>
        )}
        <span className="eval-progress-elapsed">
          {elapsed}{eta && ` · ETA ${eta}`}
        </span>
      </div>
      <div className="eval-progress-bar">
        <div className="eval-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {total_samples > 0 && (
        <div className="eval-progress-text">
          {current_sample} / {total_samples} samples ({pct.toFixed(0)}%)
        </div>
      )}
    </div>
  );
}

// ── Eval Controls ───────────────────────────────────────

const AVAILABLE_TASKS = ["perplexity", "blimp", "lambada"];

function EvalControls({
  availableModels,
  checkpointOptions,
  evalStatus,
  onRunEval,
  onStopEval,
}: {
  availableModels: AvailableModel[];
  checkpointOptions: CheckpointOption[];
  evalStatus: EvalStatus;
  onRunEval: (request: RunEvalRequest) => void;
  onStopEval: () => void;
}) {
  const [source, setSource] = useState<"hf" | "checkpoint">("hf");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(
    new Set(["perplexity", "blimp"]),
  );

  const isRunning = evalStatus.status === "running";

  const toggleTask = (task: string) => {
    setSelectedTasks((prev) => {
      const next = new Set(prev);
      if (next.has(task)) next.delete(task);
      else next.add(task);
      return next;
    });
  };

  const handleRun = () => {
    if (selectedTasks.size === 0) return;
    if (source === "hf") {
      if (!selectedModel) return;
      onRunEval({ source: "hf", model_name: selectedModel, tasks: [...selectedTasks] });
    } else {
      if (!selectedCheckpoint) return;
      const cp = checkpointOptions.find(
        (c) => `${c.runId}-${c.step}` === selectedCheckpoint,
      );
      if (!cp) return;
      onRunEval({ source: "checkpoint", run_id: cp.runId, step: cp.step, tasks: [...selectedTasks] });
    }
  };

  const canRun =
    !isRunning &&
    selectedTasks.size > 0 &&
    (source === "hf" ? !!selectedModel : !!selectedCheckpoint);

  return (
    <div className="panel">
      <h3 className="panel-title">Run Evaluation</h3>
      <div className="eval-controls">
        <div className="eval-controls-row">
          <label className="eval-control-label">Source</label>
          <div className="eval-source-toggle">
            <button
              className={`eval-source-btn ${source === "hf" ? "active" : ""}`}
              onClick={() => setSource("hf")}
              disabled={isRunning}
            >
              HF Model
            </button>
            <button
              className={`eval-source-btn ${source === "checkpoint" ? "active" : ""}`}
              onClick={() => setSource("checkpoint")}
              disabled={isRunning}
            >
              Checkpoint
            </button>
          </div>
        </div>
        <div className="eval-controls-row">
          <label className="eval-control-label">
            {source === "hf" ? "Model" : "Checkpoint"}
          </label>
          {source === "hf" ? (
            <select
              className="eval-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isRunning}
            >
              <option value="">Select a model...</option>
              {availableModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.hf_id})
                </option>
              ))}
            </select>
          ) : (
            <select
              className="eval-select"
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              disabled={isRunning}
            >
              <option value="">Select a checkpoint...</option>
              {checkpointOptions.map((cp) => (
                <option key={`${cp.runId}-${cp.step}`} value={`${cp.runId}-${cp.step}`}>
                  {cp.label}
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="eval-controls-row">
          <label className="eval-control-label">Tasks</label>
          <div className="eval-task-checkboxes">
            {AVAILABLE_TASKS.map((task) => (
              <label key={task} className="eval-checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedTasks.has(task)}
                  onChange={() => toggleTask(task)}
                  disabled={isRunning}
                />
                {task}
              </label>
            ))}
          </div>
        </div>
        <div className="eval-controls-row">
          <button
            className="eval-run-btn"
            onClick={handleRun}
            disabled={!canRun}
          >
            {isRunning ? "Running..." : "Run Eval"}
          </button>
          {isRunning && (
            <button
              className="eval-stop-btn"
              onClick={onStopEval}
            >
              Stop
            </button>
          )}
          {evalStatus.status === "error" && evalStatus.error && (
            <span className="eval-status-text eval-error-text">
              Error: {evalStatus.error}
            </span>
          )}
          {evalStatus.status === "stopped" && (
            <span className="eval-status-text eval-stopped-text">
              Evaluation stopped
            </span>
          )}
        </div>
        {isRunning && evalStatus.model_name && (
          <EvalProgress evalStatus={evalStatus} />
        )}
      </div>
    </div>
  );
}

// ── Leaderboard ────────────────────────────────────────

function Leaderboard({ allEvals }: { allEvals: GroupedEvals }) {
  const modelNames = Object.keys(allEvals);

  if (modelNames.length === 0) {
    return null;
  }

  // Sort by perplexity (lower is better), models without perplexity go last
  const sorted = [...modelNames].sort((a, b) => {
    const pa = allEvals[a].perplexity?.perplexity ?? Infinity;
    const pb = allEvals[b].perplexity?.perplexity ?? Infinity;
    return pa - pb;
  });

  const cols = "minmax(120px, 1.5fr) repeat(4, 1fr)";

  return (
    <div className="panel">
      <h3 className="panel-title">Leaderboard</h3>
      <div className="eval-table">
        <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
          <span>Model</span>
          <span>Perplexity</span>
          <span>BPC</span>
          <span>BLiMP Acc</span>
          <span>LAMBADA Acc</span>
        </div>
        {sorted.map((name) => {
          const ppl = allEvals[name].perplexity;
          const blimp = allEvals[name].blimp;
          const lambada = allEvals[name].lambada;
          return (
            <div key={name} className="eval-table-row" style={{ gridTemplateColumns: cols }}>
              <span className="eval-cell-model">{name}</span>
              <span className="eval-cell-val">
                {ppl?.perplexity != null ? ppl.perplexity.toFixed(2) : "—"}
              </span>
              <span className="eval-cell-val">
                {ppl?.bpc != null ? ppl.bpc.toFixed(4) : "—"}
              </span>
              <span className="eval-cell-val">
                {blimp?.accuracy != null ? (blimp.accuracy * 100).toFixed(1) + "%" : "—"}
              </span>
              <span className="eval-cell-val">
                {lambada?.accuracy != null ? (lambada.accuracy * 100).toFixed(1) + "%" : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Perplexity Panel ────────────────────────────────────

function PerplexityPanel({ allEvals }: { allEvals: GroupedEvals }) {
  const modelNames = Object.keys(allEvals).filter(
    (name) => allEvals[name].perplexity != null,
  );

  if (modelNames.length === 0) {
    return (
      <div className="panel">
        <h3 className="panel-title">Perplexity</h3>
        <div className="panel-empty">No perplexity evals yet.</div>
      </div>
    );
  }

  const cols = "minmax(120px, 1.5fr) 1fr 1fr";

  return (
    <div className="panel">
      <h3 className="panel-title">Perplexity</h3>
      <div className="eval-table">
        <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
          <span>Model</span>
          <span>Perplexity</span>
          <span>BPC</span>
        </div>
        {modelNames.map((name) => {
          const m = allEvals[name].perplexity;
          return (
            <div key={name} className="eval-table-row" style={{ gridTemplateColumns: cols }}>
              <span className="eval-cell-model">{name}</span>
              <span className="eval-cell-val">
                {m?.perplexity != null ? m.perplexity.toFixed(2) : "—"}
              </span>
              <span className="eval-cell-val">
                {m?.bpc != null ? m.bpc.toFixed(4) : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── BLiMP Panel ─────────────────────────────────────────

const BLIMP_CATEGORIES = [
  { key: "accuracy", label: "Overall" },
  { key: "accuracy_morphology", label: "Morphology" },
  { key: "accuracy_syntax", label: "Syntax" },
  { key: "accuracy_semantics", label: "Semantics" },
  { key: "accuracy_syntax_semantics", label: "Syntax-Semantics" },
];

function BlimpPanel({ allEvals }: { allEvals: GroupedEvals }) {
  const modelNames = Object.keys(allEvals).filter(
    (name) => allEvals[name].blimp != null,
  );

  if (modelNames.length === 0) {
    return (
      <div className="panel">
        <h3 className="panel-title">BLiMP</h3>
        <div className="panel-empty">No BLiMP evals yet.</div>
      </div>
    );
  }

  const cols = `minmax(120px, 1.5fr) repeat(${modelNames.length}, 1fr)`;

  return (
    <div className="panel">
      <h3 className="panel-title">BLiMP</h3>
      <div className="eval-table">
        <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
          <span>Category</span>
          {modelNames.map((name) => (
            <span key={name}>{name}</span>
          ))}
        </div>
        {BLIMP_CATEGORIES.map(({ key, label }) => (
          <div key={key} className="eval-table-row" style={{ gridTemplateColumns: cols }}>
            <span className="eval-cell-model">{label}</span>
            {modelNames.map((name) => {
              const val = allEvals[name].blimp?.[key];
              return (
                <span key={name} className="eval-cell-val">
                  {val != null ? (val * 100).toFixed(1) + "%" : "—"}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── LAMBADA Panel ───────────────────────────────────────

function LambadaPanel({ allEvals }: { allEvals: GroupedEvals }) {
  const modelNames = Object.keys(allEvals).filter(
    (name) => allEvals[name].lambada != null,
  );

  if (modelNames.length === 0) {
    return (
      <div className="panel">
        <h3 className="panel-title">LAMBADA</h3>
        <div className="panel-empty">No LAMBADA evals yet.</div>
      </div>
    );
  }

  const cols = "minmax(120px, 1.5fr) 1fr 1fr";

  return (
    <div className="panel">
      <h3 className="panel-title">LAMBADA</h3>
      <div className="eval-table">
        <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
          <span>Model</span>
          <span>Accuracy</span>
          <span>Target PPL</span>
        </div>
        {modelNames.map((name) => {
          const m = allEvals[name].lambada;
          return (
            <div key={name} className="eval-table-row" style={{ gridTemplateColumns: cols }}>
              <span className="eval-cell-model">{name}</span>
              <span className="eval-cell-val">
                {m?.accuracy != null ? (m.accuracy * 100).toFixed(1) + "%" : "—"}
              </span>
              <span className="eval-cell-val">
                {m?.target_perplexity != null ? m.target_perplexity.toFixed(2) : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────

export interface EvalPageProps {
  allEvals: GroupedEvals;
  availableModels: AvailableModel[];
  checkpointOptions: CheckpointOption[];
  evalStatus: EvalStatus;
  onRunEval: (request: RunEvalRequest) => void;
  onStopEval: () => void;
}

export function EvalPage({
  allEvals,
  availableModels,
  checkpointOptions,
  evalStatus,
  onRunEval,
  onStopEval,
}: EvalPageProps) {
  return (
    <main className="eval-layout">
      <EvalControls
        availableModels={availableModels}
        checkpointOptions={checkpointOptions}
        evalStatus={evalStatus}
        onRunEval={onRunEval}
        onStopEval={onStopEval}
      />
      <Leaderboard allEvals={allEvals} />
      <PerplexityPanel allEvals={allEvals} />
      <BlimpPanel allEvals={allEvals} />
      <LambadaPanel allEvals={allEvals} />
    </main>
  );
}
