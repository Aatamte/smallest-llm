import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { EvalResultInfo } from "../storage";
import type { AvailableModel, EvalStatus } from "../api/client";
import { CHART_COLORS, baseOpts } from "../types/chart";

// ── Helpers ─────────────────────────────────────────────

function CompareValue({
  value,
  baseline,
  lowerIsBetter,
  format,
}: {
  value: number | undefined;
  baseline: number;
  lowerIsBetter: boolean;
  format: (n: number) => string;
}) {
  if (value === undefined) return <span className="layer-val">—</span>;
  const better = lowerIsBetter ? value < baseline : value > baseline;
  return (
    <span className={`layer-val ${better ? "eval-better" : "eval-worse"}`}>
      {format(value)}
    </span>
  );
}

// ── Eval Controls ───────────────────────────────────────

const AVAILABLE_TASKS = ["perplexity", "blimp", "lambada"];

function EvalControls({
  availableModels,
  evalStatus,
  onRunEval,
}: {
  availableModels: AvailableModel[];
  evalStatus: EvalStatus;
  onRunEval: (modelName: string, tasks: string[]) => void;
}) {
  const [selectedModel, setSelectedModel] = useState("");
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
    if (!selectedModel || selectedTasks.size === 0) return;
    onRunEval(selectedModel, [...selectedTasks]);
  };

  return (
    <div className="panel">
      <h3 className="panel-title">Run Baseline Evaluation</h3>
      <div className="eval-controls">
        <div className="eval-controls-row">
          <label className="eval-control-label">Model</label>
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
            disabled={isRunning || !selectedModel || selectedTasks.size === 0}
          >
            {isRunning ? "Running..." : "Run Eval"}
          </button>
          {isRunning && evalStatus.model_name && (
            <span className="eval-status-text">
              Evaluating <strong>{evalStatus.model_name}</strong>
              {evalStatus.task && (
                <>
                  {" "}
                  — task: <strong>{evalStatus.task}</strong>
                </>
              )}
            </span>
          )}
          {evalStatus.status === "error" && evalStatus.error && (
            <span className="eval-status-text eval-error-text">
              Error: {evalStatus.error}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Perplexity Panel ────────────────────────────────────

function PerplexityPanel({
  latest,
  baselines,
}: {
  latest: EvalResultInfo | undefined;
  baselines: Record<string, Record<string, Record<string, number>>>;
}) {
  const metrics = latest?.metrics;
  const baselineNames = Object.keys(baselines);

  return (
    <div className="panel">
      <h3 className="panel-title">Perplexity</h3>
      {!metrics ? (
        <div className="panel-empty">No perplexity evals yet.</div>
      ) : (
        <>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Perplexity</div>
              <div className="metric-value">
                {metrics.perplexity?.toFixed(2) ?? "—"}
              </div>
              {latest && <div className="metric-sub">step {latest.step}</div>}
            </div>
            <div className="metric-card">
              <div className="metric-label">BPC</div>
              <div className="metric-value">
                {metrics.bpc?.toFixed(4) ?? "—"}
              </div>
              <div className="metric-sub">bits/char</div>
            </div>
          </div>
          {baselineNames.length > 0 && (
            <div className="eval-comparison-row">
              {baselineNames.map((name) => {
                const b = baselines[name]?.perplexity;
                if (!b) return null;
                return (
                  <div key={name} className="eval-baseline">
                    <span className="eval-baseline-name">{name}</span>
                    {b.perplexity != null && (
                      <span className="eval-baseline-value">
                        ppl={b.perplexity.toFixed(1)}
                      </span>
                    )}
                    {b.bpc != null && (
                      <span className="eval-baseline-value">
                        bpc={b.bpc.toFixed(3)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
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

function BlimpPanel({
  latest,
  baselines,
}: {
  latest: EvalResultInfo | undefined;
  baselines: Record<string, Record<string, Record<string, number>>>;
}) {
  const metrics = latest?.metrics;
  const baselineNames = Object.keys(baselines);

  return (
    <div className="panel">
      <h3 className="panel-title">BLiMP (Linguistic Minimal Pairs)</h3>
      {!metrics ? (
        <div className="panel-empty">No BLiMP evals yet.</div>
      ) : (
        <div className="layer-stats-container">
          <div className="eval-table-header">
            <span>Category</span>
            <span>Your Model</span>
            {baselineNames.map((n) => (
              <span key={n}>{n}</span>
            ))}
          </div>
          {BLIMP_CATEGORIES.map(({ key, label }) => (
            <div key={key} className="eval-table-row">
              <span className="layer-name">{label}</span>
              <span className="layer-val">
                {metrics[key] !== undefined
                  ? (metrics[key] * 100).toFixed(1) + "%"
                  : "—"}
              </span>
              {baselineNames.map((name) => {
                const bVal = baselines[name]?.blimp?.[key];
                return (
                  <CompareValue
                    key={name}
                    value={
                      metrics[key] !== undefined ? metrics[key] : undefined
                    }
                    baseline={bVal ?? 0}
                    lowerIsBetter={false}
                    format={(v) => (v * 100).toFixed(1) + "%"}
                  />
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Eval Over Steps Chart ───────────────────────────────

function EvalChart({ evals }: { evals: EvalResultInfo[] }) {
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!divRef.current) return;

    const perplexityEvals = evals
      .filter((e) => e.task === "perplexity" && e.metrics.perplexity != null)
      .sort((a, b) => a.step - b.step);

    const blimpEvals = evals
      .filter((e) => e.task === "blimp" && e.metrics.accuracy != null)
      .sort((a, b) => a.step - b.step);

    if (perplexityEvals.length === 0 && blimpEvals.length === 0) return;

    const allSteps = [
      ...new Set([
        ...perplexityEvals.map((e) => e.step),
        ...blimpEvals.map((e) => e.step),
      ]),
    ].sort((a, b) => a - b);

    const perpMap = new Map(
      perplexityEvals.map((e) => [e.step, e.metrics.perplexity]),
    );
    const blimpMap = new Map(
      blimpEvals.map((e) => [e.step, e.metrics.accuracy]),
    );

    const xData = Float64Array.from(allSteps);
    const perpData = allSteps.map((s) => perpMap.get(s) ?? null);
    const blimpData = allSteps.map((s) => blimpMap.get(s) ?? null);

    plotRef.current?.destroy();
    plotRef.current = new uPlot(
      {
        ...baseOpts(800, 300),
        series: [
          {},
          {
            label: "Perplexity",
            stroke: CHART_COLORS.trainLoss,
            width: 2,
            scale: "perp",
          },
          {
            label: "BLiMP Acc",
            stroke: CHART_COLORS.updateRatio,
            width: 2,
            scale: "acc",
          },
        ],
        scales: {
          x: { auto: true },
          perp: { auto: true },
          acc: { auto: true, range: [0, 1] },
        },
        axes: [
          {
            stroke: CHART_COLORS.text,
            grid: { stroke: CHART_COLORS.grid, width: 1 },
            font: "11px Inter, sans-serif",
          },
          {
            scale: "perp",
            stroke: CHART_COLORS.trainLoss,
            grid: { stroke: CHART_COLORS.grid, width: 1 },
            font: "11px Inter, sans-serif",
            label: "Perplexity",
          },
          {
            scale: "acc",
            side: 1,
            stroke: CHART_COLORS.updateRatio,
            grid: { show: false },
            font: "11px Inter, sans-serif",
            label: "Accuracy",
          },
        ],
      } as uPlot.Options,
      [xData as unknown as number[], perpData, blimpData] as uPlot.AlignedData,
      divRef.current,
    );

    return () => {
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, [evals]);

  if (evals.length === 0) return null;

  return (
    <div className="panel">
      <h3 className="panel-title">Eval Metrics Over Training</h3>
      <div ref={divRef} />
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────

export interface EvalPageProps {
  latestPerplexity: EvalResultInfo | undefined;
  latestBlimp: EvalResultInfo | undefined;
  evals: EvalResultInfo[];
  availableModels: AvailableModel[];
  evalStatus: EvalStatus;
  baselineEvals: Record<string, Record<string, Record<string, number>>>;
  onRunEval: (modelName: string, tasks: string[]) => void;
}

export function EvalPage({
  latestPerplexity,
  latestBlimp,
  evals,
  availableModels,
  evalStatus,
  baselineEvals,
  onRunEval,
}: EvalPageProps) {
  return (
    <main className="eval-layout">
      <EvalControls
        availableModels={availableModels}
        evalStatus={evalStatus}
        onRunEval={onRunEval}
      />
      <PerplexityPanel latest={latestPerplexity} baselines={baselineEvals} />
      <BlimpPanel latest={latestBlimp} baselines={baselineEvals} />
      <EvalChart evals={evals} />
    </main>
  );
}
