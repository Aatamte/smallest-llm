import type { RunDetail, Checkpoint, WeightLayer } from "../api/client";
import { WeightHeatmap } from "./WeightHeatmap";

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

const TRANSFORMER_LAYERS = (n: number) => [
  { label: "Token + Position Embedding", highlight: false },
  ...Array.from({ length: n }, (_, i) => ({
    label: `TransformerBlock ${i}  (Attn + FFN)`,
    highlight: true,
  })),
  { label: "LayerNorm", highlight: false },
  { label: "Linear (head)", highlight: false },
];

const MAMBA_LAYERS = (n: number) => [
  { label: "Token Embedding", highlight: false },
  ...Array.from({ length: n }, (_, i) => ({
    label: `MambaBlock ${i}  (SSM + Conv1d)`,
    highlight: true,
  })),
  { label: "RMSNorm", highlight: false },
  { label: "Linear (head)", highlight: false },
];

function ModelLayers({
  modelName,
  extraArgs,
}: {
  modelName: string;
  extraArgs: Record<string, unknown>;
}) {
  const nLayers = Number(extraArgs.n_layers ?? extraArgs.num_layers ?? 4);
  const layers =
    modelName === "mamba" ? MAMBA_LAYERS(nLayers) : TRANSFORMER_LAYERS(nLayers);

  return (
    <div className="model-layers-stack">
      {layers.map((layer, i) => (
        <div key={i}>
          {i > 0 && <div className="model-layer-arrow">|</div>}
          <div
            className={`model-layer-block${layer.highlight ? " highlight" : ""}`}
          >
            {layer.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function CheckpointsTable({ checkpoints }: { checkpoints: Checkpoint[] }) {
  if (checkpoints.length === 0) {
    return <div className="panel-empty">No checkpoints saved yet.</div>;
  }

  return (
    <div className="layer-stats-container">
      <div className="checkpoints-header">
        <span>Step</span>
        <span>Path</span>
        <span>Best</span>
        <span>Metrics</span>
      </div>
      {checkpoints.map((cp) => (
        <div key={cp.id} className="checkpoint-row">
          <span className="layer-val">{cp.step.toLocaleString()}</span>
          <span className="layer-name">{cp.path}</span>
          <span>
            <span
              className="best-dot"
              style={{
                backgroundColor: cp.is_best ? "#22c55e" : "var(--border)",
              }}
            />
          </span>
          <span className="layer-val" style={{ textAlign: "left" }}>
            {Object.entries(cp.metrics)
              .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`)
              .join(", ") || "\u2014"}
          </span>
        </div>
      ))}
    </div>
  );
}

export interface ModelsPageProps {
  runId: number | null;
  run: RunDetail | null;
  checkpoints: Checkpoint[];
  selectedStep: number | null;
  onSelectStep: (step: number | null) => void;
  weights: WeightLayer[];
  loadingWeights: boolean;
}

export function ModelsPage({
  runId,
  run,
  checkpoints,
  selectedStep,
  onSelectStep,
  weights,
  loadingWeights,
}: ModelsPageProps) {
  if (runId == null) {
    return (
      <main className="models-layout">
        <div className="panel">
          <div className="panel-empty">Select a run to view model details.</div>
        </div>
      </main>
    );
  }

  const modelConfig = run?.config?.model;
  const modelName = modelConfig?.name ?? "unknown";
  const extraArgs = modelConfig?.extra_args ?? {};

  return (
    <main className="models-layout">
      <div className="panel">
        <h3 className="panel-title">Model Architecture</h3>
        {!run ? (
          <div className="panel-empty">Loading...</div>
        ) : (
          <div className="metrics-grid">
            <Card label="Type" value={modelName} />
            {Object.entries(extraArgs).map(([k, v]) => (
              <Card key={k} label={formatKey(k)} value={String(v)} />
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <h3 className="panel-title">Layer Structure</h3>
        <ModelLayers modelName={modelName} extraArgs={extraArgs} />
      </div>

      <div className="panel">
        <h3 className="panel-title">Checkpoints</h3>
        <CheckpointsTable checkpoints={checkpoints} />
      </div>

      <div className="panel">
        <div className="checkpoint-selector-row">
          <h3 className="panel-title" style={{ marginBottom: 0 }}>Weight Heatmaps</h3>
          {checkpoints.length > 0 && (
            <select
              className="run-selector"
              value={selectedStep ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                onSelectStep(val === "" ? null : Number(val));
              }}
            >
              <option value="">Select checkpoint...</option>
              {checkpoints.map((cp) => (
                <option key={cp.step} value={cp.step}>
                  Step {cp.step.toLocaleString()}
                  {cp.is_best ? " (best)" : ""}
                </option>
              ))}
            </select>
          )}
        </div>
        {selectedStep == null ? (
          <div className="panel-empty">
            {checkpoints.length === 0
              ? "No checkpoints available."
              : "Select a checkpoint to view weights."}
          </div>
        ) : loadingWeights ? (
          <div className="panel-empty">Loading weights...</div>
        ) : weights.length === 0 ? (
          <div className="panel-empty">No weight data available.</div>
        ) : (
          <div className="weights-grid">
            {weights.map((w) => (
              <WeightHeatmap
                key={w.name}
                name={w.name}
                shape={w.shape}
                data={w.data}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
