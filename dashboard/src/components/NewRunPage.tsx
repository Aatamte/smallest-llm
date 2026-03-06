import type { TrainingStatus } from "../types/metrics";

type StageData = Record<string, unknown>;

/** Key fields to highlight at the top of each stage card. */
const STAGE_KEY_FIELDS = ["max_steps", "seq_len", "lr", "dataset_name", "stage_type"];

function ConfigSection({
  title,
  data,
  onChange,
}: {
  title: string;
  data: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  return (
    <div className="config-section">
      <h4 className="config-section-title">{title}</h4>
      <div className="config-fields">
        {Object.entries(data).map(([key, value]) => (
          <div className="config-field" key={key}>
            <span className="config-key">{key}</span>
            <ConfigInput value={value} onChange={(v) => onChange(key, v)} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfigInput({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (typeof value === "boolean") {
    return (
      <input
        className="config-checkbox"
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  }
  if (typeof value === "number") {
    return (
      <input
        className="config-input"
        type="text"
        value={String(value)}
        onChange={(e) => {
          const n = Number(e.target.value);
          onChange(isNaN(n) ? e.target.value : n);
        }}
      />
    );
  }
  return (
    <input
      className="config-input"
      type="text"
      value={String(value ?? "")}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function StagesSection({
  stages,
  onStageChange,
}: {
  stages: StageData[];
  onStageChange: (index: number, key: string, value: unknown) => void;
}) {
  return (
    <div className="stages-section">
      <h4 className="config-section-title">Training Pipeline</h4>
      <div className="stages-pipeline">
        {stages.map((stage, i) => {
          const name = String(stage.name ?? `Stage ${i + 1}`);
          const stageType = String(stage.stage_type ?? "pretrain");
          // Split fields: key fields shown prominently, rest collapsed
          const keyEntries = STAGE_KEY_FIELDS
            .filter((k) => k in stage && k !== "name")
            .map((k) => [k, stage[k]] as const);
          const otherEntries = Object.entries(stage).filter(
            ([k]) => k !== "name" && !STAGE_KEY_FIELDS.includes(k) && stage[k] !== null,
          );

          return (
            <div className="stage-card" key={i}>
              <div className="stage-header">
                <span className="stage-index">{i + 1}</span>
                <span className="stage-name">{name}</span>
                <span className={`stage-type stage-type--${stageType}`}>{stageType}</span>
              </div>
              {i < stages.length - 1 && <div className="stage-arrow" />}
              <div className="stage-key-fields">
                {keyEntries.map(([k, v]) => (
                  <div className="config-field" key={k}>
                    <span className="config-key">{k}</span>
                    <ConfigInput value={v} onChange={(val) => onStageChange(i, k, val)} />
                  </div>
                ))}
              </div>
              {otherEntries.length > 0 && (
                <details className="stage-details">
                  <summary className="stage-details-toggle">more fields</summary>
                  <div className="stage-other-fields">
                    {otherEntries.map(([k, v]) => (
                      <div className="config-field" key={k}>
                        <span className="config-key">{k}</span>
                        <ConfigInput value={v} onChange={(val) => onStageChange(i, k, val)} />
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export interface NewRunPageProps {
  status: TrainingStatus;
  presets: { name: string; label: string; description?: string }[];
  activePreset: string;
  onPresetChange: (name: string) => void;
  evalPresets: { name: string; label: string }[];
  activeEvalPreset: string;
  onEvalPresetChange: (name: string) => void;
  starting: boolean;
  error: string | null;
  onStart: () => void;
}

export function NewRunPage({
  status,
  presets,
  activePreset,
  onPresetChange,
  evalPresets,
  activeEvalPreset,
  onEvalPresetChange,
  starting,
  error,
  onStart,
}: NewRunPageProps) {
  const isRunning = status === "training";

  return (
    <main className="runs-layout">
      <div className="runs-header-row">
        <h3 className="panel-title" style={{ marginBottom: 0 }}>New Run</h3>
      </div>

      <div className="panel">
        <div className="preset-selector-row">
          <h3 className="panel-title" style={{ marginBottom: 0 }}>Training Preset</h3>
          {presets.length > 0 && (
            <>
              <select
                className="run-selector"
                value={activePreset}
                onChange={(e) => onPresetChange(e.target.value)}
              >
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>{p.label}</option>
                ))}
              </select>
              {(() => {
                const desc = presets.find((p) => p.name === activePreset)?.description;
                return desc ? <span className="preset-description">{desc}</span> : null;
              })()}
            </>
          )}
        </div>
        <div className="preset-selector-row" style={{ marginTop: 8 }}>
          <h3 className="panel-title" style={{ marginBottom: 0 }}>Eval Preset</h3>
          {evalPresets.length > 0 && (
            <select
              className="run-selector"
              value={activeEvalPreset}
              onChange={(e) => onEvalPresetChange(e.target.value)}
            >
              {evalPresets.map((p) => (
                <option key={p.name} value={p.name}>{p.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="run-start-row">
          <button
            className="run-start-btn"
            onClick={onStart}
            disabled={isRunning || starting}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            {starting ? "Starting..." : isRunning ? "Running..." : "Start Training"}
          </button>
          {error && <span className="run-error">{error}</span>}
        </div>
      </div>
    </main>
  );
}
