import { navigateTo } from "../storage/atoms/uiAtoms";
import type { TrainingStatus } from "../types/metrics";

interface Run {
  id: number;
  name: string;
  status: string;
  created_at: string;
}

const STATUS_COLORS: Record<string, string> = {
  running: "#22c55e",
  completed: "#3b82f6",
  failed: "#ef4444",
};

// ── Run List (default #/runs) ────────────────────────────

function RunList({
  runs,
  onSelectRun,
}: {
  runs: Run[];
  onSelectRun: (id: number) => void;
}) {
  return (
    <main className="runs-layout">
      <div className="runs-header-row">
        <h3 className="panel-title" style={{ marginBottom: 0 }}>Runs</h3>
        <button className="run-new-btn" onClick={() => navigateTo("runs", "new")}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Run
        </button>
      </div>

      <div className="panel" style={{ flex: 1 }}>
        {runs.length === 0 ? (
          <div className="panel-empty">No runs yet. Start one!</div>
        ) : (
          <div className="runs-list">
            {runs
              .slice()
              .reverse()
              .map((run) => (
                <button
                  className="run-item run-item-clickable"
                  key={run.id}
                  onClick={() => onSelectRun(run.id)}
                >
                  <div className="run-item-left">
                    <span
                      className="run-status-dot"
                      style={{ backgroundColor: STATUS_COLORS[run.status] ?? "#6b7280" }}
                    />
                    <span className="run-item-name">{run.name}</span>
                    <span className="run-item-id">#{run.id}</span>
                  </div>
                  <div className="run-item-right">
                    <span className="run-item-status">{run.status}</span>
                    <span className="run-item-date">
                      {new Date(run.created_at).toLocaleString()}
                    </span>
                  </div>
                </button>
              ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ── New Run (#/runs/new) ─────────────────────────────────

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

function NewRunPage({
  status,
  config,
  presets,
  activePreset,
  onPresetChange,
  onConfigChange,
  onTopLevelChange,
  starting,
  error,
  onStart,
}: {
  status: TrainingStatus;
  config: Record<string, unknown> | null;
  presets: { name: string; label: string }[];
  activePreset: string;
  onPresetChange: (name: string) => void;
  onConfigChange: (section: string, key: string, value: unknown) => void;
  onTopLevelChange: (key: string, value: unknown) => void;
  starting: boolean;
  error: string | null;
  onStart: () => void;
}) {
  const isRunning = status === "training";

  const topLevel = config
    ? Object.entries(config).filter(([, v]) => typeof v !== "object" || v === null)
    : [];
  const sections = config
    ? Object.entries(config).filter(
        ([, v]) => typeof v === "object" && v !== null && !Array.isArray(v),
      )
    : [];

  return (
    <main className="runs-layout">
      <div className="runs-header-row">
        <button className="run-back-btn" onClick={() => navigateTo("runs")}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12" /><polyline points="12 19 5 12 12 5" />
          </svg>
          Back
        </button>
        <h3 className="panel-title" style={{ marginBottom: 0 }}>New Run</h3>
      </div>

      <div className="panel">
        <h3 className="panel-title">Start Training</h3>
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

      <div className="panel" style={{ flex: 1 }}>
        <div className="preset-selector-row">
          <h3 className="panel-title" style={{ marginBottom: 0 }}>Configuration</h3>
          {presets.length > 0 && (
            <select
              className="run-selector"
              value={activePreset}
              onChange={(e) => onPresetChange(e.target.value)}
            >
              {presets.map((p) => (
                <option key={p.name} value={p.name}>{p.label}</option>
              ))}
            </select>
          )}
        </div>
        {!config ? (
          <div className="panel-empty">Loading config...</div>
        ) : (
          <div className="config-grid">
            {topLevel.length > 0 && (
              <ConfigSection
                title="General"
                data={Object.fromEntries(topLevel)}
                onChange={(key, value) => onTopLevelChange(key, value)}
              />
            )}
            {sections.map(([sectionKey, value]) => (
              <ConfigSection
                key={sectionKey}
                title={sectionKey}
                data={value as Record<string, unknown>}
                onChange={(key, val) => onConfigChange(sectionKey, key, val)}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ── Router ───────────────────────────────────────────────

export interface RunPageProps {
  sub: string | null;
  runs: Run[];
  onSelectRun: (id: number) => void;
  status: TrainingStatus;
  config: Record<string, unknown> | null;
  presets: { name: string; label: string }[];
  activePreset: string;
  onPresetChange: (name: string) => void;
  onConfigChange: (section: string, key: string, value: unknown) => void;
  onTopLevelChange: (key: string, value: unknown) => void;
  starting: boolean;
  error: string | null;
  onStart: () => void;
}

export function RunPage(props: RunPageProps) {
  return props.sub === "new" ? (
    <NewRunPage
      status={props.status}
      config={props.config}
      presets={props.presets}
      activePreset={props.activePreset}
      onPresetChange={props.onPresetChange}
      onConfigChange={props.onConfigChange}
      onTopLevelChange={props.onTopLevelChange}
      starting={props.starting}
      error={props.error}
      onStart={props.onStart}
    />
  ) : (
    <RunList runs={props.runs} onSelectRun={props.onSelectRun} />
  );
}
