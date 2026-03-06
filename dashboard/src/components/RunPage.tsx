import { useState } from "react";
import { navigateTo } from "../storage/atoms/uiAtoms";
import type { TrainingStatus } from "../types/metrics";
import type { RunInfo } from "../storage";

const STATUS_COLORS: Record<string, string> = {
  running: "#22c55e",
  completed: "#3b82f6",
  failed: "#ef4444",
};

// ── Run List (default #/runs) ────────────────────────────

function RunList({
  runs,
  onSelectRun,
  onDeleteRun,
  onBulkDelete,
}: {
  runs: RunInfo[];
  onSelectRun: (id: number) => void;
  onDeleteRun: (id: number) => void;
  onBulkDelete: (ids: number[]) => void;
}) {
  const [selected, setSelected] = useState<Set<number>>(new Set());

  function toggleOne(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll() {
    if (selected.size === runs.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(runs.map((r) => r.id)));
    }
  }

  function handleBulkDelete() {
    const ids = [...selected];
    if (ids.length === 0) return;
    if (!window.confirm(`Delete ${ids.length} run(s)? This cannot be undone.`)) return;
    onBulkDelete(ids);
    setSelected(new Set());
  }

  return (
    <main className="runs-layout">
      <div className="runs-header-row">
        <h3 className="panel-title" style={{ marginBottom: 0 }}>Runs</h3>
        {selected.size > 0 && (
          <button className="run-bulk-delete-btn" onClick={handleBulkDelete}>
            Delete {selected.size} selected
          </button>
        )}
      </div>

      <div className="panel">
        {runs.length === 0 ? (
          <div className="panel-empty">No runs yet. Start one!</div>
        ) : (
          <>
            {runs.length > 1 && (
              <div className="runs-select-all">
                <label className="run-checkbox-label">
                  <input
                    type="checkbox"
                    checked={selected.size === runs.length}
                    onChange={toggleAll}
                  />
                  Select all
                </label>
              </div>
            )}
            <div className="runs-list">
              {runs
                .slice()
                .reverse()
                .map((run) => (
                  <div className="run-item-row" key={run.id}>
                    <input
                      type="checkbox"
                      className="run-checkbox"
                      checked={selected.has(run.id)}
                      onChange={(e) => {
                        e.stopPropagation();
                        toggleOne(run.id);
                      }}
                    />
                    <div
                      className="run-item run-item-clickable"
                      role="button"
                      tabIndex={0}
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
                        <button
                          className="run-delete-btn"
                          title="Delete run"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (window.confirm(`Delete run #${run.id} "${run.name}"? This cannot be undone.`)) {
                              onDeleteRun(run.id);
                            }
                          }}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </>
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

      <div className="panel">
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
  runs: RunInfo[];
  onSelectRun: (id: number) => void;
  onDeleteRun: (id: number) => void;
  onBulkDelete: (ids: number[]) => void;
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
    <RunList runs={props.runs} onSelectRun={props.onSelectRun} onDeleteRun={props.onDeleteRun} onBulkDelete={props.onBulkDelete} />
  );
}
