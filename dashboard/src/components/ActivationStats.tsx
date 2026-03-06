import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getActivationStats } from "../db/queries";

function deadColor(pct: number): string {
  if (pct < 5) return "#22c55e";
  if (pct < 20) return "#eab308";
  return "#ef4444";
}

function stdColor(std: number): string {
  if (std < 0.01) return "#eab308";
  if (std > 10) return "#ef4444";
  return "var(--text)";
}

export function ActivationStats() {
  const runId = useAtomValue(activeRunIdAtom);
  const stats = useQuery(useCallback(() => getActivationStats(runId), [runId]));

  if (stats.length === 0) {
    return (
      <div className="panel">
        <h3 className="panel-title">Activation Stats</h3>
        <div className="panel-empty">Waiting for data...</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Activation Stats</h3>
      <div className="layer-stats-container">
        <div className="activation-header">
          <span>Layer</span>
          <span>Mean</span>
          <span>Std</span>
          <span>Max</span>
          <span>% Dead</span>
        </div>
        {stats.map((s) => (
          <div key={s.name} className="activation-row">
            <span className="layer-name">{s.name}</span>
            <span className="layer-val">{s.mean.toFixed(4)}</span>
            <span className="layer-val" style={{ color: stdColor(s.std) }}>
              {s.std.toFixed(4)}
            </span>
            <span className="layer-val">{s.max.toFixed(3)}</span>
            <span className="layer-val" style={{ color: deadColor(s.pctZero) }}>
              {s.pctZero.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
