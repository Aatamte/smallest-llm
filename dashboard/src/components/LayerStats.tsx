import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getLayerStats } from "../db/queries";

function normToColor(value: number, max: number): string {
  const ratio = Math.min(value / max, 1);
  if (ratio < 0.3) return "#22c55e";
  if (ratio < 0.6) return "#eab308";
  if (ratio < 0.8) return "#f97316";
  return "#ef4444";
}

export function LayerStats() {
  const runId = useAtomValue(activeRunIdAtom);
  const layerStats = useQuery(useCallback(() => getLayerStats(runId), [runId]));

  if (layerStats.length === 0) {
    return (
      <div className="panel">
        <h3 className="panel-title">Layer Health</h3>
        <div className="panel-empty">Waiting for data...</div>
      </div>
    );
  }

  const maxGrad = Math.max(...layerStats.map((l) => l.gradNorm), 0.01);

  return (
    <div className="panel">
      <h3 className="panel-title">Layer Health</h3>
      <div className="layer-stats-container">
        <div className="layer-header">
          <span>Layer</span>
          <span>Grad Norm</span>
          <span>Weight Norm</span>
          <span>Update Ratio</span>
        </div>
        {layerStats.map((layer) => (
          <div key={layer.name} className="layer-row">
            <span className="layer-name">{layer.name}</span>
            <span className="layer-bar-cell">
              <div
                className="layer-bar"
                style={{
                  width: `${(layer.gradNorm / maxGrad) * 100}%`,
                  backgroundColor: normToColor(layer.gradNorm, maxGrad),
                }}
              />
              <span className="layer-val">{layer.gradNorm.toFixed(3)}</span>
            </span>
            <span className="layer-val">{layer.weightNorm.toFixed(2)}</span>
            <span
              className="layer-val"
              style={{
                color:
                  layer.updateRatio > 0.002
                    ? "#ef4444"
                    : layer.updateRatio < 0.0003
                      ? "#eab308"
                      : "#22c55e",
              }}
            >
              {layer.updateRatio.toExponential(1)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
