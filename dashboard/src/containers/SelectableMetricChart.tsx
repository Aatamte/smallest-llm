import { useCallback, useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";
import { useQuery } from "../db/hooks";
import { getMetricSeries } from "../db/queries";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 320;
const ACTIVE_COLOR = "#3b82f6";

const METRIC_OPTIONS: { key: string; label: string }[] = [
  { key: "trainLoss", label: "Train Loss" },
  { key: "valLoss", label: "Val Loss" },
  { key: "lr", label: "Learning Rate" },
  { key: "gradNorm", label: "Grad Norm" },
  { key: "tokensPerSec", label: "Tokens/s" },
  { key: "tokensSeen", label: "Tokens Seen" },
  { key: "bpc", label: "BPC" },
  { key: "stepTime", label: "Step Time" },
];

export function SelectableMetricChart() {
  const [selected, setSelected] = useState("trainLoss");

  const points = useQuery(useCallback(() => getMetricSeries(selected), [selected]));

  const traces = useMemo((): Data[] => {
    const label = METRIC_OPTIONS.find((o) => o.key === selected)?.label ?? selected;
    return [
      {
        x: points.map((p) => p.step),
        y: points.map((p) => p.value),
        name: label,
        type: "scatter",
        mode: "lines",
        line: { color: ACTIVE_COLOR, width: 1.5 },
        fill: "tozeroy",
        fillcolor: ACTIVE_COLOR + "18",
      },
    ];
  }, [points, selected]);

  const layout = useMemo(
    () => basePlotlyLayout({ height: CHART_HEIGHT }),
    [],
  );
  const config = useMemo(() => basePlotlyConfig(), []);

  const last = points.length > 0 ? points[points.length - 1] : null;
  const currentValue = last ? last.value.toFixed(4) : "—";

  return (
    <div className="eval-explorer-card">
      <div className="eval-explorer-header">
        <select
          className="eval-explorer-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {METRIC_OPTIONS.map((opt) => (
            <option key={opt.key} value={opt.key}>{opt.label}</option>
          ))}
        </select>
        <span className="metric-chart-value" style={{ color: ACTIVE_COLOR }}>
          {currentValue}
        </span>
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={config}
        useResizeHandler
        style={{ width: "100%", height: CHART_HEIGHT }}
      />
    </div>
  );
}
