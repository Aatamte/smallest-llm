import { useEffect, useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";
import { useEvals, useEvalSeries } from "../db/hooks";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 320;
const CHART_COLOR = "#3b82f6";

export function EvalExplorerChart() {
  const allEvals = useEvals();

  const options = useMemo(() => {
    const seen = new Set<string>();
    for (const e of allEvals) {
      seen.add(`${e.task}/${e.metric_name}`);
    }
    return [...seen].sort();
  }, [allEvals]);

  const [selected, setSelected] = useState<string>("");

  useEffect(() => {
    if (!selected && options.length > 0) {
      setSelected(options[0]);
    }
  }, [options, selected]);

  const [task, metric] = selected.split("/", 2);
  const rawSeries = useEvalSeries(task ?? "", metric ?? "");
  const series = task && metric ? rawSeries : [];

  const traces = useMemo((): Data[] => [
    {
      x: series.map((p) => p.step),
      y: series.map((p) => p.value),
      name: selected || "value",
      type: "scatter",
      mode: "lines+markers",
      line: { color: CHART_COLOR, width: 2 },
      marker: { size: 6, color: CHART_COLOR },
      fill: "tozeroy",
      fillcolor: CHART_COLOR + "18",
    },
  ], [series, selected]);

  const layout = useMemo(() => basePlotlyLayout({ height: CHART_HEIGHT }), []);
  const config = useMemo(() => basePlotlyConfig(), []);

  if (options.length === 0) return null;

  const lastY = series.length > 0 ? series[series.length - 1].value : null;
  const currentValue = lastY !== null ? lastY.toFixed(4) : "—";

  return (
    <div className="eval-explorer-card">
      <div className="eval-explorer-header">
        <select
          className="eval-explorer-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <span className="metric-chart-value" style={{ color: CHART_COLOR }}>
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
