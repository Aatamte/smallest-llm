import { useEffect, useMemo, useState } from "react";
import { useAtomValue } from "jotai";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";
import { activeRunIdAtom } from "../storage";
import { useMetricSeries, useMetricKeys } from "../db/hooks";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 400;
const ACTIVE_COLOR = "#3b82f6";

export function SelectableMetricChart() {
  const runId = useAtomValue(activeRunIdAtom);
  const keys = useMetricKeys(runId);
  const [selected, setSelected] = useState("");

  // Auto-select first key when keys change and current selection is invalid
  useEffect(() => {
    if (keys.length > 0 && (!selected || !keys.includes(selected))) {
      setSelected(keys[0]);
    }
  }, [keys, selected]);

  const points = useMetricSeries(selected, runId);

  const traces = useMemo((): Data[] => {
    if (!selected) return [];
    return [
      {
        x: points.map((p) => p.step),
        y: points.map((p) => p.value),
        name: selected,
        type: "scatter",
        mode: "lines",
        line: { color: ACTIVE_COLOR, width: 1.5 },
        fill: "tozeroy",
        fillcolor: ACTIVE_COLOR + "18",
      },
    ];
  }, [points, selected]);

  const layout = useMemo(() => basePlotlyLayout({ height: CHART_HEIGHT }), []);
  const config = useMemo(() => basePlotlyConfig(), []);

  const last = points.length > 0 ? points[points.length - 1] : null;
  const currentValue = last ? last.value.toFixed(4) : "—";

  if (keys.length === 0) return null;

  return (
    <div className="eval-explorer-card">
      <div className="eval-explorer-header">
        <select
          className="eval-explorer-select"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {keys.map((key) => (
            <option key={key} value={key}>{key}</option>
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
