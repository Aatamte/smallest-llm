import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 150;

export interface MetricChartProps {
  label: string;
  currentValue: string;
  valueColor: string;
  sub?: string;
  traces: Data[];
}

export function MetricChart({ label, currentValue, valueColor, sub, traces }: MetricChartProps) {
  const layout = useMemo(() => basePlotlyLayout({ height: CHART_HEIGHT }), []);
  const config = useMemo(() => basePlotlyConfig(), []);

  return (
    <div className="metric-chart-card">
      <div className="metric-chart-header">
        <span className="metric-label">{label}</span>
        <span className="metric-chart-value" style={{ color: valueColor }}>
          {currentValue}
          {sub && <span className="metric-sub"> {sub}</span>}
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
