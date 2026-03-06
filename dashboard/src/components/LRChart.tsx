import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { CHART_COLORS, basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 400;

export interface LRChartProps {
  x: number[];
  y: number[];
}

export function LRChart({ x, y }: LRChartProps) {
  const traces = useMemo((): Data[] => [
    {
      x,
      y,
      name: "LR",
      type: "scatter",
      mode: "lines",
      line: { color: CHART_COLORS.lr, width: 2 },
    },
  ], [x, y]);

  const layout = useMemo(() => basePlotlyLayout({ height: CHART_HEIGHT }), []);
  const config = useMemo(() => basePlotlyConfig(), []);

  return (
    <div className="panel">
      <h3 className="panel-title">Learning Rate</h3>
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
