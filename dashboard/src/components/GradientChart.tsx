import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { CHART_COLORS, basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 400;

export interface GradientChartProps {
  x: number[];
  grad: number[];
  ratio: number[];
}

export function GradientChart({ x, grad, ratio }: GradientChartProps) {
  const traces = useMemo((): Data[] => [
    {
      x,
      y: grad,
      name: "Grad Norm",
      type: "scatter",
      mode: "lines",
      line: { color: CHART_COLORS.gradNorm, width: 2 },
    },
    {
      x,
      y: ratio,
      name: "Update/Param",
      type: "scatter",
      mode: "lines",
      line: { color: CHART_COLORS.updateRatio, width: 2 },
    },
  ], [x, grad, ratio]);

  const layout = useMemo(() => ({
    ...basePlotlyLayout({ height: CHART_HEIGHT }),
    showlegend: true,
    legend: { font: { color: CHART_COLORS.text, size: 10 }, bgcolor: "transparent" },
  }), []);
  const config = useMemo(() => basePlotlyConfig(), []);

  return (
    <div className="panel">
      <h3 className="panel-title">Gradient Health</h3>
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
