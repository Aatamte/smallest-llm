import { useMemo, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { CHART_COLORS, basePlotlyLayout, basePlotlyConfig } from "../types/chart";
import type { Data } from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);

const CHART_HEIGHT = 450;

export interface CompareSeries {
  name: string;
  x: number[];
  y: number[];
  color: string;
}

export interface LossChartProps {
  trainX: number[];
  trainY: number[];
  valX: number[];
  valY: number[];
  compareSeries: CompareSeries[];
}

export function LossChart({ trainX, trainY, valX, valY, compareSeries }: LossChartProps) {
  const [isLog, setIsLog] = useState(false);

  const traces = useMemo((): Data[] => {
    const t: Data[] = [
      {
        x: trainX,
        y: trainY,
        name: "Train",
        type: "scatter",
        mode: "lines",
        line: { color: CHART_COLORS.trainLoss, width: 2 },
      },
    ];
    if (valX.length > 0) {
      t.push({
        x: valX,
        y: valY,
        name: "Val",
        type: "scatter",
        mode: "lines",
        line: { color: CHART_COLORS.valLoss, width: 2, dash: "dash" },
      });
    }
    for (const s of compareSeries) {
      t.push({
        x: s.x,
        y: s.y,
        name: s.name,
        type: "scatter",
        mode: "lines",
        line: { color: s.color, width: 1.5 },
      });
    }
    return t;
  }, [trainX, trainY, valX, valY, compareSeries]);

  const layout = useMemo(() => {
    const base = basePlotlyLayout({ height: CHART_HEIGHT });
    return {
      ...base,
      yaxis: {
        ...(base.yaxis as object),
        ...(isLog ? { type: "log" } : {}),
      },
      showlegend: true,
      legend: { font: { color: CHART_COLORS.text, size: 10 }, bgcolor: "transparent" },
    };
  }, [isLog]);
  const config = useMemo(() => basePlotlyConfig(), []);

  return (
    <div className="panel">
      <div className="metric-chart-header">
        <h3 className="panel-title" style={{ marginBottom: 0 }}>Loss</h3>
        <button
          className={`metric-log-toggle ${isLog ? "active" : ""}`}
          onClick={() => setIsLog((v) => !v)}
        >
          LOG
        </button>
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
