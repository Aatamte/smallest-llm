import { useMemo } from "react";
import { MetricChart } from "../components/MetricChart";
import type { Data } from "plotly.js-dist-min";
import { useEvalSeries } from "../db/hooks";
import { getEvalSeries } from "../db/queries";

const COMPARE_COLORS = ["#f97316", "#a855f7", "#14b8a6", "#e11d48", "#84cc16", "#f59e0b"];

export interface EvalChartContainerProps {
  task: string;
  metric: string;
  runId?: number | null;
  label: string;
  color: string;
  format?: (v: number) => string;
  sub?: string;
  compareRunIds?: number[];
  runs?: { id: number; name: string }[];
}

export function EvalChartContainer({
  task,
  metric,
  runId,
  label,
  color,
  format = (v) => v.toFixed(4),
  sub,
  compareRunIds = [],
  runs = [],
}: EvalChartContainerProps) {
  const points = useEvalSeries(task, metric, runId);

  const traces = useMemo((): Data[] => {
    const result: Data[] = [
      {
        x: points.map((p) => p.step),
        y: points.map((p) => p.value),
        name: label,
        type: "scatter",
        mode: "lines+markers",
        line: { color, width: 1.5 },
        marker: { size: 5, color },
        fill: "tozeroy",
        fillcolor: color + "18",
      },
    ];
    for (let i = 0; i < compareRunIds.length; i++) {
      const rid = compareRunIds[i];
      const pts = getEvalSeries(task, metric, rid);
      const c = COMPARE_COLORS[i % COMPARE_COLORS.length];
      const run = runs.find((r) => r.id === rid);
      result.push({
        x: pts.map((p) => p.step),
        y: pts.map((p) => p.value),
        name: `#${rid} ${run?.name ?? ""}`.trim(),
        type: "scatter",
        mode: "lines+markers",
        line: { color: c, width: 1.5, dash: "dash" },
        marker: { size: 4, color: c },
      });
    }
    return result;
  }, [points, label, color, compareRunIds, runs, task, metric]);

  const last = points.length > 0 ? points[points.length - 1] : null;
  const currentValue = last ? format(last.value) : "—";

  return (
    <MetricChart
      label={label}
      currentValue={currentValue}
      valueColor={color}
      sub={sub}
      traces={traces}
      allowLog
    />
  );
}
