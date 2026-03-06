import { useCallback, useMemo } from "react";
import { MetricChart } from "../components/MetricChart";
import type { Data } from "plotly.js-dist-min";
import { useQuery } from "../db/hooks";
import { getEvalSeries } from "../db/queries";

export interface EvalChartContainerProps {
  task: string;
  metric: string;
  label: string;
  color: string;
  format?: (v: number) => string;
  sub?: string;
}

export function EvalChartContainer({
  task,
  metric,
  label,
  color,
  format = (v) => v.toFixed(4),
  sub,
}: EvalChartContainerProps) {
  const points = useQuery(useCallback(() => getEvalSeries(task, metric), [task, metric]));

  const traces = useMemo((): Data[] => [
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
  ], [points, label, color]);

  const last = points.length > 0 ? points[points.length - 1] : null;
  const currentValue = last ? format(last.value) : "—";

  return (
    <MetricChart
      label={label}
      currentValue={currentValue}
      valueColor={color}
      sub={sub}
      traces={traces}
    />
  );
}
