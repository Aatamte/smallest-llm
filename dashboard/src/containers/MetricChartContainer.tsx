import { useMemo } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { MetricChart } from "../components/MetricChart";
import { useMetricSeries } from "../db/hooks";
import type { Data } from "plotly.js-dist-min";

export interface MetricChartContainerProps {
  metricKey: string;
  label: string;
  color: string;
  format?: (v: number) => string;
  sub?: string;
}

export function MetricChartContainer({
  metricKey,
  label,
  color,
  format = (v) => v.toFixed(4),
  sub,
}: MetricChartContainerProps) {
  const runId = useAtomValue(activeRunIdAtom);

  const points = useMetricSeries(metricKey, runId);

  const traces = useMemo((): Data[] => [
    {
      x: points.map((p) => p.step),
      y: points.map((p) => p.value),
      name: label,
      type: "scatter",
      mode: "lines",
      line: { color, width: 1.5 },
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
