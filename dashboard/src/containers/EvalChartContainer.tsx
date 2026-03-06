import { useCallback, useMemo } from "react";
import { useAtomValue } from "jotai";
import { compareRunsDataAtom } from "../storage";
import { MetricChart } from "../components/MetricChart";
import { COMPARE_COLORS } from "../types/chart";
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
  const compareData = useAtomValue(compareRunsDataAtom);
  const compareEntries = Object.entries(compareData);

  const traces = useMemo((): Data[] => {
    return [
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
      ...compareEntries.map(([, d], i) => {
        const x: number[] = [];
        const y: number[] = [];
        for (const e of d.evals) {
          if (e.task === task && e.metrics[metric] !== undefined) {
            x.push(e.step);
            y.push(e.metrics[metric]);
          }
        }
        return {
          x,
          y,
          name: d.name,
          type: "scatter" as const,
          mode: "lines+markers" as const,
          line: { color: COMPARE_COLORS[i % COMPARE_COLORS.length], width: 1.5, dash: "dash" as const },
          marker: { size: 4, color: COMPARE_COLORS[i % COMPARE_COLORS.length] },
        };
      }),
    ];
  }, [points, compareData, task, metric, label, color]);

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
