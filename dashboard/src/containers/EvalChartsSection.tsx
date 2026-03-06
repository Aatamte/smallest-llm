import { useMemo } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { EvalChartContainer } from "./EvalChartContainer";
import { useEvalTaskMetrics, useCompositeSeries } from "../db/hooks";
import { getCompositeSeries } from "../db/queries";
import { MetricChart } from "../components/MetricChart";
import { COMPARE_COLORS } from "../components/CompareRunsSelector";
import type { Data } from "plotly.js-dist-min";

const EVAL_COLORS = ["#f59e0b", "#10b981", "#ec4899", "#06b6d4", "#8b5cf6"];
const COMPOSITE_COLOR = "#3b82f6";

export interface EvalChartsSectionProps {
  compareRunIds?: number[];
  runs?: { id: number; name: string }[];
}

export function EvalChartsSection({ compareRunIds = [], runs = [] }: EvalChartsSectionProps) {
  const activeRunId = useAtomValue(activeRunIdAtom);
  const taskMetric = useEvalTaskMetrics(activeRunId);
  const compositeSeries = useCompositeSeries(activeRunId);

  const allTasks = [...taskMetric.keys()].sort();
  const hasData = allTasks.length > 0 || compositeSeries.length > 0;

  const compositeTraces = useMemo((): Data[] => {
    const traces: Data[] = [
      {
        x: compositeSeries.map((p) => p.step),
        y: compositeSeries.map((p) => p.value),
        name: `#${activeRunId ?? "?"} (active)`,
        type: "scatter",
        mode: "lines+markers",
        line: { color: COMPOSITE_COLOR, width: 2 },
        marker: { size: 6, color: COMPOSITE_COLOR },
        fill: "tozeroy",
        fillcolor: COMPOSITE_COLOR + "18",
      },
    ];
    for (let i = 0; i < compareRunIds.length; i++) {
      const rid = compareRunIds[i];
      const pts = getCompositeSeries(rid);
      const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
      const run = runs.find((r) => r.id === rid);
      traces.push({
        x: pts.map((p) => p.step),
        y: pts.map((p) => p.value),
        name: `#${rid} ${run?.name ?? ""}`.trim(),
        type: "scatter",
        mode: "lines+markers",
        line: { color, width: 1.5, dash: "dash" },
        marker: { size: 4, color },
      });
    }
    return traces;
  }, [compositeSeries, compareRunIds, activeRunId, runs]);

  const lastComposite = compositeSeries.length > 0
    ? compositeSeries[compositeSeries.length - 1].value.toFixed(4)
    : "—";

  return (
    <>
      <h3 className="eval-charts-heading">Eval Metrics</h3>
      {!hasData ? (
        <div className="panel-empty">No eval results yet.</div>
      ) : (
        <>
          {compositeSeries.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <MetricChart
                label="Composite Score"
                currentValue={lastComposite}
                valueColor={COMPOSITE_COLOR}
                sub="avg accuracy"
                traces={compositeTraces}
              />
            </div>
          )}
          <div className="metric-charts-grid">
            {allTasks.map((task, i) => {
              const metric = taskMetric.get(task)!;
              return (
                <EvalChartContainer
                  key={task}
                  task={task}
                  metric={metric}
                  runId={activeRunId}
                  label={`${task} / ${metric}`}
                  color={EVAL_COLORS[i % EVAL_COLORS.length]}
                  compareRunIds={compareRunIds}
                  runs={runs}
                />
              );
            })}
          </div>
        </>
      )}
    </>
  );
}
