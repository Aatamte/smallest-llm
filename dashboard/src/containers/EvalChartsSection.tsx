import { EvalChartContainer } from "./EvalChartContainer";
import { useEvalTaskMetrics } from "../db/hooks";

const EVAL_COLORS = ["#f59e0b", "#10b981", "#ec4899", "#06b6d4", "#8b5cf6"];

/**
 * Renders eval charts only for tasks that have data.
 * Hidden entirely when no eval results exist yet.
 * Tasks and metrics are discovered dynamically from the data.
 */
export function EvalChartsSection() {
  const taskMetric = useEvalTaskMetrics();

  const allTasks = [...taskMetric.keys()].sort();
  if (allTasks.length === 0) return null;

  return (
    <>
      <h3 className="eval-charts-heading">Eval Metrics</h3>
      <div className="metric-charts-grid">
        {allTasks.map((task, i) => {
          const metric = taskMetric.get(task)!;
          return (
            <EvalChartContainer
              key={task}
              task={task}
              metric={metric}
              label={`${task} / ${metric}`}
              color={EVAL_COLORS[i % EVAL_COLORS.length]}
            />
          );
        })}
      </div>
    </>
  );
}
