import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { GradientChart } from "../components/GradientChart";
import { useMetricSeries } from "../db/hooks";

export function GradientChartContainer() {
  const runId = useAtomValue(activeRunIdAtom);

  const gradPoints = useMetricSeries("gradNorm", runId);
  // update_ratio may not exist for all runs; fallback to zeros
  const ratioPoints = useMetricSeries("train/update_ratio", runId);

  // Align on grad steps (ratio may be sparse or empty)
  const x = gradPoints.map((p) => p.step);
  const grad = gradPoints.map((p) => p.value);

  const ratioMap = new Map(ratioPoints.map((p) => [p.step, p.value]));
  const ratio = x.map((s) => ratioMap.get(s) ?? 0);

  return <GradientChart x={x} grad={grad} ratio={ratio} />;
}
