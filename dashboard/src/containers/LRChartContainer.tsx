import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { LRChart } from "../components/LRChart";
import { useMetricSeries } from "../db/hooks";

export function LRChartContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const points = useMetricSeries("lr", runId);

  return (
    <LRChart
      x={points.map((p) => p.step)}
      y={points.map((p) => p.value)}
    />
  );
}
