import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { LRChart } from "../components/LRChart";
import { useQuery } from "../db/hooks";
import { getMetricSeries } from "../db/queries";

export function LRChartContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const points = useQuery(useCallback(() => getMetricSeries("lr", runId), [runId]));

  return (
    <LRChart
      x={points.map((p) => p.step)}
      y={points.map((p) => p.value)}
    />
  );
}
