import { useMemo } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { LossChart } from "../components/LossChart";
import type { CompareSeries } from "../components/LossChart";
import { useMetricSeries } from "../db/hooks";
import { getMetricSeries } from "../db/queries";
import { COMPARE_COLORS } from "../components/CompareRunsSelector";

export interface LossChartContainerProps {
  compareRunIds?: number[];
  runs?: { id: number; name: string }[];
}

export function LossChartContainer({ compareRunIds = [], runs = [] }: LossChartContainerProps) {
  const runId = useAtomValue(activeRunIdAtom);

  const train = useMetricSeries("trainLoss", runId);
  const val = useMetricSeries("valLoss", runId);

  const compareSeries = useMemo((): CompareSeries[] => {
    return compareRunIds.map((rid, i) => {
      const pts = getMetricSeries("trainLoss", rid);
      const run = runs.find((r) => r.id === rid);
      return {
        name: `#${rid} ${run?.name ?? ""}`.trim(),
        x: pts.map((p) => p.step),
        y: pts.map((p) => p.value),
        color: COMPARE_COLORS[i % COMPARE_COLORS.length],
      };
    });
  }, [compareRunIds, runs]);

  return (
    <LossChart
      trainX={train.map((p) => p.step)}
      trainY={train.map((p) => p.value)}
      valX={val.map((p) => p.step)}
      valY={val.map((p) => p.value)}
      compareSeries={compareSeries}
    />
  );
}
