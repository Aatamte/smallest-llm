import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { LossChart } from "../components/LossChart";
import { useQuery } from "../db/hooks";
import { getTrainLossSeries, getValLossSeries } from "../db/queries";

export function LossChartContainer() {
  const runId = useAtomValue(activeRunIdAtom);

  const train = useQuery(useCallback(() => getTrainLossSeries(runId), [runId]));
  const val = useQuery(useCallback(() => getValLossSeries(runId), [runId]));

  return (
    <LossChart
      trainX={train.map((p) => p.step)}
      trainY={train.map((p) => p.value)}
      valX={val.map((p) => p.step)}
      valY={val.map((p) => p.value)}
      compareSeries={[]}
    />
  );
}
