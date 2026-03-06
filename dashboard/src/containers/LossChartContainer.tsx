import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { LossChart } from "../components/LossChart";
import { useMetricSeries } from "../db/hooks";

export function LossChartContainer() {
  const runId = useAtomValue(activeRunIdAtom);

  const train = useMetricSeries("trainLoss", runId);
  const val = useMetricSeries("valLoss", runId);

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
