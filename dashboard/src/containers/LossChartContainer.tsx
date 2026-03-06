import { useAtomValue } from "jotai";
import { stepsAtom } from "../storage";
import { LossChart } from "../components/LossChart";

export function LossChartContainer() {
  const steps = useAtomValue(stepsAtom);

  const trainX = steps.map((s) => s.step);
  const trainY = steps.map((s) => s.trainLoss);

  const valX: number[] = [];
  const valY: number[] = [];
  for (const s of steps) {
    if (s.valLoss !== undefined) {
      valX.push(s.step);
      valY.push(s.valLoss);
    }
  }

  return (
    <LossChart
      trainX={trainX}
      trainY={trainY}
      valX={valX}
      valY={valY}
      compareSeries={[]}
    />
  );
}
