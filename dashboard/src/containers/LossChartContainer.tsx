import { useEffect, useRef } from "react";
import { useStore } from "jotai";
import { stepsAtom } from "../storage";
import { LossChart } from "../components/LossChart";

export function LossChartContainer() {
  const store = useStore();
  const xData = useRef<number[]>([0]);
  const trainData = useRef<number[]>([0]);
  const valData = useRef<number[]>([0]);
  const lastLen = useRef(0);
  const onDataRef = useRef<((x: number[], train: number[], val: number[]) => void) | null>(null);

  useEffect(() => {
    const unsub = store.sub(stepsAtom, () => {
      const steps = store.get(stepsAtom);

      // Reset refs when steps are cleared (e.g. run switch)
      if (steps.length === 0) {
        xData.current = [0];
        trainData.current = [0];
        valData.current = [0];
        lastLen.current = 0;
        onDataRef.current?.(xData.current.slice(), trainData.current.slice(), valData.current.slice());
        return;
      }

      if (steps.length <= lastLen.current) {
        // Array shrank but not to zero (cap trimming) — resync
        lastLen.current = steps.length;
        return;
      }

      const news = steps.slice(lastLen.current);
      lastLen.current = steps.length;
      for (const m of news) {
        xData.current.push(m.step);
        trainData.current.push(m.trainLoss);
        valData.current.push(m.valLoss ?? valData.current[valData.current.length - 1]);
      }
      onDataRef.current?.(
        xData.current.slice(),
        trainData.current.slice(),
        valData.current.slice(),
      );
    });
    return unsub;
  }, [store]);

  return (
    <LossChart
      initialX={xData.current}
      initialTrain={trainData.current}
      initialVal={valData.current}
      onDataRef={onDataRef}
    />
  );
}
