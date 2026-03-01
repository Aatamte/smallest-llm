import { useEffect, useRef } from "react";
import { useStore } from "jotai";
import { stepsAtom } from "../storage";
import { LRChart } from "../components/LRChart";

export function LRChartContainer() {
  const store = useStore();
  const xData = useRef<number[]>([0]);
  const yData = useRef<number[]>([0]);
  const lastLen = useRef(0);

  const onDataRef = useRef<((x: number[], y: number[]) => void) | null>(null);

  useEffect(() => {
    const unsub = store.sub(stepsAtom, () => {
      const steps = store.get(stepsAtom);

      if (steps.length === 0) {
        xData.current = [0];
        yData.current = [0];
        lastLen.current = 0;
        onDataRef.current?.(xData.current.slice(), yData.current.slice());
        return;
      }

      if (steps.length <= lastLen.current) {
        lastLen.current = steps.length;
        return;
      }

      const news = steps.slice(lastLen.current);
      lastLen.current = steps.length;
      for (const m of news) {
        xData.current.push(m.step);
        yData.current.push(m.lr);
      }
      onDataRef.current?.(
        xData.current.slice(),
        yData.current.slice(),
      );
    });
    return unsub;
  }, [store]);

  return (
    <LRChart
      initialX={xData.current}
      initialY={yData.current}
      onDataRef={onDataRef}
    />
  );
}
