import { useEffect, useRef } from "react";
import { useStore } from "jotai";
import { stepsAtom } from "../storage";
import { GradientChart } from "../components/GradientChart";

export function GradientChartContainer() {
  const store = useStore();
  const xData = useRef<number[]>([0]);
  const gradData = useRef<number[]>([0]);
  const ratioData = useRef<number[]>([0]);
  const lastLen = useRef(0);

  const onDataRef = useRef<((x: number[], grad: number[], ratio: number[]) => void) | null>(null);

  useEffect(() => {
    const unsub = store.sub(stepsAtom, () => {
      const steps = store.get(stepsAtom);

      if (steps.length === 0) {
        xData.current = [0];
        gradData.current = [0];
        ratioData.current = [0];
        lastLen.current = 0;
        onDataRef.current?.(xData.current.slice(), gradData.current.slice(), ratioData.current.slice());
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
        gradData.current.push(m.gradNorm);
        ratioData.current.push(m.updateParamRatio ?? 0);
      }
      onDataRef.current?.(
        xData.current.slice(),
        gradData.current.slice(),
        ratioData.current.slice(),
      );
    });
    return unsub;
  }, [store]);

  return (
    <GradientChart
      initialX={xData.current}
      initialGrad={gradData.current}
      initialRatio={ratioData.current}
      onDataRef={onDataRef}
    />
  );
}
