import { useEffect, useRef } from "react";
import { useStore, useAtomValue } from "jotai";
import { stepsAtom } from "../storage";
import { MetricChart } from "../components/MetricChart";
import type { StepMetrics } from "../types/metrics";

type MetricKey = keyof StepMetrics;

export interface MetricChartContainerProps {
  metricKey: MetricKey;
  label: string;
  color: string;
  format?: (v: number) => string;
  sub?: string;
}

export function MetricChartContainer({
  metricKey,
  label,
  color,
  format = (v) => v.toFixed(4),
  sub,
}: MetricChartContainerProps) {
  const store = useStore();
  const steps = useAtomValue(stepsAtom);
  const xData = useRef<number[]>([0]);
  const yData = useRef<number[]>([0]);
  const lastLen = useRef(0);
  const onDataRef = useRef<((x: number[], y: number[]) => void) | null>(null);

  useEffect(() => {
    const unsub = store.sub(stepsAtom, () => {
      const allSteps = store.get(stepsAtom);

      if (allSteps.length === 0) {
        xData.current = [0];
        yData.current = [0];
        lastLen.current = 0;
        onDataRef.current?.(xData.current.slice(), yData.current.slice());
        return;
      }

      if (allSteps.length <= lastLen.current) {
        lastLen.current = allSteps.length;
        return;
      }

      const news = allSteps.slice(lastLen.current);
      lastLen.current = allSteps.length;
      for (const m of news) {
        const val = m[metricKey];
        if (val !== undefined && typeof val === "number") {
          xData.current.push(m.step);
          yData.current.push(val);
        }
      }
      onDataRef.current?.(xData.current.slice(), yData.current.slice());
    });
    return unsub;
  }, [store, metricKey]);

  // Current value from latest step
  const last = steps.length > 0 ? steps[steps.length - 1] : null;
  const rawVal = last ? (last[metricKey] as number | undefined) : undefined;
  const currentValue = rawVal !== undefined ? format(rawVal) : "—";

  return (
    <MetricChart
      label={label}
      color={color}
      currentValue={currentValue}
      sub={sub}
      initialX={xData.current}
      initialY={yData.current}
      onDataRef={onDataRef}
    />
  );
}
