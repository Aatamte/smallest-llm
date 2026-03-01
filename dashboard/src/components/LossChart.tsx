import { useEffect, useRef, type MutableRefObject } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { CHART_COLORS, baseOpts } from "../types/chart";

export interface LossChartProps {
  initialX: number[];
  initialTrain: number[];
  initialVal: number[];
  onDataRef: MutableRefObject<((x: number[], train: number[], val: number[]) => void) | null>;
}

export function LossChart({ initialX, initialTrain, initialVal, onDataRef }: LossChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const el = divRef.current;
    if (!wrap || !el) return;

    const w = wrap.clientWidth;

    const plot = new uPlot({
      ...baseOpts(w, 350),
      series: [
        {},
        { label: "Train", stroke: CHART_COLORS.trainLoss, width: 2 },
        { label: "Val", stroke: CHART_COLORS.valLoss, width: 2, dash: [4, 4] },
      ],
    } as uPlot.Options, [initialX, initialTrain, initialVal], el);

    plotRef.current = plot;

    onDataRef.current = (x, train, val) => {
      plotRef.current?.setData([x, train, val]);
    };

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry && plotRef.current) {
        plotRef.current.setSize({ width: entry.contentRect.width, height: 350 });
      }
    });
    ro.observe(wrap);

    return () => {
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
      onDataRef.current = null;
    };
  }, []);

  return (
    <div className="panel" ref={wrapRef}>
      <h3 className="panel-title">Loss</h3>
      <div ref={divRef} />
    </div>
  );
}
