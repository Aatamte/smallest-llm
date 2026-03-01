import { useEffect, useRef, type MutableRefObject } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { CHART_COLORS, baseOpts } from "../types/chart";

export interface LRChartProps {
  initialX: number[];
  initialY: number[];
  onDataRef: MutableRefObject<((x: number[], y: number[]) => void) | null>;
}

export function LRChart({ initialX, initialY, onDataRef }: LRChartProps) {
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const el = divRef.current;
    if (!el) return;

    const plot = new uPlot({
      ...baseOpts(800, 400),
      series: [
        {},
        { label: "LR", stroke: CHART_COLORS.lr, width: 2 },
      ],
    } as uPlot.Options, [initialX, initialY], el);

    plotRef.current = plot;

    onDataRef.current = (x, y) => {
      plotRef.current?.setData([x, y]);
    };

    return () => {
      plot.destroy();
      plotRef.current = null;
      onDataRef.current = null;
    };
  }, []);

  return (
    <div className="panel">
      <h3 className="panel-title">Learning Rate</h3>
      <div ref={divRef} />
    </div>
  );
}
