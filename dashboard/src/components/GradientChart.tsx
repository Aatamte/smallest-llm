import { useEffect, useRef, type MutableRefObject } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { CHART_COLORS, baseOpts } from "../types/chart";

export interface GradientChartProps {
  initialX: number[];
  initialGrad: number[];
  initialRatio: number[];
  onDataRef: MutableRefObject<((x: number[], grad: number[], ratio: number[]) => void) | null>;
}

export function GradientChart({ initialX, initialGrad, initialRatio, onDataRef }: GradientChartProps) {
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const el = divRef.current;
    if (!el) return;

    const plot = new uPlot({
      ...baseOpts(800, 400),
      series: [
        {},
        { label: "Grad Norm", stroke: CHART_COLORS.gradNorm, width: 2 },
        { label: "Update/Param", stroke: CHART_COLORS.updateRatio, width: 2 },
      ],
    } as uPlot.Options, [initialX, initialGrad, initialRatio], el);

    plotRef.current = plot;

    onDataRef.current = (x, grad, ratio) => {
      plotRef.current?.setData([x, grad, ratio]);
    };

    return () => {
      plot.destroy();
      plotRef.current = null;
      onDataRef.current = null;
    };
  }, []);

  return (
    <div className="panel">
      <h3 className="panel-title">Gradient Health</h3>
      <div ref={divRef} />
    </div>
  );
}
