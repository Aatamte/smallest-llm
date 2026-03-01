import { useEffect, useRef, type MutableRefObject } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { baseOpts } from "../types/chart";

export interface MetricChartProps {
  label: string;
  color: string;
  currentValue: string;
  sub?: string;
  initialX: number[];
  initialY: number[];
  onDataRef: MutableRefObject<((x: number[], y: number[]) => void) | null>;
}

const CHART_HEIGHT = 150;

export function MetricChart({
  label,
  color,
  currentValue,
  sub,
  initialX,
  initialY,
  onDataRef,
}: MetricChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const el = divRef.current;
    if (!wrap || !el) return;

    const w = wrap.clientWidth;

    const opts = {
      ...baseOpts(w, CHART_HEIGHT),
      series: [
        {},
        { label, stroke: color, width: 1.5, fill: color + "18" },
      ],
      legend: { show: false },
      cursor: { show: true, points: { show: false } },
    } as uPlot.Options;

    const plot = new uPlot(opts, [initialX, initialY], el);
    plotRef.current = plot;

    onDataRef.current = (x, y) => {
      plotRef.current?.setData([x, y]);
    };

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry && plotRef.current) {
        plotRef.current.setSize({ width: entry.contentRect.width, height: CHART_HEIGHT });
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
    <div className="metric-chart-card" ref={wrapRef}>
      <div className="metric-chart-header">
        <span className="metric-label">{label}</span>
        <span className="metric-chart-value" style={{ color }}>
          {currentValue}
          {sub && <span className="metric-sub"> {sub}</span>}
        </span>
      </div>
      <div ref={divRef} />
    </div>
  );
}
