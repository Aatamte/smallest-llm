import type uPlot from "uplot";

export const CHART_COLORS = {
  trainLoss: "#3b82f6",
  valLoss: "#f97316",
  lr: "#a855f7",
  gradNorm: "#ef4444",
  updateRatio: "#14b8a6",
  grid: "#1f2937",
  text: "#9ca3af",
  bg: "#0a0a0f",
} as const;

/** Base options shared by all charts (dark theme). */
export function baseOpts(width: number, height: number): Partial<uPlot.Options> {
  return {
    width,
    height,
    cursor: { show: true },
    axes: [
      {
        stroke: CHART_COLORS.text,
        grid: { stroke: CHART_COLORS.grid, width: 1 },
        ticks: { stroke: CHART_COLORS.grid, width: 1 },
        font: "11px Inter, sans-serif",
      },
      {
        stroke: CHART_COLORS.text,
        grid: { stroke: CHART_COLORS.grid, width: 1 },
        ticks: { stroke: CHART_COLORS.grid, width: 1 },
        font: "11px Inter, sans-serif",
      },
    ],
  };
}
