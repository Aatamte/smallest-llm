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

export function basePlotlyLayout(opts: { height?: number } = {}): Record<string, unknown> {
  return {
    height: opts.height ?? 300,
    paper_bgcolor: CHART_COLORS.bg,
    plot_bgcolor: CHART_COLORS.bg,
    font: { color: CHART_COLORS.text, family: "Inter, sans-serif", size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: CHART_COLORS.grid, zerolinecolor: CHART_COLORS.grid },
    yaxis: { gridcolor: CHART_COLORS.grid, zerolinecolor: CHART_COLORS.grid },
    showlegend: false,
  };
}

export function basePlotlyConfig(): Record<string, unknown> {
  return {
    displayModeBar: false,
    responsive: true,
  };
}
