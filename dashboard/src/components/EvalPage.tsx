import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { EvalResult } from "../api/client";
import { CHART_COLORS, baseOpts } from "../types/chart";

// ── Hardcoded baseline summary metrics ──────────────────

const BASELINES: Record<string, Record<string, Record<string, number>>> = {
  "smollm-135m": {
    perplexity: { perplexity: 1119.08, bpc: 3.6172, cross_entropy_nats: 7.0203 },
    blimp: {
      accuracy: 0.8149,
      accuracy_morphology: 0.9333,
      accuracy_semantics: 0.8667,
      accuracy_syntax: 0.7154,
      accuracy_syntax_semantics: 0.8143,
    },
  },
  "qwen2.5-0.5b": {
    perplexity: { perplexity: 2408.19, bpc: 4.012, cross_entropy_nats: 7.787 },
    blimp: {
      accuracy: 0.7761,
      accuracy_morphology: 0.8889,
      accuracy_semantics: 0.7778,
      accuracy_syntax: 0.6923,
      accuracy_syntax_semantics: 0.7571,
    },
  },
};

const BASELINE_NAMES = Object.keys(BASELINES);

// ── Helpers ─────────────────────────────────────────────

function CompareValue({
  value,
  baseline,
  lowerIsBetter,
  format,
}: {
  value: number | undefined;
  baseline: number;
  lowerIsBetter: boolean;
  format: (n: number) => string;
}) {
  if (value === undefined) return <span className="layer-val">—</span>;
  const better = lowerIsBetter ? value < baseline : value > baseline;
  return (
    <span className={`layer-val ${better ? "eval-better" : "eval-worse"}`}>
      {format(value)}
    </span>
  );
}

// ── Perplexity Panel ────────────────────────────────────

function PerplexityPanel({ latest }: { latest: EvalResult | undefined }) {
  const metrics = latest?.metrics;

  return (
    <div className="panel">
      <h3 className="panel-title">Perplexity</h3>
      {!metrics ? (
        <div className="panel-empty">No perplexity evals yet.</div>
      ) : (
        <>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Perplexity</div>
              <div className="metric-value">
                {metrics.perplexity?.toFixed(2) ?? "—"}
              </div>
              {latest && <div className="metric-sub">step {latest.step}</div>}
            </div>
            <div className="metric-card">
              <div className="metric-label">BPC</div>
              <div className="metric-value">
                {metrics.bpc?.toFixed(4) ?? "—"}
              </div>
              <div className="metric-sub">bits/char</div>
            </div>
          </div>
          <div className="eval-comparison-row">
            {BASELINE_NAMES.map((name) => {
              const b = BASELINES[name]?.perplexity;
              if (!b) return null;
              return (
                <div key={name} className="eval-baseline">
                  <span className="eval-baseline-name">{name}</span>
                  <span className="eval-baseline-value">
                    ppl={b.perplexity.toFixed(1)}
                  </span>
                  <span className="eval-baseline-value">
                    bpc={b.bpc.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// ── BLiMP Panel ─────────────────────────────────────────

const BLIMP_CATEGORIES = [
  { key: "accuracy", label: "Overall" },
  { key: "accuracy_morphology", label: "Morphology" },
  { key: "accuracy_syntax", label: "Syntax" },
  { key: "accuracy_semantics", label: "Semantics" },
  { key: "accuracy_syntax_semantics", label: "Syntax-Semantics" },
];

function BlimpPanel({ latest }: { latest: EvalResult | undefined }) {
  const metrics = latest?.metrics;

  return (
    <div className="panel">
      <h3 className="panel-title">BLiMP (Linguistic Minimal Pairs)</h3>
      {!metrics ? (
        <div className="panel-empty">No BLiMP evals yet.</div>
      ) : (
        <div className="layer-stats-container">
          <div className="eval-table-header">
            <span>Category</span>
            <span>Your Model</span>
            {BASELINE_NAMES.map((n) => (
              <span key={n}>{n}</span>
            ))}
          </div>
          {BLIMP_CATEGORIES.map(({ key, label }) => (
            <div key={key} className="eval-table-row">
              <span className="layer-name">{label}</span>
              <span className="layer-val">
                {metrics[key] !== undefined
                  ? (metrics[key] * 100).toFixed(1) + "%"
                  : "—"}
              </span>
              {BASELINE_NAMES.map((name) => {
                const bVal = BASELINES[name]?.blimp?.[key];
                return (
                  <CompareValue
                    key={name}
                    value={
                      metrics[key] !== undefined ? metrics[key] : undefined
                    }
                    baseline={bVal ?? 0}
                    lowerIsBetter={false}
                    format={(v) => (v * 100).toFixed(1) + "%"}
                  />
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Eval Over Steps Chart ───────────────────────────────

function EvalChart({ evals }: { evals: EvalResult[] }) {
  const divRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!divRef.current) return;

    const perplexityEvals = evals
      .filter((e) => e.task === "perplexity" && e.metrics.perplexity != null)
      .sort((a, b) => a.step - b.step);

    const blimpEvals = evals
      .filter((e) => e.task === "blimp" && e.metrics.accuracy != null)
      .sort((a, b) => a.step - b.step);

    if (perplexityEvals.length === 0 && blimpEvals.length === 0) return;

    const allSteps = [
      ...new Set([
        ...perplexityEvals.map((e) => e.step),
        ...blimpEvals.map((e) => e.step),
      ]),
    ].sort((a, b) => a - b);

    const perpMap = new Map(
      perplexityEvals.map((e) => [e.step, e.metrics.perplexity]),
    );
    const blimpMap = new Map(
      blimpEvals.map((e) => [e.step, e.metrics.accuracy]),
    );

    const xData = Float64Array.from(allSteps);
    const perpData = allSteps.map((s) => perpMap.get(s) ?? null);
    const blimpData = allSteps.map((s) => blimpMap.get(s) ?? null);

    plotRef.current?.destroy();
    plotRef.current = new uPlot(
      {
        ...baseOpts(800, 300),
        series: [
          {},
          {
            label: "Perplexity",
            stroke: CHART_COLORS.trainLoss,
            width: 2,
            scale: "perp",
          },
          {
            label: "BLiMP Acc",
            stroke: CHART_COLORS.updateRatio,
            width: 2,
            scale: "acc",
          },
        ],
        scales: {
          x: { auto: true },
          perp: { auto: true },
          acc: { auto: true, range: [0, 1] },
        },
        axes: [
          {
            stroke: CHART_COLORS.text,
            grid: { stroke: CHART_COLORS.grid, width: 1 },
            font: "11px Inter, sans-serif",
          },
          {
            scale: "perp",
            stroke: CHART_COLORS.trainLoss,
            grid: { stroke: CHART_COLORS.grid, width: 1 },
            font: "11px Inter, sans-serif",
            label: "Perplexity",
          },
          {
            scale: "acc",
            side: 1,
            stroke: CHART_COLORS.updateRatio,
            grid: { show: false },
            font: "11px Inter, sans-serif",
            label: "Accuracy",
          },
        ],
      } as uPlot.Options,
      [xData as unknown as number[], perpData, blimpData] as uPlot.AlignedData,
      divRef.current,
    );

    return () => {
      plotRef.current?.destroy();
      plotRef.current = null;
    };
  }, [evals]);

  if (evals.length === 0) return null;

  return (
    <div className="panel">
      <h3 className="panel-title">Eval Metrics Over Training</h3>
      <div ref={divRef} />
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────

export interface EvalPageProps {
  latestPerplexity: EvalResult | undefined;
  latestBlimp: EvalResult | undefined;
  evals: EvalResult[];
}

export function EvalPage({ latestPerplexity, latestBlimp, evals }: EvalPageProps) {
  return (
    <main className="eval-layout">
      <PerplexityPanel latest={latestPerplexity} />
      <BlimpPanel latest={latestBlimp} />
      <EvalChart evals={evals} />
    </main>
  );
}
