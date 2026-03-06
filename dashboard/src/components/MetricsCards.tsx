import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getLatestMetric } from "../db/queries";

interface CardProps {
  label: string;
  value: string;
  sub?: string;
}

function Card({ label, value, sub }: CardProps) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export function MetricsCards() {
  const runId = useAtomValue(activeRunIdAtom);

  const trainLoss = useQuery(useCallback(() => getLatestMetric(runId, "trainLoss"), [runId]));
  const valLoss = useQuery(useCallback(() => getLatestMetric(runId, "valLoss"), [runId]));
  const bpc = useQuery(useCallback(() => getLatestMetric(runId, "bpc"), [runId]));
  const tokSec = useQuery(useCallback(() => getLatestMetric(runId, "tokensPerSec"), [runId]));
  const tokensSeen = useQuery(useCallback(() => getLatestMetric(runId, "tokensSeen"), [runId]));

  const formatTokens = (n: number) => {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toString();
  };

  return (
    <div className="panel metrics-cards-panel">
      <h3 className="panel-title">Key Metrics</h3>
      <div className="metrics-grid">
        <Card label="Train Loss" value={trainLoss > 0 ? trainLoss.toFixed(4) : "—"} />
        <Card label="Val Loss" value={valLoss > 0 ? valLoss.toFixed(4) : "—"} />
        <Card label="BPC" value={bpc > 0 ? bpc.toFixed(3) : "—"} sub="bits/char" />
        <Card label="Tokens/s" value={tokSec > 0 ? formatTokens(Math.round(tokSec)) : "—"} />
        <Card label="Tokens Seen" value={tokensSeen > 0 ? formatTokens(tokensSeen) : "—"} />
      </div>
    </div>
  );
}
