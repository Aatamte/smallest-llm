import { useAtomValue } from "jotai";
import {
  currentTrainLossAtom,
  currentValLossAtom,
  bestValLossAtom,
  tokensPerSecAtom,
  tokensSeenAtom,
  bpcAtom,
} from "../storage";

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
  const trainLoss = useAtomValue(currentTrainLossAtom);
  const valLoss = useAtomValue(currentValLossAtom);
  const bestVal = useAtomValue(bestValLossAtom);
  const tokSec = useAtomValue(tokensPerSecAtom);
  const tokensSeen = useAtomValue(tokensSeenAtom);
  const bpc = useAtomValue(bpcAtom);

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
        <Card label="Train Loss" value={trainLoss.toFixed(4)} />
        <Card label="Val Loss" value={valLoss > 0 ? valLoss.toFixed(4) : "—"} />
        <Card
          label="Best Val"
          value={bestVal < Infinity ? bestVal.toFixed(4) : "—"}
        />
        <Card label="BPC" value={bpc > 0 ? bpc.toFixed(3) : "—"} sub="bits/char" />
        <Card label="Tokens/s" value={formatTokens(Math.round(tokSec))} />
        <Card label="Tokens Seen" value={formatTokens(tokensSeen)} />
      </div>
    </div>
  );
}
