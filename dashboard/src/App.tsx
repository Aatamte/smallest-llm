import { useState } from "react";
import { useAtomValue } from "jotai";
import { activePageAtom, activeRunIdAtom } from "./storage";
import { subPageAtom } from "./storage/atoms/uiAtoms";
import { Layout } from "./components/Layout";
import { TrainingInfoBarContainer } from "./containers/TrainingInfoBarContainer";
import { LossChartContainer } from "./containers/LossChartContainer";
import { MetricChartContainer } from "./containers/MetricChartContainer";
import { SelectableMetricChart } from "./containers/SelectableMetricChart";
import { EvalChartsSection } from "./containers/EvalChartsSection";
import { RunContainer } from "./containers/RunContainer";
import { LogContainer } from "./containers/LogContainer";
import { ModelsContainer } from "./containers/ModelsContainer";
import { EvalContainer } from "./containers/EvalContainer";
import { ChatContainer } from "./containers/ChatContainer";
import { LeaderboardContainer } from "./containers/LeaderboardContainer";
import { TablesPage } from "./components/TablesPage";
import { CompareRunsSelector } from "./components/CompareRunsSelector";
import { useWebSocket } from "./hooks/useWebSocket";
import { useHashRouter } from "./hooks/useHashRouter";
import { useRuns } from "./db/hooks";
import { CHART_COLORS } from "./types/chart";

const hidden = { display: "none" } as const;

const fmtTokens = (n: number) => {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
};

export default function App() {
  const page = useAtomValue(activePageAtom);
  const sub = useAtomValue(subPageAtom);
  const activeRunId = useAtomValue(activeRunIdAtom);
  const runs = useRuns();
  useWebSocket();
  useHashRouter();

  const [compareRunIds, setCompareRunIds] = useState<number[]>([]);

  const isTrain = page === "train";
  const trainSub = isTrain ? (sub ?? "metrics") : null;

  return (
    <Layout>
      {/* Training pages: #/train/metrics, #/train/logs */}
      {isTrain && <TrainingInfoBarContainer />}

      <main className="metrics-layout" style={trainSub === "metrics" ? undefined : hidden}>
        <CompareRunsSelector
          runs={runs}
          activeRunId={activeRunId}
          selected={compareRunIds}
          onChange={setCompareRunIds}
        />
        <LossChartContainer compareRunIds={compareRunIds} runs={runs} />
        <div className="metric-charts-grid">
          <MetricChartContainer metricKey="lr" label="Learning Rate" color={CHART_COLORS.lr} format={(v) => v.toExponential(2)} />
          <MetricChartContainer metricKey="gradNorm" label="Grad Norm" color={CHART_COLORS.gradNorm} allowLog />
          <MetricChartContainer metricKey="tokensPerSec" label="Tokens/s" color={CHART_COLORS.updateRatio} format={fmtTokens} />
          <MetricChartContainer metricKey="bpc" label="BPC" color="#eab308" format={(v) => v.toFixed(3)} sub="bits/char" allowLog />
          <MetricChartContainer metricKey="stepTime" label="Step Time" color="#8b5cf6" format={(v) => v.toFixed(3)} sub="sec" allowLog />
          <SelectableMetricChart />
        </div>
        <EvalChartsSection compareRunIds={compareRunIds} runs={runs} />
      </main>
      {trainSub === "logs" && <LogContainer />}

      {/* Other pages */}
      {page === "runs" && <RunContainer />}
      {page === "models" && <ModelsContainer />}
      {page === "eval" && <EvalContainer />}
      {page === "chat" && <ChatContainer />}
      {page === "leaderboard" && <LeaderboardContainer />}
      {page === "tables" && <TablesPage />}
    </Layout>
  );
}
