import { useAtomValue } from "jotai";
import { activePageAtom } from "./storage";
import { Layout } from "./components/Layout";
import { LossChartContainer } from "./containers/LossChartContainer";
import { MetricChartContainer } from "./containers/MetricChartContainer";
import { GradientChartContainer } from "./containers/GradientChartContainer";
import { LayerStats } from "./components/LayerStats";
import { ActivationStats } from "./components/ActivationStats";
import { RunContainer } from "./containers/RunContainer";
import { LogContainer } from "./containers/LogContainer";
import { ModelsContainer } from "./containers/ModelsContainer";
import { EvalContainer } from "./containers/EvalContainer";
import { ChatContainer } from "./containers/ChatContainer";
import { TablesPage } from "./components/TablesPage";
import { useWebSocket } from "./hooks/useWebSocket";
import { useHashRouter } from "./hooks/useHashRouter";
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
  useWebSocket();
  useHashRouter();

  return (
    <Layout>
      {/* Chart pages stay mounted — hidden via CSS to preserve uPlot instances */}
      <main className="metrics-layout" style={page === "metrics" ? undefined : hidden}>
        <LossChartContainer />
        <div className="metric-charts-grid">
          <MetricChartContainer metricKey="lr" label="Learning Rate" color={CHART_COLORS.lr} format={(v) => v.toExponential(2)} />
          <MetricChartContainer metricKey="gradNorm" label="Grad Norm" color={CHART_COLORS.gradNorm} />
          <MetricChartContainer metricKey="tokensPerSec" label="Tokens/s" color={CHART_COLORS.updateRatio} format={fmtTokens} />
          <MetricChartContainer metricKey="bpc" label="BPC" color="#eab308" format={(v) => v.toFixed(3)} sub="bits/char" />
          <MetricChartContainer metricKey="stepTime" label="Step Time" color="#8b5cf6" format={(v) => v.toFixed(3)} sub="sec" />
        </div>
      </main>
      <main className="gradients-layout" style={page === "gradients" ? undefined : hidden}>
        <GradientChartContainer />
        <LayerStats />
        <ActivationStats />
      </main>
      {/* Other pages mount/unmount normally */}
      {page === "runs" && <RunContainer />}
      {page === "models" && <ModelsContainer />}
      {page === "eval" && <EvalContainer />}
      {page === "chat" && <ChatContainer />}
      {page === "tables" && <TablesPage />}
      {page === "logs" && <LogContainer />}
    </Layout>
  );
}
