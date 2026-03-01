import { useAtomValue } from "jotai";
import { activePageAtom } from "./storage";
import { Layout } from "./components/Layout";
import { LossChartContainer } from "./containers/LossChartContainer";
import { GradientChartContainer } from "./containers/GradientChartContainer";
import { MetricsCards } from "./components/MetricsCards";
import { LayerStats } from "./components/LayerStats";
import { RunContainer } from "./containers/RunContainer";
import { LogContainer } from "./containers/LogContainer";
import { ModelsContainer } from "./containers/ModelsContainer";
import { EvalContainer } from "./containers/EvalContainer";
import { useWebSocket } from "./hooks/useWebSocket";
import { useHashRouter } from "./hooks/useHashRouter";

function MetricsPage() {
  return (
    <main className="metrics-layout">
      <LossChartContainer />
      <MetricsCards />
    </main>
  );
}

function GradientsPage() {
  return (
    <main className="gradients-layout">
      <GradientChartContainer />
      <LayerStats />
    </main>
  );
}

export default function App() {
  const page = useAtomValue(activePageAtom);
  useWebSocket();
  useHashRouter();

  return (
    <Layout>
      {page === "metrics" && <MetricsPage />}
      {page === "gradients" && <GradientsPage />}
      {page === "runs" && <RunContainer />}
      {page === "models" && <ModelsContainer />}
      {page === "eval" && <EvalContainer />}
      {page === "logs" && <LogContainer />}
    </Layout>
  );
}
