import type { LeaderboardEntry } from "../db/queries";
import { navigateTo } from "../storage/atoms/uiAtoms";

const fmtPct = (v: number) => (v * 100).toFixed(2) + "%";
const fmtFlops = (v: number) => {
  if (v >= 1e15) return (v / 1e15).toFixed(2) + " PF";
  if (v >= 1e12) return (v / 1e12).toFixed(1) + " TF";
  if (v >= 1e9) return (v / 1e9).toFixed(1) + " GF";
  return v.toExponential(1);
};

export function LeaderboardPage({ entries }: { entries: LeaderboardEntry[] }) {
  const cols = "40px minmax(120px, 1.5fr) minmax(100px, 1fr) 100px 100px 80px";

  return (
    <main className="eval-layout">
      <div className="panel">
        <h3 className="panel-title">Composite Eval Leaderboard</h3>
        {entries.length === 0 ? (
          <div className="panel-empty">
            No composite eval scores yet. Run training with evals enabled.
          </div>
        ) : (
          <div className="eval-table">
            <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
              <span>#</span>
              <span>Run</span>
              <span>Architecture</span>
              <span>Composite</span>
              <span>FLOPs</span>
              <span>Step</span>
            </div>
            {entries.map((e, idx) => (
              <div
                key={e.run_id}
                className="eval-table-row eval-table-row-clickable"
                style={{ gridTemplateColumns: cols, cursor: "pointer" }}
                onClick={() => navigateTo("train", "metrics")}
              >
                <span className="eval-cell-val" style={{ opacity: 0.5 }}>{idx + 1}</span>
                <span className="eval-cell-model">{e.run_name}</span>
                <span className="eval-cell-val">{e.model_name}</span>
                <span className="eval-cell-val" style={{ fontWeight: 600 }}>{fmtPct(e.composite)}</span>
                <span className="eval-cell-val">{fmtFlops(e.flops)}</span>
                <span className="eval-cell-val">{e.step.toLocaleString()}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
