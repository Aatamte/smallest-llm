import type { GroupedEvals } from "../containers/EvalContainer";

// Per-task: which metric to rank by, direction, and how to display each metric
interface MetricDef {
  key: string;
  label: string;
  format: (v: number) => string;
}

interface TaskDef {
  rankBy: string;
  lowerIsBetter: boolean;
  metrics: MetricDef[];
}

const fmt2 = (v: number) => v.toFixed(2);
const fmt4 = (v: number) => v.toFixed(4);
const fmtPct = (v: number) => (v * 100).toFixed(1) + "%";

const KNOWN_TASKS: Record<string, TaskDef> = {
  quick_loss: {
    rankBy: "top1_accuracy", lowerIsBetter: false,
    metrics: [
      { key: "top1_accuracy", label: "Top-1 Acc", format: fmtPct },
      { key: "top5_accuracy", label: "Top-5 Acc", format: fmtPct },
      { key: "loss", label: "Loss", format: fmt4 },
      { key: "perplexity", label: "PPL", format: fmt2 },
      { key: "entropy", label: "Entropy", format: fmt4 },
    ],
  },
  perplexity: {
    rankBy: "perplexity", lowerIsBetter: true,
    metrics: [
      { key: "perplexity", label: "Perplexity", format: fmt2 },
      { key: "bpc", label: "BPC", format: fmt4 },
    ],
  },
  blimp: {
    rankBy: "accuracy", lowerIsBetter: false,
    metrics: [
      { key: "accuracy", label: "Overall", format: fmtPct },
      { key: "accuracy_morphology", label: "Morphology", format: fmtPct },
      { key: "accuracy_syntax", label: "Syntax", format: fmtPct },
      { key: "accuracy_semantics", label: "Semantics", format: fmtPct },
    ],
  },
  lambada: {
    rankBy: "accuracy", lowerIsBetter: false,
    metrics: [
      { key: "accuracy", label: "Accuracy", format: fmtPct },
      { key: "target_perplexity", label: "Target PPL", format: fmt2 },
    ],
  },
};

function getTaskDef(taskName: string): TaskDef {
  if (KNOWN_TASKS[taskName]) return KNOWN_TASKS[taskName];
  return {
    rankBy: "acc_norm", lowerIsBetter: false,
    metrics: [
      { key: "acc_norm", label: "Acc (norm)", format: fmtPct },
      { key: "acc", label: "Acc", format: fmtPct },
    ],
  };
}

function prettyTaskName(task: string): string {
  if (task === "quick_loss") return "Quick Eval";
  if (task === "perplexity") return "Perplexity";
  if (task === "blimp") return "BLiMP";
  if (task === "lambada") return "LAMBADA";
  if (task.startsWith("harness/")) return task.replace("harness/", "");
  return task;
}

function RankedTaskPanel({ task, allEvals }: { task: string; allEvals: GroupedEvals }) {
  const def = getTaskDef(task);
  const models = Object.keys(allEvals).filter((m) => allEvals[m][task] != null);
  if (models.length === 0) return null;

  const sorted = [...models].sort((a, b) => {
    const va = allEvals[a][task]?.[def.rankBy] ?? (def.lowerIsBetter ? Infinity : -Infinity);
    const vb = allEvals[b][task]?.[def.rankBy] ?? (def.lowerIsBetter ? Infinity : -Infinity);
    return def.lowerIsBetter ? va - vb : vb - va;
  });

  const cols = `40px minmax(120px, 1.5fr) repeat(${def.metrics.length}, 1fr)`;

  return (
    <div className="panel">
      <h3 className="panel-title">{prettyTaskName(task)}</h3>
      <div className="eval-table">
        <div className="eval-table-header" style={{ gridTemplateColumns: cols }}>
          <span>#</span>
          <span>Model</span>
          {def.metrics.map((m) => (
            <span key={m.key}>{m.label}</span>
          ))}
        </div>
        {sorted.map((name, idx) => {
          const taskMetrics = allEvals[name][task];
          return (
            <div key={name} className="eval-table-row" style={{ gridTemplateColumns: cols }}>
              <span className="eval-cell-val" style={{ opacity: 0.5 }}>{idx + 1}</span>
              <span className="eval-cell-model">{name}</span>
              {def.metrics.map((m) => {
                const v = taskMetrics?.[m.key];
                return (
                  <span key={m.key} className="eval-cell-val">
                    {v != null ? m.format(v) : "\u2014"}
                  </span>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function LeaderboardPage({ allEvals }: { allEvals: GroupedEvals }) {
  const taskSet = new Set<string>();
  for (const tasks of Object.values(allEvals)) {
    for (const t of Object.keys(tasks)) taskSet.add(t);
  }

  const knownOrder = ["quick_loss", "perplexity", "blimp", "lambada"];
  const ordered: string[] = [];
  for (const t of knownOrder) {
    if (taskSet.has(t)) ordered.push(t);
  }
  const rest = [...taskSet].filter((t) => !knownOrder.includes(t)).sort();
  ordered.push(...rest);

  return (
    <main className="eval-layout">
      {ordered.length === 0 ? (
        <div className="panel">
          <div className="panel-empty">No eval results yet. Run evaluations from the Eval page.</div>
        </div>
      ) : (
        ordered.map((task) => (
          <RankedTaskPanel key={task} task={task} allEvals={allEvals} />
        ))
      )}
    </main>
  );
}
