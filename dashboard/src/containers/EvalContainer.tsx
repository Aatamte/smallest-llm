import { useEffect, useState } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { fetchEvals, type EvalResult } from "../api/client";
import { EvalPage } from "../components/EvalPage";

function latestByTask(evals: EvalResult[]): Record<string, EvalResult> {
  const latest: Record<string, EvalResult> = {};
  for (const e of evals) {
    if (!latest[e.task] || e.step > latest[e.task].step) {
      latest[e.task] = e;
    }
  }
  return latest;
}

export function EvalContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const [evals, setEvals] = useState<EvalResult[]>([]);

  useEffect(() => {
    if (runId != null) {
      fetchEvals(runId).then(setEvals).catch((e) => console.warn("Failed to fetch evals:", e));
    } else {
      setEvals([]);
    }
  }, [runId]);

  const latest = latestByTask(evals);

  return (
    <EvalPage
      latestPerplexity={latest["perplexity"]}
      latestBlimp={latest["blimp"]}
      evals={evals}
    />
  );
}
