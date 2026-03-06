import { useEffect, useCallback, useState } from "react";
import { fetchAllEvals } from "../api/client";
import type { GroupedEvals } from "./EvalContainer";
import { LeaderboardPage } from "../components/LeaderboardPage";

export function LeaderboardContainer() {
  const [allEvals, setAllEvals] = useState<GroupedEvals>({});

  const loadAllEvals = useCallback(() => {
    fetchAllEvals()
      .then((evals) => {
        const grouped: GroupedEvals = {};
        for (const e of evals) {
          const name = e.model_name;
          if (!name) continue;
          if (!grouped[name]) grouped[name] = {};
          grouped[name][e.task] = e.metrics;
        }
        setAllEvals(grouped);
      })
      .catch((e) => console.warn("Failed to fetch evals:", e));
  }, []);

  useEffect(() => {
    loadAllEvals();
  }, [loadAllEvals]);

  return <LeaderboardPage allEvals={allEvals} />;
}
