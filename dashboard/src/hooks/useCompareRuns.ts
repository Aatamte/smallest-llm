import { useEffect } from "react";
import { useStore, useAtomValue } from "jotai";
import { compareRunIdsAtom, compareRunsDataAtom, type CompareRunData } from "../storage";
import { fetchTrainingState, fetchEvals } from "../api/client";

/**
 * Watches compareRunIdsAtom and fetches training state for each selected run.
 * Populates compareRunsDataAtom with the results.
 * IDs are persisted (survive refresh), data is fetched fresh each time.
 */
export function useCompareRuns() {
  const store = useStore();
  const compareIds = useAtomValue(compareRunIdsAtom);

  useEffect(() => {
    // Abort tracking — if compareIds changes, abandon in-flight fetches
    let cancelled = false;

    const current = store.get(compareRunsDataAtom);
    const idSet = new Set(compareIds);

    // Keep entries that are still selected and already fetched
    const kept: Record<number, CompareRunData> = {};
    for (const [id, data] of Object.entries(current)) {
      if (idSet.has(Number(id))) {
        kept[Number(id)] = data;
      }
    }

    // Determine which need fetching
    const toFetch = compareIds.filter((id) => !(id in kept));

    if (toFetch.length === 0) {
      // Just clean removed entries
      if (Object.keys(kept).length !== Object.keys(current).length) {
        store.set(compareRunsDataAtom, kept);
      }
      return;
    }

    // Set cleaned state immediately (removes deselected runs)
    store.set(compareRunsDataAtom, kept);

    // Fetch all needed runs, merge results atomically
    Promise.all(
      toFetch.map((runId) =>
        Promise.all([
          fetchTrainingState(runId),
          fetchEvals(runId).catch(() => []),
        ]).then(([state, evals]): [number, CompareRunData] => [
          runId,
          {
            name: state.experimentName,
            series: state.series ?? {},
            evals: evals.map((e) => ({
              step: e.step,
              task: e.task,
              metrics: e.metrics,
            })),
          },
        ]),
      ),
    ).then((results) => {
      if (cancelled) return;
      const prev = store.get(compareRunsDataAtom);
      const merged = { ...prev };
      for (const [id, data] of results) {
        merged[id] = data;
      }
      store.set(compareRunsDataAtom, merged);
    }).catch((err) => {
      console.warn("Failed to fetch compare runs:", err);
    });

    return () => { cancelled = true; };
  }, [store, compareIds]);
}
