import { useEffect } from "react";
import { useStore } from "jotai";
import { useAtomValue } from "jotai";
import { fetchTrainingState } from "../api/client";
import {
  statusAtom,
  connectionStatusAtom,
  layerStatsAtom,
  stepsAtom,
  generationsAtom,
  startTimeAtom,
  pushStepAtom,
  hydrateStepsAtom,
  addGenerationAtom,
  addLogAtom,
  activeRunIdAtom,
} from "../storage";
import { createWebSocket } from "../ws/client";
import type { StepMetrics } from "../types/metrics";

/**
 * Connects to the backend: fetches historical state via REST,
 * then subscribes to live updates via WebSocket.
 * Re-runs when activeRunId changes.
 */
export function useWebSocket() {
  const store = useStore();
  const runId = useAtomValue(activeRunIdAtom);

  useEffect(() => {
    // Clear previous run data
    store.set(stepsAtom, []);
    store.set(layerStatsAtom, []);
    store.set(generationsAtom, []);
    store.set(statusAtom, "idle");
    store.set(startTimeAtom, Date.now());

    // Hydrate from REST
    if (runId !== null) {
      fetchTrainingState(runId)
        .then((state) => {
          if (state.status) store.set(statusAtom, state.status);
          if (state.steps && state.steps.length > 0) {
            store.set(hydrateStepsAtom, state.steps as StepMetrics[]);
          }
        })
        .catch((err) => console.warn("Failed to load historical state:", err));
    }

    // Live updates via WebSocket
    const cleanup = createWebSocket({
      onStep: (data) => store.set(pushStepAtom, data),
      onLayers: (data) => store.set(layerStatsAtom, data),
      onGeneration: (data) => store.set(addGenerationAtom, data),
      onStatus: (data) => store.set(statusAtom, data),
      onLog: (data) => store.set(addLogAtom, {
        timestamp: Date.now(),
        level: data.level,
        source: "train",
        message: data.message,
      }),
      onConnectionChange: (data) => store.set(connectionStatusAtom, data),
    });

    return cleanup;
  }, [store, runId]);
}
