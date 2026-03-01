import { useEffect } from "react";
import { useStore } from "jotai";
import { useAtomValue } from "jotai";
import { fetchTrainingState } from "../api/client";
import {
  statusAtom,
  connectionStatusAtom,
  layerStatsAtom,
  activationStatsAtom,
  stepsAtom,
  generationsAtom,
  startTimeAtom,
  maxStepsAtom,
  hydratingAtom,
  tokensPerStepAtom,
  pushStepAtom,
  hydrateStepsAtom,
  addGenerationAtom,
  addLogAtom,
  activeRunIdAtom,
  evalsAtom,
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
    store.set(activationStatsAtom, []);
    store.set(generationsAtom, []);
    store.set(evalsAtom, []);
    store.set(statusAtom, "idle");
    store.set(tokensPerStepAtom, 0);

    // Hydrate from REST
    if (runId !== null) {
      store.set(hydratingAtom, true);
      fetchTrainingState(runId)
        .then((state) => {
          if (state.status) store.set(statusAtom, state.status);
          if (state.startTime) store.set(startTimeAtom, state.startTime);
          if (state.maxSteps) store.set(maxStepsAtom, state.maxSteps);
          if (state.steps && state.steps.length > 0) {
            store.set(hydrateStepsAtom, state.steps as StepMetrics[]);

            // Derive tokens-per-step from the last step's data
            const last = state.steps[state.steps.length - 1];
            if (last.tokensSeen && last.step > 0) {
              store.set(tokensPerStepAtom, Math.round(last.tokensSeen / last.step));
            }
          }
        })
        .catch((err) => console.warn("Failed to load historical state:", err))
        .finally(() => store.set(hydratingAtom, false));
    } else {
      store.set(startTimeAtom, Date.now());
    }

    // Live updates via WebSocket
    const cleanup = createWebSocket({
      onStep: (data) => store.set(pushStepAtom, data),
      onLayers: (data) => store.set(layerStatsAtom, data),
      onActivations: (data) => store.set(activationStatsAtom, data),
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
