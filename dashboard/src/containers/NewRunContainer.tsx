import { useEffect, useState } from "react";
import { useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { navigateTo } from "../storage/atoms/uiAtoms";
import { useStatus } from "../db/hooks";
import { startRun, fetchPresets, fetchPreset, fetchEvalPresets, fetchEvalPreset } from "../api/client";
import { NewRunPage } from "../components/NewRunPage";

export function NewRunContainer() {
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);
  const status = useStatus(activeRunId);

  const [presets, setPresets] = useState<{ name: string; label: string; description?: string }[]>([]);
  const [activePreset, setActivePreset] = useState("quick-transformer");
  const [evalPresets, setEvalPresets] = useState<{ name: string; label: string }[]>([]);
  const [activeEvalPreset, setActiveEvalPreset] = useState("standard");
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPresets().then(setPresets).catch((e) => console.warn("Failed to fetch presets:", e));
    fetchEvalPresets().then(setEvalPresets).catch((e) => console.warn("Failed to fetch eval presets:", e));
    // Load initial preset config
    fetchPreset("quick-transformer").then(setConfig).catch((e) => console.warn("Failed to fetch preset:", e));
  }, []);

  function handlePresetChange(name: string) {
    setActivePreset(name);
    fetchPreset(name).then(setConfig).catch((e) => console.warn("Failed to fetch preset:", e));
  }

  function handleEvalPresetChange(name: string) {
    setActiveEvalPreset(name);
    fetchEvalPreset(name).then((evalFields) => {
      setConfig((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          training: { ...(prev.training as Record<string, unknown>), ...evalFields },
        };
      });
    }).catch((e) => console.warn("Failed to fetch eval preset:", e));
  }

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      const result = await startRun(config ?? undefined);
      setActiveRunId(result.run_id);
      navigateTo("train", "metrics");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setStarting(false);
    }
  }

  return (
    <NewRunPage
      status={status}
      presets={presets}
      activePreset={activePreset}
      onPresetChange={handlePresetChange}
      evalPresets={evalPresets}
      activeEvalPreset={activeEvalPreset}
      onEvalPresetChange={handleEvalPresetChange}
      starting={starting}
      error={error}
      onStart={handleStart}
    />
  );
}
