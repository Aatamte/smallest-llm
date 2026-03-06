import { useEffect, useState } from "react";
import { useAtomValue, useAtom, useSetAtom } from "jotai";
import { statusAtom, activeRunIdAtom, resetAtom } from "../storage";
import { navigateTo } from "../storage/atoms/uiAtoms";
import { startRun, fetchConfig, fetchPresets, fetchPreset, fetchEvalPresets, fetchEvalPreset } from "../api/client";
import { NewRunPage } from "../components/NewRunPage";

export function NewRunContainer() {
  const status = useAtomValue(statusAtom);
  const [, setActiveRunId] = useAtom(activeRunIdAtom);
  const setStatus = useSetAtom(statusAtom);
  const reset = useSetAtom(resetAtom);

  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [presets, setPresets] = useState<{ name: string; label: string; description?: string }[]>([]);
  const [activePreset, setActivePreset] = useState("quick-transformer");
  const [evalPresets, setEvalPresets] = useState<{ name: string; label: string }[]>([]);
  const [activeEvalPreset, setActiveEvalPreset] = useState("standard");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig).catch((e) => console.warn("Failed to fetch config:", e));
    fetchPresets().then(setPresets).catch((e) => console.warn("Failed to fetch presets:", e));
    fetchEvalPresets().then(setEvalPresets).catch((e) => console.warn("Failed to fetch eval presets:", e));
  }, []);

  function handlePresetChange(name: string) {
    setActivePreset(name);
    fetchPreset(name).then(setConfig).catch((e) => console.warn("Failed to fetch preset:", e));
  }

  function handleEvalPresetChange(name: string) {
    setActiveEvalPreset(name);
    // Apply eval preset fields into the training section of the config
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

  function updateConfig(section: string, key: string, value: unknown) {
    if (!config) return;
    setConfig({
      ...config,
      [section]: { ...(config[section] as Record<string, unknown>), [key]: value },
    });
  }

  function updateTopLevel(key: string, value: unknown) {
    if (!config) return;
    setConfig({ ...config, [key]: value });
  }

  function updateStage(index: number, key: string, value: unknown) {
    if (!config) return;
    const stages = (config.stages as Record<string, unknown>[] | null) ?? [];
    const updated = stages.map((s, i) => (i === index ? { ...s, [key]: value } : s));
    setConfig({ ...config, stages: updated });
  }

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      reset();
      const result = await startRun(config ?? undefined);
      setActiveRunId(result.run_id);
      setStatus("training");
      navigateTo("metrics");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setStarting(false);
    }
  }

  return (
    <NewRunPage
      status={status}
      config={config}
      presets={presets}
      activePreset={activePreset}
      onPresetChange={handlePresetChange}
      evalPresets={evalPresets}
      activeEvalPreset={activeEvalPreset}
      onEvalPresetChange={handleEvalPresetChange}
      onConfigChange={updateConfig}
      onTopLevelChange={updateTopLevel}
      onStageChange={updateStage}
      starting={starting}
      error={error}
      onStart={handleStart}
    />
  );
}
