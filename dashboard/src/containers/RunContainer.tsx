import { useEffect, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { statusAtom, activeRunIdAtom, resetAtom, subPageAtom } from "../storage";
import { navigateTo } from "../storage/atoms/uiAtoms";
import { startRun, fetchConfig, fetchRuns, fetchPresets, fetchPreset } from "../api/client";
import { RunPage } from "../components/RunPage";

interface Run {
  id: number;
  name: string;
  status: string;
  created_at: string;
}

export function RunContainer() {
  const sub = useAtomValue(subPageAtom);
  const status = useAtomValue(statusAtom);
  const setActiveRunId = useSetAtom(activeRunIdAtom);
  const setStatus = useSetAtom(statusAtom);
  const reset = useSetAtom(resetAtom);

  // Run list state
  const [runs, setRuns] = useState<Run[]>([]);

  // New run state
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [presets, setPresets] = useState<{ name: string; label: string }[]>([]);
  const [activePreset, setActivePreset] = useState("default");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sub !== "new") {
      fetchRuns().then(setRuns).catch((e) => console.warn("Failed to fetch runs:", e));
    }
  }, [sub]);

  useEffect(() => {
    if (sub === "new") {
      fetchConfig().then(setConfig).catch((e) => console.warn("Failed to fetch config:", e));
      fetchPresets().then(setPresets).catch((e) => console.warn("Failed to fetch presets:", e));
    }
  }, [sub]);

  function selectRun(id: number) {
    setActiveRunId(id);
    navigateTo("metrics");
  }

  function handlePresetChange(name: string) {
    setActivePreset(name);
    fetchPreset(name).then(setConfig).catch((e) => console.warn("Failed to fetch preset:", e));
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
    <RunPage
      sub={sub}
      runs={runs}
      onSelectRun={selectRun}
      status={status}
      config={config}
      presets={presets}
      activePreset={activePreset}
      onPresetChange={handlePresetChange}
      onConfigChange={updateConfig}
      onTopLevelChange={updateTopLevel}
      starting={starting}
      error={error}
      onStart={handleStart}
    />
  );
}
