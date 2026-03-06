import { useCallback, useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { subPageAtom } from "../storage/atoms/uiAtoms";
import { navigateTo } from "../storage/atoms/uiAtoms";
import { useQuery } from "../db/hooks";
import { getStatus, getRuns } from "../db/queries";
import { startRun, deleteRun, bulkDeleteRuns, fetchConfig, fetchPresets, fetchPreset } from "../api/client";
import { RunPage } from "../components/RunPage";

export function RunContainer() {
  const sub = useAtomValue(subPageAtom);
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);

  const status = useQuery(useCallback(() => getStatus(activeRunId), [activeRunId]));
  const runs = useQuery(useCallback(() => getRuns(), []));

  // New run state
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [presets, setPresets] = useState<{ name: string; label: string }[]>([]);
  const [activePreset, setActivePreset] = useState("default");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sub === "new") {
      fetchConfig().then(setConfig).catch((e) => console.warn("Failed to fetch config:", e));
      fetchPresets().then(setPresets).catch((e) => console.warn("Failed to fetch presets:", e));
    }
  }, [sub]);

  function selectRun(id: number) {
    setActiveRunId(id);
    navigateTo("train", "metrics");
  }

  async function handleDeleteRun(id: number) {
    try {
      await deleteRun(id);
      if (activeRunId === id) {
        setActiveRunId(null);
      }
    } catch (err) {
      console.warn("Failed to delete run:", err);
    }
  }

  async function handleBulkDelete(ids: number[]) {
    try {
      await bulkDeleteRuns(ids);
      const idSet = new Set(ids);
      if (activeRunId != null && idSet.has(activeRunId)) {
        setActiveRunId(null);
      }
    } catch (err) {
      console.warn("Failed to bulk delete runs:", err);
    }
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
      onDeleteRun={handleDeleteRun}
      onBulkDelete={handleBulkDelete}
    />
  );
}
