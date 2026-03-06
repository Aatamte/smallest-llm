import { useEffect, useState } from "react";
import { useAtomValue, useAtom } from "jotai";
import { activeRunIdAtom } from "../storage";
import { subPageAtom } from "../storage/atoms/uiAtoms";
import { navigateTo } from "../storage/atoms/uiAtoms";
import { useStatus, useRuns } from "../db/hooks";
import { startRun, deleteRun, bulkDeleteRuns, fetchPresets, fetchPreset, fetchEvalPresets, fetchFlopsBudgets } from "../api/client";
import { RunPage } from "../components/RunPage";

export function RunContainer() {
  const sub = useAtomValue(subPageAtom);
  const [activeRunId, setActiveRunId] = useAtom(activeRunIdAtom);

  const status = useStatus(activeRunId);
  const runs = useRuns();

  // New run state
  const [presets, setPresets] = useState<{ name: string; label: string; description?: string }[]>([]);
  const [activePreset, setActivePreset] = useState("improved-mamba3");
  const [evalPresets, setEvalPresets] = useState<{ name: string; label: string }[]>([]);
  const [activeEvalPreset, setActiveEvalPreset] = useState("standard");
  const [flopsBudgets, setFlopsBudgets] = useState<{ name: string; label: string }[]>([]);
  const [activeFlopsBudget, setActiveFlopsBudget] = useState("standard");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sub === "new") {
      fetchPresets().then(setPresets).catch((e) => console.warn("Failed to fetch presets:", e));
      fetchEvalPresets().then(setEvalPresets).catch((e) => console.warn("Failed to fetch eval presets:", e));
      fetchFlopsBudgets().then(setFlopsBudgets).catch((e) => console.warn("Failed to fetch FLOPs budgets:", e));
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
  }

  function handleEvalPresetChange(name: string) {
    setActiveEvalPreset(name);
  }

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      // Fetch the selected preset config, then merge eval preset name and start
      const config = await fetchPreset(activePreset);
      const result = await startRun({ ...config, eval_preset: activeEvalPreset, flops_budget: activeFlopsBudget });
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
      presets={presets}
      activePreset={activePreset}
      onPresetChange={handlePresetChange}
      evalPresets={evalPresets}
      activeEvalPreset={activeEvalPreset}
      onEvalPresetChange={handleEvalPresetChange}
      flopsBudgets={flopsBudgets}
      activeFlopsBudget={activeFlopsBudget}
      onFlopsBudgetChange={setActiveFlopsBudget}
      starting={starting}
      error={error}
      onStart={handleStart}
      onDeleteRun={handleDeleteRun}
      onBulkDelete={handleBulkDelete}
    />
  );
}
