import { useEffect, useState, useCallback } from "react";
import {
  fetchChatStatus,
  loadChatModel,
  generateChat,
  unloadChatModel,
  fetchAvailableModels,
  fetchRuns,
  fetchCheckpoints,
  type ChatStatus,
} from "../api/client";
import { ChatPage, type ChatMessage, type ModelOption } from "../components/ChatPage";

export function ChatContainer() {
  const [status, setStatus] = useState<ChatStatus>({
    loaded: false,
    source: null,
    name: null,
  });
  const [models, setModels] = useState<ModelOption[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch chat status + available models on mount
  useEffect(() => {
    fetchChatStatus().then(setStatus).catch(() => {});

    // Build model options: HF models + checkpoints from runs
    async function loadOptions() {
      const opts: ModelOption[] = [];

      // HF models
      try {
        const hfModels = await fetchAvailableModels();
        for (const m of hfModels) {
          opts.push({
            label: `${m.name} (${m.hf_id})`,
            group: "HF Models",
            value: { source: "hf" as const, model_name: m.name },
          });
        }
      } catch {}

      // Checkpoints from runs
      try {
        const runs = await fetchRuns();
        for (const run of runs.slice(0, 10)) {
          try {
            const checkpoints = await fetchCheckpoints(run.id);
            for (const cp of checkpoints) {
              opts.push({
                label: `${run.name} — step ${cp.step}`,
                group: "Checkpoints",
                value: { source: "checkpoint" as const, run_id: run.id, step: cp.step },
              });
            }
          } catch {}
        }
      } catch {}

      setModels(opts);
    }
    loadOptions();
  }, []);

  const handleLoad = useCallback(
    async (option: ModelOption["value"]) => {
      setLoading(true);
      setError(null);
      try {
        const result = await loadChatModel(option);
        setStatus({ loaded: true, source: option.source, name: result.name });
        setMessages([]);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleUnload = useCallback(async () => {
    try {
      await unloadChatModel();
      setStatus({ loaded: false, source: null, name: null });
      setMessages([]);
    } catch {}
  }, []);

  const handleSend = useCallback(
    async (prompt: string, params: { max_tokens: number; temperature: number; top_k: number }) => {
      setMessages((prev) => [...prev, { role: "user", content: prompt }]);
      setGenerating(true);
      setError(null);
      try {
        const result = await generateChat(prompt, params);
        setMessages((prev) => [...prev, { role: "model", content: result.text }]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setMessages((prev) => [...prev, { role: "model", content: `Error: ${msg}` }]);
      } finally {
        setGenerating(false);
      }
    },
    [],
  );

  return (
    <ChatPage
      status={status}
      models={models}
      messages={messages}
      loading={loading}
      generating={generating}
      error={error}
      onLoad={handleLoad}
      onUnload={handleUnload}
      onSend={handleSend}
    />
  );
}
