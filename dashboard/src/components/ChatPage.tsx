import { useState, useRef, useEffect } from "react";
import type { ChatStatus } from "../api/client";

export interface ChatMessage {
  role: "user" | "model";
  content: string;
}

export interface ModelOption {
  label: string;
  group: string;
  value:
    | { source: "hf"; model_name: string }
    | { source: "checkpoint"; run_id: number; step: number };
}

export interface ChatPageProps {
  status: ChatStatus;
  models: ModelOption[];
  messages: ChatMessage[];
  loading: boolean;
  generating: boolean;
  error: string | null;
  onLoad: (option: ModelOption["value"]) => void;
  onUnload: () => void;
  onSend: (prompt: string, params: { max_tokens: number; temperature: number; top_k: number }) => void;
}

export function ChatPage({
  status,
  models,
  messages,
  loading,
  generating,
  error,
  onLoad,
  onUnload,
  onSend,
}: ChatPageProps) {
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleLoad = () => {
    if (selectedIdx < 0 || selectedIdx >= models.length) return;
    onLoad(models[selectedIdx].value);
  };

  const handleSend = () => {
    if (!prompt.trim() || !status.loaded || generating) return;
    onSend(prompt.trim(), { max_tokens: maxTokens, temperature, top_k: topK });
    setPrompt("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Group models by group name
  const groups = new Map<string, { idx: number; option: ModelOption }[]>();
  models.forEach((m, idx) => {
    const list = groups.get(m.group) ?? [];
    list.push({ idx, option: m });
    groups.set(m.group, list);
  });

  return (
    <main className="chat-layout">
      {/* Model bar */}
      <div className="panel chat-model-bar">
        <div className="chat-model-row">
          <select
            className="eval-select"
            value={selectedIdx}
            onChange={(e) => setSelectedIdx(Number(e.target.value))}
            disabled={loading}
          >
            <option value={-1}>Select a model...</option>
            {[...groups.entries()].map(([group, items]) => (
              <optgroup key={group} label={group}>
                {items.map(({ idx, option }) => (
                  <option key={idx} value={idx}>
                    {option.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <button
            className="eval-run-btn"
            onClick={handleLoad}
            disabled={loading || selectedIdx < 0}
          >
            {loading ? "Loading..." : "Load"}
          </button>
          {status.loaded && (
            <button className="chat-unload-btn" onClick={onUnload}>
              Unload
            </button>
          )}
          {status.loaded && (
            <span className="chat-status-loaded">
              {status.name}
            </span>
          )}
          {error && <span className="eval-error-text">{error}</span>}
        </div>
      </div>

      {/* Messages */}
      <div className="panel chat-messages-panel">
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="panel-empty">
              {status.loaded
                ? "Type a prompt below to start generating."
                : "Load a model to start chatting."}
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`chat-message ${msg.role === "user" ? "chat-user" : "chat-model"}`}
            >
              <span className="chat-role">{msg.role === "user" ? "You" : "Model"}</span>
              <pre className="chat-content">{msg.content}</pre>
            </div>
          ))}
          {generating && (
            <div className="chat-message chat-model">
              <span className="chat-role">Model</span>
              <span className="chat-generating">Generating...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="panel chat-input-panel">
        <div className="chat-input-row">
          <textarea
            className="chat-textarea"
            placeholder={status.loaded ? "Type a prompt..." : "Load a model first"}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!status.loaded || generating}
            rows={2}
          />
          <button
            className="eval-run-btn chat-send-btn"
            onClick={handleSend}
            disabled={!status.loaded || generating || !prompt.trim()}
          >
            {generating ? "..." : "Send"}
          </button>
        </div>
        <div className="chat-params">
          <label className="chat-param">
            <span>Temperature</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
          </label>
          <label className="chat-param">
            <span>Max Tokens</span>
            <input
              type="number"
              min={1}
              max={1024}
              step={1}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </label>
          <label className="chat-param">
            <span>Top-K</span>
            <input
              type="number"
              min={1}
              max={500}
              step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
        </div>
      </div>
    </main>
  );
}
