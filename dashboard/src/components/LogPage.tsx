import { useEffect, useRef } from "react";
import type { LogEntry } from "../types/metrics";

function formatTime(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, "0");
  const m = d.getMinutes().toString().padStart(2, "0");
  const s = d.getSeconds().toString().padStart(2, "0");
  return `${h}:${m}:${s}`;
}

export interface LogPageProps {
  logs: LogEntry[];
  onClear: () => void;
}

export function LogPage({ logs, onClear }: LogPageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  useEffect(() => {
    if (autoScrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs.length]);

  function handleScroll() {
    const el = containerRef.current;
    if (!el) return;
    autoScrollRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  }

  return (
    <main className="logs-layout">
      <div className="panel logs-panel">
        <div className="logs-toolbar">
          <h3 className="panel-title" style={{ marginBottom: 0 }}>Training Logs</h3>
          <span className="logs-count">{logs.length} entries</span>
          <button className="logs-clear-btn" onClick={onClear}>
            Clear
          </button>
        </div>

        <div
          className="logs-container"
          ref={containerRef}
          onScroll={handleScroll}
        >
          {logs.length === 0 ? (
            <div className="panel-empty">No training logs yet. Start a run to see output.</div>
          ) : (
            logs.map((entry, i) => (
              <div className="log-entry" key={i}>
                <span className="log-time">{formatTime(entry.timestamp)}</span>
                <span className="log-message">{entry.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </main>
  );
}
