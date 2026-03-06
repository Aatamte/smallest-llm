import type { ConnectionStatus } from "../storage";

const CONNECTION_COLORS: Record<ConnectionStatus, string> = {
  connected: "#22c55e",
  reconnecting: "#eab308",
  disconnected: "#ef4444",
};

export interface HeaderProps {
  connectionStatus: ConnectionStatus;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export function Header({
  connectionStatus,
  sidebarOpen,
  onToggleSidebar,
}: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <button
          className="sidebar-toggle-btn"
          onClick={onToggleSidebar}
          title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <span className="header-title">smallest-llm</span>
      </div>
      <div className="header-right">
        <span
          className="connection-dot"
          style={{ backgroundColor: CONNECTION_COLORS[connectionStatus] }}
          title={connectionStatus}
        />
      </div>
    </header>
  );
}
