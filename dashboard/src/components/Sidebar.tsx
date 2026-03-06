import { useAtomValue } from "jotai";
import { activePageAtom } from "../storage";
import { subPageAtom, navigateTo } from "../storage/atoms/uiAtoms";

// ── Nav items ───────────────────────────────────────────

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const PRIMARY_ITEMS: NavItem[] = [
  {
    id: "train",
    label: "Train",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>,
  },
  {
    id: "runs",
    label: "Runs",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12h4l3 9L14 3l3 9h4" /></svg>,
  },
  {
    id: "new-run",
    label: "New Run",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>,
  },
  {
    id: "models",
    label: "Models",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></svg>,
  },
  {
    id: "eval",
    label: "Eval",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>,
  },
  {
    id: "chat",
    label: "Chat",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>,
  },
  {
    id: "leaderboard",
    label: "Leaderboard",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="14" width="6" height="8" /><rect x="9" y="4" width="6" height="18" /><rect x="16" y="9" width="6" height="13" /></svg>,
  },
  {
    id: "tables",
    label: "Tables",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="3" y1="9" x2="21" y2="9" /><line x1="3" y1="15" x2="21" y2="15" /><line x1="9" y1="3" x2="9" y2="21" /></svg>,
  },
];

const TRAIN_SUBS: { sub: string; label: string }[] = [
  { sub: "metrics", label: "Metrics" },
  { sub: "logs", label: "Logs" },
];

export function Sidebar() {
  const activePage = useAtomValue(activePageAtom);
  const activeSub = useAtomValue(subPageAtom);

  function isActive(id: string): boolean {
    if (id === "new-run") return activePage === "runs" && activeSub === "new";
    if (id === "runs") return activePage === "runs" && activeSub !== "new";
    return activePage === id;
  }

  function handleClick(id: string) {
    if (id === "train") {
      if (activePage !== "train") navigateTo("train", "metrics");
    } else if (id === "new-run") {
      navigateTo("runs", "new");
    } else {
      navigateTo(id as Parameters<typeof navigateTo>[0]);
    }
  }

  const showSecondary = activePage === "train";

  return (
    <div className="sidebar-wrapper">
      <nav className="sidebar-primary">
        {PRIMARY_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`sidebar-icon ${isActive(item.id) ? "active" : ""}`}
            onClick={() => handleClick(item.id)}
            title={item.label}
          >
            {item.icon}
            <span className="sidebar-icon-label">{item.label}</span>
          </button>
        ))}
      </nav>

      {showSecondary && (
        <nav className="sidebar-secondary">
          <div className="sidebar-secondary-title">Training</div>
          {TRAIN_SUBS.map((item) => (
            <button
              key={item.sub}
              className={`sidebar-secondary-item ${activeSub === item.sub ? "active" : ""}`}
              onClick={() => navigateTo("train", item.sub)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      )}
    </div>
  );
}
