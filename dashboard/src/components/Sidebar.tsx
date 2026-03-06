import { useAtomValue } from "jotai";
import { activePageAtom, type PageId } from "../storage";
import { subPageAtom, navigateTo } from "../storage/atoms/uiAtoms";

// ── Nav items ───────────────────────────────────────────

const PRIMARY_ITEMS: { page: PageId; label: string; icon: React.ReactNode }[] = [
  {
    page: "train",
    label: "Train",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>,
  },
  {
    page: "runs",
    label: "Runs",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg>,
  },
  {
    page: "models",
    label: "Models",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" /><line x1="8" y1="21" x2="16" y2="21" /><line x1="12" y1="17" x2="12" y2="21" /></svg>,
  },
  {
    page: "eval",
    label: "Eval",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>,
  },
  {
    page: "chat",
    label: "Chat",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>,
  },
  {
    page: "tables",
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

  function handlePrimaryClick(page: PageId) {
    if (page === "train") {
      // Go to train/metrics by default
      if (activePage !== "train") navigateTo("train", "metrics");
    } else if (page === "runs") {
      navigateTo("runs", "new");
    } else {
      navigateTo(page);
    }
  }

  const showSecondary = activePage === "train";

  return (
    <div className="sidebar-wrapper">
      <nav className="sidebar-primary">
        {PRIMARY_ITEMS.map((item) => (
          <button
            key={item.page}
            className={`sidebar-icon ${activePage === item.page ? "active" : ""}`}
            onClick={() => handlePrimaryClick(item.page)}
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
