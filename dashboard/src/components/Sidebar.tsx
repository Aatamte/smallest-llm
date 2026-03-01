import { useAtomValue } from "jotai";
import { activePageAtom, type PageId } from "../storage";
import { sidebarTabAtom, navigateTo } from "../storage/atoms/uiAtoms";

const TRAIN_ITEMS: { page: PageId; label: string; icon: React.ReactNode }[] = [
  {
    page: "metrics",
    label: "Metrics",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>,
  },
  {
    page: "gradients",
    label: "Gradients",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>,
  },
  {
    page: "runs",
    label: "Runs",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg>,
  },
  {
    page: "logs",
    label: "Logs",
    icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>,
  },
];

const INSPECT_ITEMS: { page: PageId; label: string; icon: React.ReactNode }[] = [
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
];

export function Sidebar() {
  const activePage = useAtomValue(activePageAtom);
  const tab = useAtomValue(sidebarTabAtom);

  const items = tab === "train" ? TRAIN_ITEMS : INSPECT_ITEMS;

  return (
    <nav className="sidebar">
      <div className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.page}
            className={`sidebar-item ${activePage === item.page ? "active" : ""}`}
            onClick={() => navigateTo(item.page)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
