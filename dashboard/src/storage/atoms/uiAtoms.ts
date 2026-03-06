import { atom } from "jotai";
import { persistAtom } from "../persist";

export type PageId = "train" | "runs" | "models" | "eval" | "chat" | "tables" | "leaderboard";

/** Whether the sidebar is visible. Persisted across sessions. */
export const sidebarOpenAtom = persistAtom<boolean>("sllm:sidebarOpen", true);

const VALID_PAGES = new Set<string>(["train", "runs", "models", "eval", "chat", "tables", "leaderboard"]);

export function parseHash(): { page: PageId; sub: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  const page = VALID_PAGES.has(parts[0]) ? (parts[0] as PageId) : "train";
  const sub = parts[1] || null;
  return { page, sub };
}

const initial = parseHash();

/** Current top-level page, driven by URL hash. */
export const activePageAtom = atom<PageId>(initial.page);

/** Sub-path after the page (e.g. "metrics" from #/train/metrics). Null if none. */
export const subPageAtom = atom<string | null>(initial.sub);

/** Navigate to a page by updating the URL hash. */
export function navigateTo(page: PageId, sub?: string) {
  window.location.hash = sub ? `#/${page}/${sub}` : `#/${page}`;
}
