import { atom } from "jotai";
import { persistAtom } from "../persist";

export type SidebarTab = "train" | "inspect";
export type PageId = "metrics" | "gradients" | "runs" | "logs" | "models" | "eval";

const VALID_PAGES = new Set<string>(["metrics", "gradients", "runs", "logs", "models", "eval"]);

const TRAIN_PAGES = new Set<PageId>(["metrics", "gradients", "runs", "logs"]);
const INSPECT_PAGES = new Set<PageId>(["models", "eval"]);

export function parseHash(): { page: PageId; sub: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  const page = VALID_PAGES.has(parts[0]) ? (parts[0] as PageId) : "metrics";
  const sub = parts[1] || null;
  return { page, sub };
}

const initial = parseHash();

/** Current top-level page, driven by URL hash. */
export const activePageAtom = atom<PageId>(initial.page);

/** Sub-path after the page (e.g. "new" from #/runs/new). Null if none. */
export const subPageAtom = atom<string | null>(initial.sub);

/** Which sidebar tab is active (persisted). */
export const sidebarTabAtom = persistAtom<SidebarTab>(
  "sllm:sidebarTab",
  TRAIN_PAGES.has(initial.page) ? "train" : "inspect",
);

/** Navigate to a page by updating the URL hash. */
export function navigateTo(page: PageId, sub?: string) {
  window.location.hash = sub ? `#/${page}/${sub}` : `#/${page}`;
}

/** Get the tab a page belongs to. */
export function tabForPage(page: PageId): SidebarTab {
  return INSPECT_PAGES.has(page) ? "inspect" : "train";
}
