import { useEffect } from "react";
import { useStore } from "jotai";
import { activePageAtom, subPageAtom } from "../storage";
import { parseHash } from "../storage/atoms/uiAtoms";

/**
 * Syncs URL hash → activePageAtom + subPageAtom.
 * Call once in App.tsx.
 */
export function useHashRouter() {
  const store = useStore();

  useEffect(() => {
    function onHashChange() {
      const { page, sub } = parseHash();
      store.set(activePageAtom, page);
      store.set(subPageAtom, sub);
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [store]);
}
