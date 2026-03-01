import { atom } from "jotai";

/**
 * Creates a Jotai atom that persists its value to localStorage.
 * Reads from localStorage on initialization, writes on every set.
 */
export function persistAtom<T>(key: string, initialValue: T) {
  let init = initialValue;
  try {
    const stored = localStorage.getItem(key);
    if (stored !== null) init = JSON.parse(stored) as T;
  } catch {
    // Corrupted localStorage entry — fall back to default
  }

  const baseAtom = atom(init);

  const persistedAtom = atom(
    (get) => get(baseAtom),
    (_get, set, value: T) => {
      set(baseAtom, value);
      localStorage.setItem(key, JSON.stringify(value));
    },
  );

  return persistedAtom;
}
