import { useSyncExternalStore } from "react";
import { db } from "../lib/db";

/**
 * Subscribe to all table mutations and re-run queryFn when data changes.
 * Returns the result of queryFn, re-evaluated on every table mutation.
 */
export function useQuery<T>(queryFn: () => T): T {
  useSyncExternalStore(
    (cb) => db.subscribe(cb),
    () => db.getVersion(),
  );
  return queryFn();
}
