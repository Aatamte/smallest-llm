import { useCallback, useMemo } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getLogs } from "../db/queries";
import { LogPage } from "../components/LogPage";
import type { LogEntry } from "../types/metrics";

export function LogContainer() {
  const runId = useAtomValue(activeRunIdAtom);
  const dbLogs = useQuery(useCallback(() => getLogs(runId), [runId]));

  const logs = useMemo((): LogEntry[] =>
    dbLogs.map((row) => ({
      timestamp: row.created_at ? new Date(row.created_at).getTime() : 0,
      level: row.level as LogEntry["level"],
      source: "train",
      message: row.message,
    })).reverse(),  // DB returns DESC, LogPage expects chronological
  [dbLogs]);

  return <LogPage logs={logs} onClear={() => {}} />;
}
