import { useAtomValue, useSetAtom } from "jotai";
import { logsAtom, clearLogsAtom } from "../storage";
import { LogPage } from "../components/LogPage";

export function LogContainer() {
  const logs = useAtomValue(logsAtom);
  const clearLogs = useSetAtom(clearLogsAtom);

  return <LogPage logs={logs} onClear={clearLogs} />;
}
