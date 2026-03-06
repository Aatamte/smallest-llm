import { useAtomValue, useAtom } from "jotai";
import { connectionStatusAtom } from "../storage";
import { sidebarOpenAtom } from "../storage/atoms/uiAtoms";
import { Header } from "../components/Header";

export function HeaderContainer() {
  const connectionStatus = useAtomValue(connectionStatusAtom);
  const [sidebarOpen, setSidebarOpen] = useAtom(sidebarOpenAtom);

  return (
    <Header
      connectionStatus={connectionStatus}
      sidebarOpen={sidebarOpen}
      onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
    />
  );
}
