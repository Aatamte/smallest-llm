import type { ReactNode } from "react";
import { useAtomValue } from "jotai";
import { sidebarOpenAtom } from "../storage/atoms/uiAtoms";
import { Sidebar } from "./Sidebar";
import { HeaderContainer } from "../containers/HeaderContainer";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const sidebarOpen = useAtomValue(sidebarOpenAtom);

  return (
    <div className="app-layout">
      <HeaderContainer />
      <div className="app-body">
        {sidebarOpen && <Sidebar />}
        <div className="app-content">
          {children}
        </div>
      </div>
    </div>
  );
}
