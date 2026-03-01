import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { HeaderContainer } from "../containers/HeaderContainer";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="app-layout">
      <HeaderContainer />
      <div className="app-body">
        <Sidebar />
        <div className="app-content">
          {children}
        </div>
      </div>
    </div>
  );
}
