import { useEffect, useRef } from "react";
import { useStore } from "jotai";
import { connectionStatusAtom } from "../storage";
import { db } from "../lib/db";
import { createWebSocket } from "../ws/client";

/**
 * Initializes the sql.js database and connects via WebSocket.
 * The sync protocol (schema → hashes → dumps → ready → CDC ops)
 * populates the sql.js tables automatically.
 */
export function useWebSocket() {
  const store = useStore();
  const dbReady = useRef(false);

  useEffect(() => {
    let cleanup: (() => void) | undefined;

    async function start() {
      // Init sql.js WASM once
      if (!dbReady.current) {
        console.log("[db] initializing sql.js...");
        await db.init();
        dbReady.current = true;
        console.log("[db] sql.js ready");
      }

      cleanup = createWebSocket({
        onConnectionChange: (status) => store.set(connectionStatusAtom, status),
        onReady: () => {
          // sync complete — tables are populated
        },
      }, store);
    }

    start().catch((err) => console.error("Failed to init db/ws:", err));

    return () => {
      cleanup?.();
    };
  }, [store]);
}
