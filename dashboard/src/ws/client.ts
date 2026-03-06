import type { Store } from "jotai";
import { db } from "../lib/db";
import type { TableSchema } from "../lib/types";
import type { ConnectionStatus } from "../storage";
import { getTableVersionAtom } from "../db/atoms";

// Use the page's host so Vite dev proxy handles it
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
export const WS_URL = `${protocol}//${window.location.host}/ws`;

export interface WSCallbacks {
  onConnectionChange: (status: ConnectionStatus) => void;
  onReady: () => void;
}

const BASE_DELAY = 2000;
const MAX_DELAY = 30000;

/**
 * CDC sync WebSocket client.
 *
 * Protocol:
 * 1. Server sends {type: "schema", tables: TableSchema[]}
 * 2. Client responds with {hashes: {tableName: hash}}
 * 3. Server sends {type: "dump", table, rows} for mismatched tables
 * 4. Server sends {type: "ready"}
 * 5. Server streams {type: "op", table, op, row?, key?}
 */
export function createWebSocket(callbacks: WSCallbacks, store: Store): () => void {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout>;
  let closed = false;
  let attempt = 0;

  function bumpTable(table: string) {
    const atom = getTableVersionAtom(table);
    store.set(atom, (v) => v + 1);
  }

  function connect() {
    if (closed) return;
    callbacks.onConnectionChange("reconnecting");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      attempt = 0;
      console.log("[ws] connected, waiting for schema...");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "schema": {
            const names = (msg.tables as TableSchema[]).map((t) => t.name);
            console.log("[ws] schema received:", names.join(", "));
            handleSchema(msg.tables as TableSchema[]);
            const hashes = db.getHashes();
            console.log("[ws] sending hashes:", hashes);
            ws?.send(JSON.stringify({ hashes }));
            break;
          }

          case "dump":
            console.log(`[ws] dump: ${msg.table} (${msg.rows.length} rows)`);
            db.applyDump(msg.table, msg.rows);
            bumpTable(msg.table);
            break;

          case "ready":
            console.log("[ws] sync complete, ready");
            db.resetDumpState();
            callbacks.onConnectionChange("connected");
            callbacks.onReady();
            break;

          case "op":
            db.applyOp(msg);
            bumpTable(msg.table);
            break;

          case "ops":
            db.applyOps(msg);
            bumpTable(msg.table);
            break;

          default:
            console.log("[ws] unknown message type:", msg.type);
        }
      } catch (err) {
        console.warn("[ws] message error:", err);
      }
    };

    ws.onclose = () => {
      if (!closed) {
        callbacks.onConnectionChange("disconnected");
        const delay = Math.min(BASE_DELAY * Math.pow(2, attempt), MAX_DELAY);
        attempt++;
        console.log(`[ws] disconnected, retrying in ${delay}ms`);
        reconnectTimer = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  connect();

  return () => {
    closed = true;
    clearTimeout(reconnectTimer);
    ws?.close();
  };
}

/** Register any new tables from the schema message. */
function handleSchema(tables: TableSchema[]) {
  for (const schema of tables) {
    if (!db.tables.has(schema.name)) {
      db.addTable(schema);
    }
  }
}
