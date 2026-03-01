import type { StepMetrics, LayerStat, ActivationStat, Generation, TrainingStatus, LogLevel } from "../types/metrics";
import type { ConnectionStatus } from "../storage";

export const WS_URL = `ws://${window.location.hostname}:8000/ws`;

export interface WSHandlers {
  onStep: (data: StepMetrics) => void;
  onLayers: (data: LayerStat[]) => void;
  onActivations: (data: ActivationStat[]) => void;
  onGeneration: (data: Generation) => void;
  onStatus: (data: TrainingStatus) => void;
  onLog: (data: { level: LogLevel; message: string }) => void;
  onConnectionChange: (status: ConnectionStatus) => void;
}

/**
 * Creates a WebSocket connection with auto-reconnect.
 * Returns a cleanup function to close the connection.
 */
export function createWebSocket(handlers: WSHandlers): () => void {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout>;
  let closed = false;

  function connect() {
    if (closed) return;
    handlers.onConnectionChange("reconnecting");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      handlers.onConnectionChange("connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "step":
            handlers.onStep(msg.data as StepMetrics);
            break;
          case "layers":
            handlers.onLayers(msg.data as LayerStat[]);
            break;
          case "activations":
            handlers.onActivations(msg.data as ActivationStat[]);
            break;
          case "generation":
            handlers.onGeneration(msg.data as Generation);
            break;
          case "status":
            handlers.onStatus(msg.data);
            break;
          case "log":
            handlers.onLog(msg.data);
            break;
        }
      } catch (err) {
        console.warn("Failed to parse WS message:", err);
      }
    };

    ws.onclose = () => {
      if (!closed) {
        handlers.onConnectionChange("disconnected");
        reconnectTimer = setTimeout(connect, 2000);
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
