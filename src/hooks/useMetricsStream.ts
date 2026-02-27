/**
 * SpeakAI — WebSocket Hook for Real-Time Metrics
 *
 * Connects to the Python backend WebSocket and provides:
 * - Live metrics stream (eye contact, posture, etc.)
 * - Coaching feedback messages
 * - Session control (start/stop/demo)
 * - Connection state management
 * - Auto-reconnect with exponential backoff
 */

import { useState, useEffect, useCallback, useRef } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SpeakingMetrics {
  eye_contact: number;
  head_stability: number;
  posture_score: number;
  facial_engagement: number;
  attention_intensity: number;
  filler_words: number;
  words_per_minute: number;
  timestamp: number;
}

export interface CoachingFeedback {
  id: number;
  severity: "info" | "warning" | "critical";
  headline: string;
  explanation: string;
  tip?: string;
  timestamp: number;
}

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error"
  | "reconnecting";

export type SessionStatus = "idle" | "active" | "demo";

interface WebSocketMessage {
  type: string;
  /** Backend may send `data` or `payload` depending on event type */
  data?: Record<string, unknown>;
  payload?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Default metrics
// ---------------------------------------------------------------------------

const DEFAULT_METRICS: SpeakingMetrics = {
  eye_contact: 0,
  head_stability: 0,
  posture_score: 0,
  facial_engagement: 0,
  attention_intensity: 0,
  filler_words: 0,
  words_per_minute: 0,
  timestamp: 0,
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface UseMetricsStreamOptions {
  url?: string;
  autoConnect?: boolean;
  maxReconnectAttempts?: number;
}

export function useMetricsStream(options: UseMetricsStreamOptions = {}) {
  const {
    url = "ws://localhost:8080/ws/metrics",
    autoConnect = true,
    maxReconnectAttempts = 10,
  } = options;

  const [metrics, setMetrics] = useState<SpeakingMetrics>(DEFAULT_METRICS);
  const [feedback, setFeedback] = useState<CoachingFeedback | null>(null);
  const [feedbackHistory, setFeedbackHistory] = useState<CoachingFeedback[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // -- Connect ---------------------------------------------------------------

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("connected");
      reconnectAttempts.current = 0;
      console.log("✅ Metrics WebSocket connected");

      // Start ping/keepalive
      pingInterval.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 15_000);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case "metrics":
          case "metrics_update":
            setMetrics(
              (message.payload ?? message.data) as unknown as SpeakingMetrics
            );
            break;

          case "feedback":
          case "feedback_update": {
            const fb = (message.payload ??
              message.data) as unknown as CoachingFeedback;
            setFeedback(fb);
            setFeedbackHistory((prev) => [fb, ...prev].slice(0, 50));
            break;
          }

          case "session_started":
            setSessionStatus("active");
            break;

          case "session_stopped":
            setSessionStatus("idle");
            break;

          case "demo_started":
            setSessionStatus("demo");
            break;

          case "pong":
            // keepalive ack
            break;

          default:
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      setConnectionStatus("error");
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      if (pingInterval.current) clearInterval(pingInterval.current);

      // Auto-reconnect with exponential backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(
          1000 * 2 ** reconnectAttempts.current,
          30_000
        );
        setConnectionStatus("reconnecting");
        reconnectTimer.current = setTimeout(() => {
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };
  }, [url, maxReconnectAttempts]);

  // -- Disconnect ------------------------------------------------------------

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (pingInterval.current) clearInterval(pingInterval.current);
    reconnectAttempts.current = maxReconnectAttempts; // prevent reconnect
    wsRef.current?.close();
    wsRef.current = null;
    setConnectionStatus("disconnected");
  }, [maxReconnectAttempts]);

  // -- Session Controls ------------------------------------------------------

  const startSession = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "start_session" }));
    }
  }, []);

  const stopSession = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop_session" }));
      setSessionStatus("idle");
    }
  }, []);

  const startDemo = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "start_demo" }));
    }
  }, []);

  // -- Send Frame ------------------------------------------------------------

  const sendFrame = useCallback((base64Jpeg: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "frame", data: base64Jpeg })
      );
    }
  }, []);

  // -- Dismiss Feedback ------------------------------------------------------

  const dismissFeedback = useCallback(() => {
    setFeedback(null);
  }, []);

  // -- Lifecycle -------------------------------------------------------------

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    // State
    metrics,
    feedback,
    feedbackHistory,
    connectionStatus,
    sessionStatus,

    // Actions
    connect,
    disconnect,
    startSession,
    stopSession,
    startDemo,
    sendFrame,
    dismissFeedback,
  };
}
