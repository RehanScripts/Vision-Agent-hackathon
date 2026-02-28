/**
 * SpeakAI — WebSocket Hook for Real-Time Metrics & Communication
 *
 * Connects to the Python backend WebSocket and provides:
 * - Live metrics stream (eye contact, posture, etc.)
 * - Coaching feedback messages
 * - Chat messages (bidirectional AI ↔ user conversation)
 * - Live transcript (speech-to-text from audio)
 * - Conversation state (who's speaking, turn count)
 * - Session control (start/stop/demo)
 * - Connection state management
 * - Auto-reconnect with exponential backoff
 *
 * The AI agent joins a Stream Video call as a real participant.
 * No base64 frame streaming — the agent receives media via Stream's edge.
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

/** Chat message in the bidirectional conversation */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  source?: string;
}

/** Speech-to-text transcript entry */
export interface TranscriptEntry {
  speaker: string;
  text: string;
  timestamp: number;
  confidence: number;
  is_final: boolean;
}

/** Conversation state */
export interface ConversationState {
  is_user_speaking: boolean;
  is_agent_speaking: boolean;
  turn_count: number;
  last_user_speech: string;
  last_agent_response: string;
}

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error"
  | "reconnecting";

export type SessionStatus = "idle" | "starting" | "active" | "demo";

/** Debug telemetry sent by the backend every ~5 s */
export interface SystemStatus {
  sdk_active: boolean;
  multimodal_active: boolean;
  agent_joined: boolean;
  audio_active: boolean;
  inference_latency_ms: number;
  frames_processed: number;
  metrics_source: string;
  transcript_entries: number;
  chat_messages: number;
  conversation_turns: number;
  source: string;
  /** Explicit session state: init | ready | active | degraded | failed */
  session_state?: SessionStateValue;
  /** Explicit operating mode: multimodal | audio_only | unavailable */
  session_mode?: SessionModeValue;
  /** Health check snapshot */
  health?: { video: boolean; audio: boolean; model: boolean };
  /** Latency trace */
  latency?: Record<string, unknown>;
  /** Pipeline diagnostics (video frame extraction health) */
  pipeline?: PipelineDiagnostics;
}

/** Video pipeline diagnostics from the processor */
export interface PipelineDiagnostics {
  frames_processed: number;
  last_latency_ms: number;
  direct_reader_active: boolean;
  forwarder_active: boolean;
  has_raw_track: boolean;
}

/** Session state values from the backend state machine */
export type SessionStateValue =
  | "init"
  | "ready"
  | "active"
  | "degraded"
  | "failed";

/** Operating mode values from the backend policy layer */
export type SessionModeValue = "multimodal" | "audio_only" | "unavailable";

interface WebSocketMessage {
  type: string;
  data?: Record<string, unknown>;
  payload?: Record<string, unknown>;
  message?: string;
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
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  // Communication state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [conversationState, setConversationState] =
    useState<ConversationState | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastMetricsTsRef = useRef<number>(0);

  // -- Connect ---------------------------------------------------------------

  const connect = useCallback(() => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    setConnectionStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("connected");
      setLastError(null);
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
            {
              const nextMetrics =
                (message.payload ?? message.data) as unknown as SpeakingMetrics;
              const nextTs = typeof nextMetrics.timestamp === "number" ? nextMetrics.timestamp : 0;
              if (nextTs > 0 && nextTs === lastMetricsTsRef.current) {
                break;
              }
              if (nextTs > 0) {
                lastMetricsTsRef.current = nextTs;
              }
              setMetrics(nextMetrics);
            }
            break;

          case "feedback":
          case "feedback_update": {
            const fb = (message.payload ??
              message.data) as unknown as CoachingFeedback;
            setFeedback(fb);
            setFeedbackHistory((prev) => [fb, ...prev].slice(0, 50));
            break;
          }

          // ── Communication events ──────────────────────────────
          case "chat": {
            const chatMsg = (message.payload ??
              message.data) as unknown as ChatMessage;
            setChatMessages((prev) => [...prev, chatMsg].slice(-100));
            break;
          }

          case "transcript": {
            const entry = (message.payload ??
              message.data) as unknown as TranscriptEntry;
            if (entry.is_final) {
              setTranscript((prev) => [...prev, entry].slice(-200));
            }
            break;
          }

          case "conversation_state": {
            const state = (message.payload ??
              message.data) as unknown as ConversationState;
            setConversationState(state);
            break;
          }

          // ── Session lifecycle ─────────────────────────────────
          case "session_started":
            setSessionStatus("active");
            setLastError(null);
            setChatMessages([]);
            setTranscript([]);
            setConversationState(null);
            break;

          case "session_starting":
            setSessionStatus("starting");
            setLastError(null);
            break;

          case "session_stopped":
            setSessionStatus("idle");
            break;

          case "demo_started":
            setSessionStatus("demo");
            setLastError(null);
            setChatMessages([]);
            setTranscript([]);
            break;

          case "pong":
            // keepalive ack
            break;

          case "system_status":
            setSystemStatus(
              (message.payload ?? message.data) as unknown as SystemStatus
            );
            break;

          case "state_transition": {
            // Update system status with fresh state/mode from transition event
            const transitionData = (message.payload ?? message.data) as Record<string, unknown>;
            setSystemStatus((prev) => ({
              ...(prev ?? {
                sdk_active: false,
                multimodal_active: false,
                agent_joined: false,
                audio_active: false,
                inference_latency_ms: 0,
                frames_processed: 0,
                metrics_source: "none",
                transcript_entries: 0,
                chat_messages: 0,
                conversation_turns: 0,
                source: "sdk",
              }),
              session_state: transitionData.session_state as SessionStateValue,
              session_mode: transitionData.session_mode as SessionModeValue,
              health: transitionData.health as { video: boolean; audio: boolean; model: boolean },
              pipeline: transitionData.pipeline as PipelineDiagnostics | undefined,
            }));
            break;
          }

          case "error":
            setLastError(message.message ?? "Unknown server error");
            console.warn("⚠️ Server error:", message.message ?? message.data);
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

  const startSession = useCallback((callId?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "start_session",
          ...(callId ? { call_id: callId } : {}),
        })
      );
    }
  }, []);

  const stopSession = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop_session" }));
      setSessionStatus("idle");
      lastMetricsTsRef.current = 0;
    }
  }, []);

  const startDemo = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "start_demo" }));
    }
  }, []);

  // -- Chat: send a text message to the AI agent ----------------------------

  const sendMessage = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && text.trim()) {
      wsRef.current.send(
        JSON.stringify({ type: "send_message", text: text.trim() })
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
    systemStatus,
    lastError,

    // Communication state
    chatMessages,
    transcript,
    conversationState,

    // Actions
    connect,
    disconnect,
    startSession,
    stopSession,
    startDemo,
    sendMessage,
    dismissFeedback,
  };
}
