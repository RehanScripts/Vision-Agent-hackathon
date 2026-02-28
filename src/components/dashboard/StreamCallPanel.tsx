"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  StreamVideo,
  StreamCall,
  StreamTheme,
  StreamVideoClient,
  SpeakerLayout,
  CallControls,
} from "@stream-io/video-react-sdk";
import { cn } from "@/lib/utils";
import {
  Video,
  VideoOff,
  Phone,
  PhoneOff,
  Wifi,
  WifiOff,
  Users,
  Loader2,
  AlertTriangle,
  Camera,
  Mic,
  Activity,
} from "lucide-react";

interface StreamCallPanelProps {
  className?: string;
  connectionStatus?: "connected" | "connecting" | "disconnected" | "error" | "reconnecting";
  sessionStatus?: "idle" | "active" | "completed" | "starting" | "demo";
  streaming?: boolean;
  onReadyChange?: (ready: boolean) => void;
  onMediaReadyChange?: (ready: boolean) => void;
  onCallIdChange?: (callId: string) => void;
  /** Pipeline diagnostics from backend system_status */
  pipelineStatus?: {
    frames_processed?: number;
    direct_reader_active?: boolean;
    forwarder_active?: boolean;
    last_latency_ms?: number;
  };
  /** Health check snapshot */
  healthStatus?: {
    video?: boolean;
    audio?: boolean;
    model?: boolean;
  };
  /** Session mode from backend */
  sessionMode?: string;
}

type CallStatus = "idle" | "joining" | "joined" | "error";

export default function StreamCallPanel({
  className,
  connectionStatus = "disconnected",
  sessionStatus = "idle",
  streaming: _streaming = false,
  onReadyChange,
  onMediaReadyChange,
  onCallIdChange,
  pipelineStatus,
  healthStatus,
  sessionMode,
}: StreamCallPanelProps) {
  void _streaming;
  const [status, setStatus] = useState<CallStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [callId, setCallId] = useState("speakai-live");
  const [client, setClient] = useState<StreamVideoClient | null>(null);
  const [call, setCall] = useState<ReturnType<StreamVideoClient["call"]> | null>(null);
  const [userId, setUserId] = useState("user-loading");
  const [showControls, setShowControls] = useState(false);
  const [joinLatencyMs, setJoinLatencyMs] = useState<number | null>(null);
  const clientRef = useRef<StreamVideoClient | null>(null);
  const callRef = useRef<ReturnType<StreamVideoClient["call"]> | null>(null);
  const controlsTimeout = useRef<NodeJS.Timeout | null>(null);
  const joinStartRef = useRef<number>(0);

  useEffect(() => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      setUserId(`user-${crypto.randomUUID().slice(0, 8)}`);
      return;
    }
    setUserId(`user-${Math.random().toString(36).slice(2, 10)}`);
  }, []);

  const apiKey = process.env.NEXT_PUBLIC_STREAM_API_KEY ?? "";
  const isConnected = connectionStatus === "connected";
  const isAnalyzing = status === "joined" && isConnected && sessionStatus !== "idle";

  const handleMouseMove = useCallback(() => {
    if (status !== "joined") return;
    setShowControls(true);
    if (controlsTimeout.current) clearTimeout(controlsTimeout.current);
    controlsTimeout.current = setTimeout(() => setShowControls(false), 3000);
  }, [status]);

  const leaveCall = useCallback(async () => {
    try {
      await callRef.current?.leave();
    } catch {
      // ignore leave errors
    }
    try {
      await clientRef.current?.disconnectUser?.();
    } catch {
      // ignore disconnect errors
    }
    callRef.current = null;
    clientRef.current = null;
    setCall(null);
    setClient(null);
    setStatus("idle");
    setError(null);
    setJoinLatencyMs(null);
    onReadyChange?.(false);
    onMediaReadyChange?.(false);
  }, [onMediaReadyChange, onReadyChange]);

  const joinCall = useCallback(async () => {
    if (status === "joining" || status === "joined") return;

    if (!apiKey) {
      setStatus("error");
      setError("Missing NEXT_PUBLIC_STREAM_API_KEY in .env");
      return;
    }

    setStatus("joining");
    setError(null);
    joinStartRef.current = performance.now(); // ⚡ Start latency timer

    if (callRef.current || clientRef.current) {
      await leaveCall();
    }

    try {
      const response = await fetch(
        `http://localhost:8080/token?user_id=${encodeURIComponent(userId)}`
      );
      const data = (await response.json()) as {
        token?: string;
        api_key?: string;
        error?: string;
      };

      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to fetch Stream token");
      }

      const token = data.token;
      const resolvedKey = data.api_key || apiKey;

      if (!token || !resolvedKey) {
        throw new Error("Missing Stream token or API key");
      }

      const videoClient = new StreamVideoClient({
        apiKey: resolvedKey,
        user: { id: userId, name: userId },
        token,
      });

      const videoCall = videoClient.call("default", callId);
      await videoCall.join({ create: true });

      // ⚡ Measure join latency
      const elapsed = Math.round(performance.now() - joinStartRef.current);
      setJoinLatencyMs(elapsed);
      console.log(`📞 Joined call in ${elapsed}ms`);

      // Enable camera + mic in parallel (non-blocking)
      Promise.all([
        videoCall.camera.enable().catch(() => null),
        videoCall.microphone.enable().catch(() => null),
      ]).then(([cam, mic]) => {
        onMediaReadyChange?.(cam !== null || mic !== null);
        if (cam === null && mic === null) {
          setError("Camera/Mic failed. Allow permissions and retry.");
        }
      });

      clientRef.current = videoClient;
      callRef.current = videoCall;
      setClient(videoClient);
      setCall(videoCall);
      setStatus("joined");
      onReadyChange?.(true);
      onCallIdChange?.(callId);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to join call";
      setStatus("error");
      setError(message);
      setJoinLatencyMs(null);
      onReadyChange?.(false);
      onMediaReadyChange?.(false);
    }
  }, [apiKey, callId, leaveCall, onCallIdChange, onMediaReadyChange, onReadyChange, status, userId]);

  useEffect(() => {
    return () => {
      void leaveCall();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* ─── Video Area ─── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="relative w-full aspect-video rounded-2xl overflow-hidden border border-white/[0.08] bg-black/40"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setShowControls(false)}
      >
        {/* Neon border glow */}
        <div className="absolute inset-0 rounded-2xl shadow-[inset_0_0_30px_rgba(79,140,255,0.08)] pointer-events-none z-10" />

        {/* ── Overlaid Status Badges ── */}
        <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
          {/* Live / Status Badge */}
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium backdrop-blur-md border",
              status === "joined" && isAnalyzing
                ? "bg-[#22C55E]/20 border-[#22C55E]/30 text-[#22C55E]"
                : status === "joined"
                  ? "bg-[#4F8CFF]/20 border-[#4F8CFF]/30 text-[#4F8CFF]"
                  : status === "joining"
                    ? "bg-[#F59E0B]/20 border-[#F59E0B]/30 text-[#F59E0B]"
                    : "bg-white/10 border-white/10 text-white/40"
            )}
          >
            {status === "joined" && isAnalyzing ? (
              <>
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22C55E] opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#22C55E]" />
                </span>
                ANALYZING
              </>
            ) : status === "joined" ? (
              <>
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4F8CFF] opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#4F8CFF]" />
                </span>
                LIVE
              </>
            ) : status === "joining" ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                CONNECTING
              </>
            ) : (
              "OFFLINE"
            )}
          </motion.div>
        </div>

        {/* Top-right badges */}
        <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
          <div
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border",
              isConnected
                ? "bg-[#22C55E]/15 border-[#22C55E]/20 text-[#22C55E]/80"
                : "bg-white/10 border-white/10 text-white/30"
            )}
          >
            {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            WS
          </div>
          {status === "joined" && (
            <div className="flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border bg-[#8B5CF6]/15 border-[#8B5CF6]/20 text-[#8B5CF6]/80">
              <Users className="w-3 h-3" />
              In Call
            </div>
          )}
          {/* Join latency badge */}
          {joinLatencyMs !== null && (
            <div
              className={cn(
                "flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border",
                joinLatencyMs < 1000
                  ? "bg-[#22C55E]/15 border-[#22C55E]/20 text-[#22C55E]/80"
                  : joinLatencyMs < 2500
                    ? "bg-[#F59E0B]/15 border-[#F59E0B]/20 text-[#F59E0B]/80"
                    : "bg-[#EF4444]/15 border-[#EF4444]/20 text-[#EF4444]/80"
              )}
            >
              <Activity className="w-3 h-3" />
              {joinLatencyMs}ms join
            </div>
          )}
          {/* Pipeline health indicators */}
          {status === "joined" && healthStatus && (
            <>
              <div
                className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border",
                  healthStatus.video
                    ? "bg-[#22C55E]/15 border-[#22C55E]/20 text-[#22C55E]/80"
                    : "bg-[#F59E0B]/15 border-[#F59E0B]/20 text-[#F59E0B]/80"
                )}
              >
                <Camera className="w-3 h-3" />
                {healthStatus.video ? "Video" : "No Video"}
              </div>
              <div
                className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border",
                  healthStatus.audio
                    ? "bg-[#22C55E]/15 border-[#22C55E]/20 text-[#22C55E]/80"
                    : "bg-[#EF4444]/15 border-[#EF4444]/20 text-[#EF4444]/80"
                )}
              >
                <Mic className="w-3 h-3" />
                {healthStatus.audio ? "Audio" : "No Audio"}
              </div>
            </>
          )}
          {/* Frame processing indicator */}
          {status === "joined" && pipelineStatus && (pipelineStatus.frames_processed ?? 0) > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 rounded-full text-[10px] backdrop-blur-md border bg-[#22C55E]/10 border-[#22C55E]/20 text-[#22C55E]/70">
              <Activity className="w-3 h-3" />
              {pipelineStatus.frames_processed} frames
              {pipelineStatus.direct_reader_active && " (direct)"}
            </div>
          )}
        </div>

        {/* ── Stream Video Content ── */}
        {client && call ? (
          <StreamVideo client={client}>
            <StreamCall call={call}>
              <StreamTheme className="speakai-stream-theme">
                <div className="absolute inset-0">
                  <SpeakerLayout participantsBarPosition="bottom" />
                </div>
                {/* Overlaid Call Controls - shown on hover */}
                <AnimatePresence>
                  {showControls && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 20 }}
                      transition={{ duration: 0.2 }}
                      className="absolute bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-black/80 via-black/40 to-transparent pt-12 pb-4 px-4"
                    >
                      <CallControls />
                    </motion.div>
                  )}
                </AnimatePresence>
              </StreamTheme>
            </StreamCall>
          </StreamVideo>
        ) : (
          /* ── Idle / Joining / Error States ── */
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
            <AnimatePresence mode="wait">
              {status === "joining" ? (
                <motion.div
                  key="joining"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div className="relative">
                    <div className="w-16 h-16 rounded-2xl bg-[#4F8CFF]/10 border border-[#4F8CFF]/20 flex items-center justify-center">
                      <Loader2 className="w-7 h-7 text-[#4F8CFF] animate-spin" />
                    </div>
                  </div>
                  <p className="text-sm text-white/50">Connecting to call...</p>
                </motion.div>
              ) : status === "error" ? (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div className="w-16 h-16 rounded-2xl bg-[#EF4444]/10 border border-[#EF4444]/20 flex items-center justify-center">
                    <AlertTriangle className="w-7 h-7 text-[#EF4444]/70" />
                  </div>
                  <p className="text-sm text-[#EF4444]/70 text-center max-w-[240px]">
                    {error || "Connection failed"}
                  </p>
                  <button
                    onClick={joinCall}
                    className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/20 hover:bg-[#EF4444]/25 transition-colors"
                  >
                    Retry
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col items-center gap-3"
                >
                  <motion.div
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    className="w-16 h-16 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center"
                  >
                    <Video className="w-7 h-7 text-white/20" />
                  </motion.div>
                  <div className="text-center">
                    <p className="text-sm text-white/40">Ready to connect</p>
                    <p className="text-[11px] text-white/20 mt-1">
                      Join a call to start your video session
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

      </motion.div>

      {/* ─── Controls Bar ─── */}
      <div className="flex items-center gap-3 px-1">
        {/* Call ID input */}
        <div className="flex-1 flex items-center gap-2">
          <input
            value={callId}
            onChange={(e) => setCallId(e.target.value)}
            disabled={status === "joined"}
            placeholder="Call ID"
            className={cn(
              "flex-1 rounded-lg bg-white/[0.04] border border-white/10 text-xs px-3 py-2 text-white/70",
              "focus:outline-none focus:ring-2 focus:ring-[#4F8CFF]/40 transition-colors",
              "placeholder:text-white/20",
              status === "joined" && "opacity-50 cursor-not-allowed"
            )}
          />
          <span
            className="text-[10px] text-white/30 font-mono truncate max-w-[100px]"
            suppressHydrationWarning
          >
            {userId}
          </span>
        </div>

        {/* Join / Leave button */}
        {status !== "joined" ? (
          <button
            onClick={joinCall}
            disabled={status === "joining" || userId === "user-loading"}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all",
              "bg-[#4F8CFF]/15 text-[#4F8CFF] border border-[#4F8CFF]/20",
              "hover:bg-[#4F8CFF]/25 hover:shadow-[0_0_20px_rgba(79,140,255,0.15)]",
              "disabled:opacity-40 disabled:cursor-not-allowed"
            )}
          >
            <Phone className="w-3.5 h-3.5" />
            {status === "joining" ? "Joining..." : "Join Call"}
          </button>
        ) : (
          <button
            onClick={leaveCall}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all",
              "bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/20",
              "hover:bg-[#EF4444]/25 hover:shadow-[0_0_20px_rgba(239,68,68,0.15)]"
            )}
          >
            <PhoneOff className="w-3.5 h-3.5" />
            Leave
          </button>
        )}
      </div>

      {/* Error message */}
      <AnimatePresence>
        {error && status !== "error" && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="text-xs text-[#EF4444]/80 px-1"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}
