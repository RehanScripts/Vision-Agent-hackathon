"use client";

import { motion } from "framer-motion";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import StreamCallPanel from "@/components/dashboard/StreamCallPanel";
import ChatPanel from "@/components/dashboard/ChatPanel";
import MetricCard from "@/components/ui/MetricCard";
import GlassCard from "@/components/ui/GlassCard";
import AnimatedNumber from "@/components/ui/AnimatedNumber";
import { useMetricsStream } from "@/hooks/useMetricsStream";
import { addSession, type StoredSession, type SessionMetricsSnapshot } from "@/lib/sessionStore";
import {
  Gauge,
  Eye,
  MessageCircleWarning,
  PersonStanding,
  Play,
  Square,
  Radio,
  Wifi,
  WifiOff,
} from "lucide-react";

// Semi-circle gauge component
function SemiCircleGauge({
  value,
  max,
  color,
}: {
  value: number;
  max: number;
  color: string;
}) {
  const percentage = Math.min(value / max, 1);

  return (
    <div className="relative w-full flex justify-center">
      <svg
        viewBox="0 0 120 70"
        className="w-full max-w-[160px]"
      >
        {/* Background arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <motion.path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: percentage }}
          transition={{ duration: 1.5, ease: [0.25, 0.1, 0.25, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
        />
        {/* Needle */}
        <motion.line
          x1="60"
          y1="65"
          x2="60"
          y2="25"
          stroke="white"
          strokeWidth="1.5"
          strokeLinecap="round"
          style={{ transformOrigin: "60px 65px" }}
          initial={{ rotate: -90 }}
          animate={{ rotate: -90 + 180 * percentage }}
          transition={{ duration: 1.5, ease: [0.25, 0.1, 0.25, 1] }}
          opacity={0.6}
        />
        <circle cx="60" cy="65" r="3" fill="white" opacity="0.4" />
      </svg>
    </div>
  );
}

// Circular progress component
function CircularProgress({
  value,
  color,
  size = 80,
}: {
  value: number;
  color: string;
  size?: number;
}) {
  const radius = (size - 10) / 2;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="5"
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference * (1 - value / 100) }}
          transition={{ duration: 1.5, ease: [0.25, 0.1, 0.25, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <AnimatedNumber
          value={value}
          suffix="%"
          className="text-sm font-bold text-white/80"
        />
      </div>
    </div>
  );
}

// Horizontal progress bar
function ProgressBar({
  value,
  color,
}: {
  value: number;
  color: string;
}) {
  return (
    <div className="w-full h-2 bg-white/[0.06] rounded-full overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 12px ${color}30` }}
        initial={{ width: "0%" }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 1.5, ease: [0.25, 0.1, 0.25, 1] }}
      />
    </div>
  );
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function DashboardPageWrapper() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-64 text-white/30 text-sm">Loading dashboard…</div>}>
      <DashboardPage />
    </Suspense>
  );
}

function DashboardPage() {
  const searchParams = useSearchParams();
  const scenarioType = searchParams.get("type") || "Free Practice";

  const [streamCallId, setStreamCallId] = useState("speakai-live");
  const [streamReady, setStreamReady] = useState(false);
  const [streamMediaReady, setStreamMediaReady] = useState(false);

  const {
    metrics,
    connectionStatus,
    sessionStatus,
    systemStatus,
    lastError,
    connect,
    startSession,
    stopSession,
    startDemo,
    chatMessages,
    transcript,
    conversationState,
    sendMessage,
  } = useMetricsStream({ url: "ws://localhost:8080/ws/metrics", autoConnect: false });

  // Auto-connect on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // -- Session tracking for persistence --
  const sessionStartRef = useRef<number | null>(null);
  const timelineRef = useRef<Array<{ time: number; metrics: SessionMetricsSnapshot }>>([]);
  const lastSnapshotRef = useRef<number>(0);
  const [sessionLabel, setSessionLabel] = useState(scenarioType);

  // Update label when query param changes
  useEffect(() => {
    setSessionLabel(searchParams.get("type") || "Free Practice");
  }, [searchParams]);

  // Accumulate timeline snapshots every 30s during active session
  useEffect(() => {
    if (sessionStatus === "idle") return;

    const snap = (): SessionMetricsSnapshot => ({
      eye_contact: metrics.eye_contact,
      head_stability: metrics.head_stability,
      posture_score: metrics.posture_score,
      facial_engagement: metrics.facial_engagement,
      attention_intensity: metrics.attention_intensity,
      filler_words: metrics.filler_words,
      words_per_minute: metrics.words_per_minute,
    });

    const now = Date.now();
    if (sessionStartRef.current === null) {
      sessionStartRef.current = now;
      timelineRef.current = [{ time: 0, metrics: snap() }];
      lastSnapshotRef.current = now;
    } else if (now - lastSnapshotRef.current >= 30_000) {
      timelineRef.current.push({
        time: Math.round((now - sessionStartRef.current) / 1000),
        metrics: snap(),
      });
      lastSnapshotRef.current = now;
    }
  }, [metrics, sessionStatus]);

  // Save session when it stops
  const handleStop = useCallback(() => {
    // Capture final metrics before stopping
    const durationSeconds = sessionStartRef.current
      ? Math.round((Date.now() - sessionStartRef.current) / 1000)
      : 0;

    const finalSnap: SessionMetricsSnapshot = {
      eye_contact: metrics.eye_contact,
      head_stability: metrics.head_stability,
      posture_score: metrics.posture_score,
      facial_engagement: metrics.facial_engagement,
      attention_intensity: metrics.attention_intensity,
      filler_words: metrics.filler_words,
      words_per_minute: metrics.words_per_minute,
    };

    const compositeScore = Math.round(
      (metrics.eye_contact +
        metrics.posture_score +
        metrics.head_stability +
        metrics.facial_engagement) /
        4
    );

    if (durationSeconds > 3) {
      const stored: StoredSession = {
        id: crypto.randomUUID(),
        title: `${sessionLabel} Session`,
        type: sessionLabel,
        date: new Date().toISOString(),
        durationSeconds,
        score: compositeScore,
        metrics: finalSnap,
        timeline: timelineRef.current,
      };
      addSession(stored);
    }

    // Reset tracking
    sessionStartRef.current = null;
    timelineRef.current = [];
    lastSnapshotRef.current = 0;

    stopSession();
  }, [metrics, sessionLabel, stopSession]);



  // Auto-start demo if query param says so
  useEffect(() => {
    if (searchParams.get("autostart") === "demo" && connectionStatus === "connected" && sessionStatus === "idle") {
      startDemo();
    }
    if (
      searchParams.get("autostart") === "live" &&
      connectionStatus === "connected" &&
      sessionStatus === "idle" &&
      streamReady
    ) {
      startSession(streamCallId);
    }
    // Only run once when connected
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectionStatus, streamReady, streamCallId]);

  const isSessionActive = sessionStatus !== "idle";

  // Derive trend helpers (simple: compare to baseline or show neutral)
  const eyeTrend = metrics.eye_contact >= 70 ? ("up" as const) : ("down" as const);
  const postureTrend = metrics.posture_score >= 80 ? ("up" as const) : ("down" as const);
  const fillerTrend = metrics.filler_words <= 3 ? ("down" as const) : ("up" as const);
  const wpmTrend = metrics.words_per_minute >= 120 && metrics.words_per_minute <= 160 ? ("up" as const) : ("down" as const);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Session Control Bar */}
      <motion.div variants={fadeUp} className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {connectionStatus === "connected" ? (
              <Wifi className="w-4 h-4 text-[#22C55E]" />
            ) : connectionStatus === "reconnecting" ? (
              <WifiOff className="w-4 h-4 text-[#F59E0B] animate-pulse" />
            ) : (
              <WifiOff className="w-4 h-4 text-white/30" />
            )}
            <span className="text-xs text-white/40 capitalize">{connectionStatus}</span>
          </div>
          {isSessionActive && (
            <div className="flex items-center gap-1.5">
              <Radio className="w-3 h-3 text-[#EF4444] animate-pulse" />
              <span className="text-xs text-white/50 uppercase tracking-wider font-medium">
                {sessionStatus === "demo" ? "Demo Mode" : "Live Session"}
              </span>
            </div>
          )}
          {/* Explicit mode indicator (requirement #5) */}
          {isSessionActive && sessionStatus !== "demo" && systemStatus?.session_mode && (
            <span
              className={`text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full border ${
                systemStatus.session_mode === "multimodal"
                  ? "text-[#22C55E] border-[#22C55E]/30 bg-[#22C55E]/10"
                  : systemStatus.session_mode === "audio_only"
                  ? "text-[#F59E0B] border-[#F59E0B]/30 bg-[#F59E0B]/10"
                  : "text-white/30 border-white/10 bg-white/5"
              }`}
            >
              {systemStatus.session_mode === "multimodal"
                ? "Multimodal"
                : systemStatus.session_mode === "audio_only"
                ? "Audio Only"
                : "Unavailable"}
            </span>
          )}
          {/* Session state badge */}
          {isSessionActive && sessionStatus !== "demo" && systemStatus?.session_state && systemStatus.session_state !== "active" && (
            <span
              className={`text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full border ${
                systemStatus.session_state === "degraded"
                  ? "text-[#F59E0B] border-[#F59E0B]/30 bg-[#F59E0B]/10"
                  : systemStatus.session_state === "failed"
                  ? "text-[#EF4444] border-[#EF4444]/30 bg-[#EF4444]/10"
                  : systemStatus.session_state === "ready"
                  ? "text-[#4F8CFF] border-[#4F8CFF]/30 bg-[#4F8CFF]/10"
                  : "text-white/30 border-white/10 bg-white/5"
              }`}
            >
              {systemStatus.session_state}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {!isSessionActive ? (
            <>
              <button
                onClick={() => startSession(streamCallId)}
                disabled={connectionStatus !== "connected" || !streamReady || !streamMediaReady}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-[#4F8CFF]/15 text-[#4F8CFF] border border-[#4F8CFF]/20 hover:bg-[#4F8CFF]/25 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Play className="w-3.5 h-3.5" /> Start Session
              </button>
              <button
                onClick={startDemo}
                disabled={connectionStatus !== "connected"}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-[#8B5CF6]/15 text-[#8B5CF6] border border-[#8B5CF6]/20 hover:bg-[#8B5CF6]/25 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Radio className="w-3.5 h-3.5" /> Demo Mode
              </button>
            </>
          ) : (
            <button
              onClick={handleStop}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-[#EF4444]/15 text-[#EF4444] border border-[#EF4444]/20 hover:bg-[#EF4444]/25 transition-colors"
            >
              <Square className="w-3.5 h-3.5" /> Stop Session
            </button>
          )}
        </div>
      </motion.div>

      {lastError && (
        <motion.p variants={fadeUp} className="text-xs text-[#EF4444]/80 -mt-2">
          {lastError}
        </motion.p>
      )}

      {sessionStatus === "active" && systemStatus?.session_mode === "audio_only" && (
        <motion.p variants={fadeUp} className="text-xs text-[#F59E0B]/80 -mt-2">
          Audio-only mode — visual metrics (eye contact, posture) unavailable. WPM and filler words are tracking from speech.
        </motion.p>
      )}

      {sessionStatus === "active" && systemStatus?.session_state === "degraded" && systemStatus?.session_mode !== "audio_only" && (
        <motion.p variants={fadeUp} className="text-xs text-[#F59E0B]/80 -mt-2">
          Session degraded — some capabilities are limited. Check camera and network.
        </motion.p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Video Feed - Left Column */}
        <motion.div variants={fadeUp} className="lg:col-span-3">
          <StreamCallPanel
            connectionStatus={connectionStatus}
            sessionStatus={sessionStatus}
            streaming={isSessionActive}
            onReadyChange={setStreamReady}
            onMediaReadyChange={setStreamMediaReady}
            onCallIdChange={setStreamCallId}
          />
          {/* AI Communication Panel */}
          <motion.div variants={fadeUp} className="mt-4">
            <ChatPanel
              messages={chatMessages}
              transcript={transcript}
              conversationState={conversationState}
              onSendMessage={sendMessage}
              sessionActive={isSessionActive}
            />
          </motion.div>
        </motion.div>

        {/* Metrics - Right Column */}
        <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
          {/* Words Per Minute */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Words Per Minute"
              value={Math.round(metrics.words_per_minute)}
              suffix="WPM"
              trend={wpmTrend}
              trendValue={wpmTrend === "up" ? "Good pace" : "Adjust pace"}
              icon={<Gauge className="w-5 h-5" />}
              color="blue"
            >
              <SemiCircleGauge value={metrics.words_per_minute} max={200} color="#4F8CFF" />
            </MetricCard>
          </motion.div>

          {/* Eye Contact */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Eye Contact"
              value={Math.round(metrics.eye_contact)}
              suffix="%"
              trend={eyeTrend}
              trendValue={eyeTrend === "up" ? "Strong" : "Needs focus"}
              icon={<Eye className="w-5 h-5" />}
              color="green"
            >
              <ProgressBar value={metrics.eye_contact} color="#22C55E" />
            </MetricCard>
          </motion.div>

          {/* Filler Words */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Filler Words"
              value={Math.round(metrics.filler_words)}
              trend={fillerTrend}
              trendValue={fillerTrend === "down" ? "Low" : "High"}
              icon={<MessageCircleWarning className="w-5 h-5" />}
              color="warning"
            >
              <ProgressBar value={Math.min(metrics.filler_words * 10, 100)} color="#F59E0B" />
            </MetricCard>
          </motion.div>

          {/* Posture Score */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Posture Score"
              value={Math.round(metrics.posture_score)}
              suffix="%"
              trend={postureTrend}
              trendValue={postureTrend === "up" ? "Aligned" : "Adjust"}
              icon={<PersonStanding className="w-5 h-5" />}
              color="violet"
            >
              <div className="flex justify-center">
                <CircularProgress value={metrics.posture_score} color="#8B5CF6" />
              </div>
            </MetricCard>
          </motion.div>
        </div>
      </div>

      {/* Session Summary Row */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-6" hover={false}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white/80">
              Session Overview
            </h3>
            <span className="text-xs text-white/30 font-mono capitalize">
              {sessionStatus === "idle" ? "Standby" : `${sessionStatus} session`}
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {[
              { label: "Head Stability", value: `${Math.round(metrics.head_stability)}%`, sub: metrics.head_stability >= 70 ? "Steady" : "Unstable" },
              { label: "Engagement", value: `${Math.round(metrics.facial_engagement)}%`, sub: metrics.facial_engagement >= 70 ? "Expressive" : "Flat" },
              { label: "Attention", value: `${Math.round(metrics.attention_intensity)}%`, sub: metrics.attention_intensity >= 70 ? "Focused" : "Distracted" },
              { label: "Overall", value: `${Math.round((metrics.eye_contact + metrics.posture_score + metrics.head_stability + metrics.facial_engagement) / 4)}%`, sub: "Composite" },
              { label: "Conversation", value: `${conversationState?.turn_count ?? 0}`, sub: conversationState?.is_agent_speaking ? "AI Speaking" : conversationState?.is_user_speaking ? "User Speaking" : `${chatMessages.length} msgs` },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 + i * 0.1 }}
              >
                <p className="text-xs text-white/40 mb-1">{stat.label}</p>
                <p className="text-xl font-bold text-white/90">{stat.value}</p>
                <p className="text-[11px] text-white/30 mt-0.5">{stat.sub}</p>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  );
}
