"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import Badge from "@/components/ui/Badge";
import {
  Clock,
  TrendingUp,
  Calendar,
  ChevronRight,
  Play,
  Trash2,
} from "lucide-react";
import { getSessions, clearSessions, StoredSession } from "@/lib/sessionStore";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

const typeColors: Record<string, "blue" | "violet" | "green" | "warning"> = {
  Interview: "blue",
  "MUN Debate": "violet",
  Pitch: "warning",
  Presentation: "green",
  "Free Practice": "neutral" as never,
};

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function HistoryPage() {
  const [sessions, setSessions] = useState<StoredSession[]>(() => getSessions());

  const handleClear = () => {
    clearSessions();
    setSessions([]);
  };

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={fadeUp} className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white/90">Session History</h2>
          <p className="text-sm text-white/40 mt-1">Review your past practice sessions</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-white/40">
          <div className="flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5" />
            <span>All time</span>
          </div>
          <span className="text-white/20">•</span>
          <span>{sessions.length} sessions</span>
          {sessions.length > 0 && (
            <>
              <span className="text-white/20">•</span>
              <button
                onClick={handleClear}
                className="flex items-center gap-1 text-red-400/60 hover:text-red-400 transition-colors"
              >
                <Trash2 className="w-3 h-3" />
                Clear
              </button>
            </>
          )}
        </div>
      </motion.div>

      {/* Sessions List */}
      {sessions.length === 0 ? (
        <motion.div variants={fadeUp}>
          <GlassCard className="p-12 text-center">
            <Play className="w-8 h-8 text-white/20 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-white/50 mb-1">No sessions yet</h3>
            <p className="text-xs text-white/30">
              Complete a practice session and it will appear here.
            </p>
          </GlassCard>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {sessions.map((session) => (
            <motion.div key={session.id} variants={fadeUp}>
              <GlassCard className="p-5 cursor-pointer group">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 min-w-0 flex-1">
                    <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center group-hover:bg-white/[0.06] transition-colors">
                      <Play className="w-4 h-4 text-white/40" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-sm font-medium text-white/80 truncate">
                          {session.title}
                        </h3>
                        <Badge variant={typeColors[session.type] || "neutral"}>
                          {session.type}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-white/40">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {formatDate(session.date)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatDuration(session.durationSeconds)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 flex-shrink-0 ml-4">
                    <div className="text-right">
                      <div className="text-lg font-bold text-white/90">
                        {session.score}
                      </div>
                      <div className="flex items-center gap-1 text-xs text-white/40">
                        <TrendingUp className="w-3 h-3" />
                        /100
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/40 transition-colors" />
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      )}

      {/* Footer */}
      <motion.div variants={fadeUp}>
        <p className="text-center text-xs text-white/20 py-4">
          {sessions.length > 0
            ? `Showing all ${sessions.length} sessions`
            : "Start a practice session to begin building your history"}
        </p>
      </motion.div>
    </motion.div>
  );
}
