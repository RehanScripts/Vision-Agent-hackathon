"use client";

import { motion } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import Badge from "@/components/ui/Badge";
import {
  Clock,
  TrendingUp,
  Calendar,
  ChevronRight,
  Play,
} from "lucide-react";

const sessions = [
  {
    id: 1,
    title: "Interview Practice — Google PM",
    date: "Feb 27, 2026",
    duration: "8m 32s",
    score: 91,
    type: "Interview",
    improvement: "+4%",
  },
  {
    id: 2,
    title: "MUN Opening Statement",
    date: "Feb 25, 2026",
    duration: "5m 14s",
    score: 87,
    type: "MUN Debate",
    improvement: "+7%",
  },
  {
    id: 3,
    title: "Startup Pitch — Series A",
    date: "Feb 23, 2026",
    duration: "12m 08s",
    score: 83,
    type: "Pitch",
    improvement: "+2%",
  },
  {
    id: 4,
    title: "Weekly Team Presentation",
    date: "Feb 20, 2026",
    duration: "6m 45s",
    score: 89,
    type: "Presentation",
    improvement: "+9%",
  },
  {
    id: 5,
    title: "Conference Talk Rehearsal",
    date: "Feb 18, 2026",
    duration: "15m 22s",
    score: 79,
    type: "Presentation",
    improvement: "+3%",
  },
  {
    id: 6,
    title: "Interview Practice — Meta",
    date: "Feb 15, 2026",
    duration: "9m 11s",
    score: 85,
    type: "Interview",
    improvement: "+6%",
  },
];

const typeColors: Record<string, "blue" | "violet" | "green" | "warning"> = {
  Interview: "blue",
  "MUN Debate": "violet",
  Pitch: "warning",
  Presentation: "green",
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
            <span>Last 30 days</span>
          </div>
          <span className="text-white/20">•</span>
          <span>{sessions.length} sessions</span>
        </div>
      </motion.div>

      {/* Sessions List */}
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
                        {session.date}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {session.duration}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4 flex-shrink-0 ml-4">
                  <div className="text-right">
                    <div className="text-lg font-bold text-white/90">
                      {session.score}
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[#22C55E]">
                      <TrendingUp className="w-3 h-3" />
                      {session.improvement}
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/40 transition-colors" />
                </div>
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      {/* Empty state hint */}
      <motion.div variants={fadeUp}>
        <p className="text-center text-xs text-white/20 py-4">
          Showing all sessions from the last 30 days
        </p>
      </motion.div>
    </motion.div>
  );
}
