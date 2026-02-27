"use client";

import { motion } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import AnimatedNumber from "@/components/ui/AnimatedNumber";
import Badge from "@/components/ui/Badge";
import PerformanceChart from "@/components/analytics/PerformanceChart";
import {
  TrendingUp,
  Eye,
  Gauge,
  MessageCircleWarning,
  PersonStanding,
  Sparkles,
  Target,
  Clock,
  Award,
} from "lucide-react";

const statCards = [
  {
    label: "Avg. WPM",
    value: 138,
    icon: Gauge,
    color: "#4F8CFF",
    trend: "+5%",
    trendUp: true,
  },
  {
    label: "Avg. Eye Contact",
    value: 85,
    suffix: "%",
    icon: Eye,
    color: "#22C55E",
    trend: "+12%",
    trendUp: true,
  },
  {
    label: "Filler Words / Min",
    value: 2,
    icon: MessageCircleWarning,
    color: "#F59E0B",
    trend: "-40%",
    trendUp: true,
  },
  {
    label: "Posture Score",
    value: 91,
    suffix: "%",
    icon: PersonStanding,
    color: "#8B5CF6",
    trend: "+8%",
    trendUp: true,
  },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function AnalyticsPage() {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Overall Score */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-8 text-center" hover={false} glow="blue">
          <div className="flex flex-col items-center">
            <div className="flex items-center gap-2 mb-3">
              <Award className="w-5 h-5 text-[#4F8CFF]" />
              <span className="text-xs font-semibold uppercase tracking-wider text-white/40">
                Overall Performance Score
              </span>
            </div>
            <div className="relative">
              <AnimatedNumber
                value={87}
                className="text-7xl font-bold bg-gradient-to-b from-white to-white/50 bg-clip-text text-transparent"
              />
              <span className="text-2xl font-bold text-white/30 ml-1">/100</span>
            </div>
            <div className="flex items-center gap-2 mt-3">
              <Badge variant="green" pulse>
                <TrendingUp className="w-3 h-3" />
                +5 from last session
              </Badge>
            </div>
            <div className="flex items-center gap-6 mt-6 text-xs text-white/40">
              <div className="flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5" />
                <span>47 sessions analyzed</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                <span>12.5 hours total</span>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, i) => (
          <motion.div key={stat.label} variants={fadeUp}>
            <GlassCard className="p-5">
              <div className="flex items-start justify-between mb-3">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: `${stat.color}15` }}
                >
                  <stat.icon
                    className="w-5 h-5"
                    style={{ color: stat.color }}
                  />
                </div>
                <div
                  className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-lg ${
                    stat.trendUp
                      ? "text-[#22C55E] bg-[#22C55E]/10"
                      : "text-[#EF4444] bg-[#EF4444]/10"
                  }`}
                >
                  <TrendingUp className="w-3 h-3" />
                  {stat.trend}
                </div>
              </div>
              <div className="flex items-baseline gap-1">
                <AnimatedNumber
                  value={stat.value}
                  className="text-3xl font-bold tracking-tight"
                  suffix={stat.suffix}
                />
                {stat.suffix && (
                  <span className="text-sm text-white/40 font-medium">
                    {stat.suffix}
                  </span>
                )}
              </div>
              <p className="text-sm text-white/50 mt-1">{stat.label}</p>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      {/* Performance Chart */}
      <motion.div variants={fadeUp}>
        <PerformanceChart />
      </motion.div>

      {/* AI Summary */}
      <motion.div variants={fadeUp}>
        <GlassCard className="p-6" hover={false} glow="violet">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-[#8B5CF6]/10 border border-[#8B5CF6]/20 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-[#8B5CF6]" />
            </div>
            <div className="flex-1">
              <h3 className="text-base font-semibold text-white/90 mb-2">
                AI Performance Summary
              </h3>
              <div className="space-y-3 text-sm text-white/50 leading-relaxed">
                <p>
                  Your speaking pace has improved significantly over the last 10
                  sessions, stabilizing around 135-145 WPM — well within the
                  optimal range for presentations.
                </p>
                <p>
                  <span className="text-[#22C55E] font-medium">Strength:</span>{" "}
                  Eye contact consistency has reached 85%, placing you in the top
                  15% of all users.
                </p>
                <p>
                  <span className="text-[#F59E0B] font-medium">
                    Area to improve:
                  </span>{" "}
                  Filler word usage tends to spike during the first 30 seconds.
                  Consider opening with a practiced statement.
                </p>
                <p>
                  <span className="text-[#8B5CF6] font-medium">
                    Recommendation:
                  </span>{" "}
                  Try a 5-minute Interview Practice session focusing on
                  eliminating opening fillers.
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    </motion.div>
  );
}
