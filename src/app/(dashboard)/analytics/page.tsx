"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import GlassCard from "@/components/ui/GlassCard";
import AnimatedNumber from "@/components/ui/AnimatedNumber";
import Badge from "@/components/ui/Badge";
import PerformanceChart from "@/components/analytics/PerformanceChart";
import {
  TrendingUp,
  TrendingDown,
  Eye,
  Gauge,
  MessageCircleWarning,
  PersonStanding,
  Sparkles,
  Target,
  Clock,
  Award,
  BarChart3,
} from "lucide-react";
import { computeAnalytics, AnalyticsOverview } from "@/lib/sessionStore";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 15 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
};

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);

  useEffect(() => {
    setAnalytics(computeAnalytics());
  }, []);

  const statCards = useMemo(() => {
    if (!analytics) return [];
    return [
      {
        label: "Avg. WPM",
        value: analytics.avgWPM,
        icon: Gauge,
        color: "#4F8CFF",
      },
      {
        label: "Avg. Eye Contact",
        value: analytics.avgEyeContact,
        suffix: "%",
        icon: Eye,
        color: "#22C55E",
      },
      {
        label: "Filler Words / Min",
        value: analytics.avgFillerWords,
        icon: MessageCircleWarning,
        color: "#F59E0B",
      },
      {
        label: "Posture Score",
        value: analytics.avgPosture,
        suffix: "%",
        icon: PersonStanding,
        color: "#8B5CF6",
      },
    ];
  }, [analytics]);

  const chartData = useMemo(() => {
    if (!analytics || analytics.trend.length === 0) return undefined;
    return analytics.trend.map((t) => ({
      time: `Session ${t.index + 1}`,
      wpm: t.wpm,
      eyeContact: t.eyeContact,
      posture: t.posture,
    }));
  }, [analytics]);

  const hasSessions = analytics !== null && analytics.totalSessions > 0;
  const scoreChangePositive = (analytics?.scoreChange ?? 0) >= 0;
  const hoursTotal = analytics ? (analytics.totalMinutes / 60).toFixed(1) : "0";

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
                value={hasSessions ? analytics.avgScore : 0}
                className="text-7xl font-bold bg-gradient-to-b from-white to-white/50 bg-clip-text text-transparent"
              />
              <span className="text-2xl font-bold text-white/30 ml-1">/100</span>
            </div>
            {hasSessions && analytics.scoreChange !== 0 && (
              <div className="flex items-center gap-2 mt-3">
                <Badge variant={scoreChangePositive ? "green" : "warning"} pulse>
                  {scoreChangePositive ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  {scoreChangePositive ? "+" : ""}
                  {analytics.scoreChange} from recent sessions
                </Badge>
              </div>
            )}
            <div className="flex items-center gap-6 mt-6 text-xs text-white/40">
              <div className="flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5" />
                <span>{analytics?.totalSessions ?? 0} sessions analyzed</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                <span>{hoursTotal} hours total</span>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* No data state */}
      {!hasSessions && (
        <motion.div variants={fadeUp}>
          <GlassCard className="p-12 text-center">
            <BarChart3 className="w-8 h-8 text-white/20 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-white/50 mb-1">No analytics data yet</h3>
            <p className="text-xs text-white/30">
              Complete a practice session to see your performance analytics.
            </p>
          </GlassCard>
        </motion.div>
      )}

      {/* Stat Cards */}
      {hasSessions && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {statCards.map((stat) => (
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
      )}

      {/* Performance Chart */}
      {hasSessions && (
        <motion.div variants={fadeUp}>
          <PerformanceChart
            data={chartData}
            subtitle={`Across last ${analytics.trend.length} sessions`}
          />
        </motion.div>
      )}

      {/* AI Summary */}
      {hasSessions && (
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
                    Based on <span className="text-white/80 font-medium">{analytics.totalSessions} sessions</span> totaling{" "}
                    <span className="text-white/80 font-medium">{analytics.totalMinutes} minutes</span> of practice,
                    your average score is <span className="text-white/80 font-medium">{analytics.avgScore}/100</span>.
                  </p>
                  {analytics.avgEyeContact >= 75 ? (
                    <p>
                      <span className="text-[#22C55E] font-medium">Strength:</span>{" "}
                      Eye contact consistency is at {analytics.avgEyeContact}% — excellent engagement.
                    </p>
                  ) : (
                    <p>
                      <span className="text-[#F59E0B] font-medium">Area to improve:</span>{" "}
                      Eye contact is at {analytics.avgEyeContact}%. Try focusing on looking into the camera more consistently.
                    </p>
                  )}
                  {analytics.avgFillerWords <= 3 ? (
                    <p>
                      <span className="text-[#22C55E] font-medium">Strength:</span>{" "}
                      Filler word usage is low at ~{analytics.avgFillerWords}/min. Great verbal control!
                    </p>
                  ) : (
                    <p>
                      <span className="text-[#F59E0B] font-medium">Area to improve:</span>{" "}
                      Filler word usage is ~{analytics.avgFillerWords}/min. Try pausing instead of using fillers.
                    </p>
                  )}
                  <p>
                    <span className="text-[#8B5CF6] font-medium">Recommendation:</span>{" "}
                    {analytics.avgWPM > 160
                      ? "Your pace tends to run fast. Try slowing down by 10-15% for better clarity."
                      : analytics.avgWPM < 110
                        ? "Your pace is on the slower side. Try increasing energy and speaking with more momentum."
                        : "Your speaking pace is in the optimal range. Keep up the consistent practice!"}
                  </p>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}
    </motion.div>
  );
}
