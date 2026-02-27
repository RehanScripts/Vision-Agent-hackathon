"use client";

import { motion } from "framer-motion";
import CameraFeed from "@/components/dashboard/CameraFeed";
import MetricCard from "@/components/ui/MetricCard";
import GlassCard from "@/components/ui/GlassCard";
import AnimatedNumber from "@/components/ui/AnimatedNumber";
import {
  Gauge,
  Eye,
  MessageCircleWarning,
  PersonStanding,
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
  const radius = 50;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference * (1 - percentage);

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

export default function DashboardPage() {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Camera Feed - Left Column */}
        <motion.div variants={fadeUp} className="lg:col-span-3">
          <CameraFeed />
        </motion.div>

        {/* Metrics - Right Column */}
        <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
          {/* Words Per Minute */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Words Per Minute"
              value={142}
              suffix="WPM"
              trend="up"
              trendValue="+8%"
              icon={<Gauge className="w-5 h-5" />}
              color="blue"
            >
              <SemiCircleGauge value={142} max={200} color="#4F8CFF" />
            </MetricCard>
          </motion.div>

          {/* Eye Contact */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Eye Contact"
              value={87}
              suffix="%"
              trend="up"
              trendValue="+12%"
              icon={<Eye className="w-5 h-5" />}
              color="green"
            >
              <ProgressBar value={87} color="#22C55E" />
            </MetricCard>
          </motion.div>

          {/* Filler Words */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Filler Words"
              value={4}
              trend="down"
              trendValue="-2"
              icon={<MessageCircleWarning className="w-5 h-5" />}
              color="warning"
            >
              <div className="flex items-center gap-3">
                {["Um", "Uh", "Like", "So"].map((word, i) => (
                  <motion.span
                    key={word}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + i * 0.1 }}
                    className="px-2 py-1 text-[10px] font-medium rounded-md bg-[#F59E0B]/10 text-[#F59E0B]/70 border border-[#F59E0B]/10"
                  >
                    {word}: {i === 0 ? 2 : i === 1 ? 1 : i === 2 ? 1 : 0}
                  </motion.span>
                ))}
              </div>
            </MetricCard>
          </motion.div>

          {/* Posture Score */}
          <motion.div variants={fadeUp}>
            <MetricCard
              label="Posture Score"
              value={92}
              suffix="%"
              trend="up"
              trendValue="+5%"
              icon={<PersonStanding className="w-5 h-5" />}
              color="violet"
            >
              <div className="flex justify-center">
                <CircularProgress value={92} color="#8B5CF6" />
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
            <span className="text-xs text-white/30 font-mono">
              Session #47
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { label: "Duration", value: "3m 42s", sub: "Active" },
              { label: "Clarity Score", value: "94%", sub: "+6% from last" },
              { label: "Engagement", value: "A+", sub: "Excellent" },
              { label: "Confidence", value: "88%", sub: "High" },
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
