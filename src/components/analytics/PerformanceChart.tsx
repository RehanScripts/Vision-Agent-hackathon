"use client";

import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import GlassCard from "@/components/ui/GlassCard";

const data = [
  { time: "0:00", wpm: 120, eyeContact: 70, posture: 80 },
  { time: "0:30", wpm: 135, eyeContact: 75, posture: 82 },
  { time: "1:00", wpm: 145, eyeContact: 65, posture: 85 },
  { time: "1:30", wpm: 130, eyeContact: 80, posture: 78 },
  { time: "2:00", wpm: 155, eyeContact: 85, posture: 90 },
  { time: "2:30", wpm: 140, eyeContact: 78, posture: 88 },
  { time: "3:00", wpm: 128, eyeContact: 90, posture: 92 },
  { time: "3:30", wpm: 135, eyeContact: 88, posture: 94 },
  { time: "4:00", wpm: 142, eyeContact: 92, posture: 91 },
  { time: "4:30", wpm: 138, eyeContact: 87, posture: 93 },
  { time: "5:00", wpm: 130, eyeContact: 91, posture: 95 },
];

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  label?: string;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#0B0F19]/95 backdrop-blur-xl border border-white/[0.08] rounded-xl p-3 shadow-2xl">
        <p className="text-white/40 text-xs mb-2 font-medium">{label}</p>
        {payload.map((entry, index) => (
          <div
            key={index}
            className="flex items-center gap-2 text-xs py-0.5"
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-white/50">{entry.name}:</span>
            <span className="text-white font-semibold">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function PerformanceChart() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <GlassCard className="p-6" hover={false}>
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-base font-semibold text-white/90">
              Performance Over Time
            </h3>
            <p className="text-xs text-white/40 mt-1">
              Real-time session metrics
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-[#4F8CFF]" />
              <span className="text-[11px] text-white/40">WPM</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-[#22C55E]" />
              <span className="text-[11px] text-white/40">Eye Contact</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-[#8B5CF6]" />
              <span className="text-[11px] text-white/40">Posture</span>
            </div>
          </div>
        </div>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.04)"
                vertical={false}
              />
              <XAxis
                dataKey="time"
                stroke="rgba(255,255,255,0.2)"
                tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                tickLine={false}
              />
              <YAxis
                stroke="rgba(255,255,255,0.2)"
                tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ display: "none" }}
              />
              <Line
                type="monotone"
                dataKey="wpm"
                name="WPM"
                stroke="#4F8CFF"
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#4F8CFF",
                  stroke: "#4F8CFF",
                  strokeWidth: 2,
                }}
              />
              <Line
                type="monotone"
                dataKey="eyeContact"
                name="Eye Contact"
                stroke="#22C55E"
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#22C55E",
                  stroke: "#22C55E",
                  strokeWidth: 2,
                }}
              />
              <Line
                type="monotone"
                dataKey="posture"
                name="Posture"
                stroke="#8B5CF6"
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 4,
                  fill: "#8B5CF6",
                  stroke: "#8B5CF6",
                  strokeWidth: 2,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>
    </motion.div>
  );
}
