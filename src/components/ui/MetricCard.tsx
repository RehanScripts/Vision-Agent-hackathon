"use client";

import GlassCard from "./GlassCard";
import AnimatedNumber from "./AnimatedNumber";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: number;
  suffix?: string;
  trend?: "up" | "down";
  trendValue?: string;
  icon: React.ReactNode;
  color?: "blue" | "violet" | "green" | "warning";
  children?: React.ReactNode;
}

const colorMap = {
  blue: "text-[#4F8CFF]",
  violet: "text-[#8B5CF6]",
  green: "text-[#22C55E]",
  warning: "text-[#F59E0B]",
};

const bgColorMap = {
  blue: "bg-[#4F8CFF]/10",
  violet: "bg-[#8B5CF6]/10",
  green: "bg-[#22C55E]/10",
  warning: "bg-[#F59E0B]/10",
};

export default function MetricCard({
  label,
  value,
  suffix = "",
  trend,
  trendValue,
  icon,
  color = "blue",
  children,
}: MetricCardProps) {
  return (
    <GlassCard className="p-5" glow="none">
      <div className="flex items-start justify-between mb-3">
        <div
          className={cn(
            "flex items-center justify-center w-10 h-10 rounded-xl",
            bgColorMap[color]
          )}
        >
          <span className={cn(colorMap[color])}>{icon}</span>
        </div>
        {trend && (
          <motion.div
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            className={cn(
              "flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-lg",
              trend === "up"
                ? "text-[#22C55E] bg-[#22C55E]/10"
                : "text-[#EF4444] bg-[#EF4444]/10"
            )}
          >
            {trend === "up" ? (
              <TrendingUp className="w-3 h-3" />
            ) : (
              <TrendingDown className="w-3 h-3" />
            )}
            {trendValue}
          </motion.div>
        )}
      </div>
      <div className="space-y-1">
        <div className="flex items-baseline gap-1">
          <AnimatedNumber
            value={value}
            className={cn("text-3xl font-bold tracking-tight", colorMap[color])}
          />
          {suffix && (
            <span className="text-sm text-white/40 font-medium">{suffix}</span>
          )}
        </div>
        <p className="text-sm text-white/50">{label}</p>
      </div>
      {children && <div className="mt-4">{children}</div>}
    </GlassCard>
  );
}
