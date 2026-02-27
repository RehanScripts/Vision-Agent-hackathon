"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "blue" | "violet" | "green" | "warning" | "critical" | "neutral";
  pulse?: boolean;
  className?: string;
}

const variantStyles = {
  blue: "bg-[#4F8CFF]/15 text-[#4F8CFF] border-[#4F8CFF]/20",
  violet: "bg-[#8B5CF6]/15 text-[#8B5CF6] border-[#8B5CF6]/20",
  green: "bg-[#22C55E]/15 text-[#22C55E] border-[#22C55E]/20",
  warning: "bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/20",
  critical: "bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/20",
  neutral: "bg-white/10 text-white/70 border-white/10",
};

export default function Badge({
  children,
  variant = "blue",
  pulse = false,
  className,
}: BadgeProps) {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border",
        variantStyles[variant],
        className
      )}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span
            className={cn(
              "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
              variant === "green" && "bg-[#22C55E]",
              variant === "blue" && "bg-[#4F8CFF]",
              variant === "violet" && "bg-[#8B5CF6]",
              variant === "warning" && "bg-[#F59E0B]",
              variant === "critical" && "bg-[#EF4444]",
              variant === "neutral" && "bg-white/50"
            )}
          />
          <span
            className={cn(
              "relative inline-flex rounded-full h-2 w-2",
              variant === "green" && "bg-[#22C55E]",
              variant === "blue" && "bg-[#4F8CFF]",
              variant === "violet" && "bg-[#8B5CF6]",
              variant === "warning" && "bg-[#F59E0B]",
              variant === "critical" && "bg-[#EF4444]",
              variant === "neutral" && "bg-white/50"
            )}
          />
        </span>
      )}
      {children}
    </motion.span>
  );
}
