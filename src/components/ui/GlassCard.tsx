"use client";

import { cn } from "@/lib/utils";
import { motion, type HTMLMotionProps } from "framer-motion";
import { forwardRef } from "react";

interface GlassCardProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: "blue" | "violet" | "green" | "warning" | "critical" | "none";
}

const glowColors = {
  blue: "shadow-[0_0_30px_-5px_rgba(79,140,255,0.15)]",
  violet: "shadow-[0_0_30px_-5px_rgba(139,92,246,0.15)]",
  green: "shadow-[0_0_30px_-5px_rgba(34,197,94,0.15)]",
  warning: "shadow-[0_0_30px_-5px_rgba(245,158,11,0.15)]",
  critical: "shadow-[0_0_30px_-5px_rgba(239,68,68,0.15)]",
  none: "",
};

const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, className, hover = true, glow = "none", ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={cn(
          "relative rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-xl",
          glowColors[glow],
          className
        )}
        whileHover={
          hover
            ? {
                y: -2,
                borderColor: "rgba(255,255,255,0.12)",
                transition: { duration: 0.2 },
              }
            : undefined
        }
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

GlassCard.displayName = "GlassCard";

export default GlassCard;
