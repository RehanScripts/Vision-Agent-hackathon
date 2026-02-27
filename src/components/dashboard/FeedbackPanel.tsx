"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Lightbulb,
  AlertTriangle,
  X,
  Zap,
  MessageCircle,
} from "lucide-react";
import GlassCard from "@/components/ui/GlassCard";

interface FeedbackItem {
  id: number;
  severity: "info" | "warning" | "critical";
  headline: string;
  explanation: string;
  tip?: string;
}

const feedbackData: FeedbackItem[] = [
  {
    id: 1,
    severity: "info",
    headline: "Great Eye Contact",
    explanation:
      "You've maintained consistent eye contact for the past 30 seconds.",
    tip: "Keep focusing on the center of the camera lens.",
  },
  {
    id: 2,
    severity: "warning",
    headline: "Speaking Too Fast",
    explanation: "Your pace increased to 185 WPM in the last segment.",
    tip: "Take a brief pause between key points to let ideas land.",
  },
  {
    id: 3,
    severity: "critical",
    headline: 'Filler Word Detected: "Um"',
    explanation:
      'You used "um" 3 times in the last 15 seconds.',
    tip: "Try replacing filler words with a deliberate pause.",
  },
  {
    id: 4,
    severity: "info",
    headline: "Posture Improved",
    explanation: "Your shoulder alignment is now at 94% — up from 78%.",
  },
];

const severityConfig = {
  info: {
    icon: Lightbulb,
    color: "text-[#4F8CFF]",
    glow: "blue" as const,
    bg: "bg-[#4F8CFF]/10",
    border: "border-[#4F8CFF]/20",
  },
  warning: {
    icon: AlertTriangle,
    color: "text-[#F59E0B]",
    glow: "warning" as const,
    bg: "bg-[#F59E0B]/10",
    border: "border-[#F59E0B]/20",
  },
  critical: {
    icon: Zap,
    color: "text-[#EF4444]",
    glow: "critical" as const,
    bg: "bg-[#EF4444]/10",
    border: "border-[#EF4444]/20",
  },
};

export default function FeedbackPanel() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % feedbackData.length);
      setVisible(true);
    }, 6000);
    return () => clearInterval(interval);
  }, []);

  const feedback = feedbackData[currentIndex];
  const config = severityConfig[feedback.severity];
  const Icon = config.icon;

  return (
    <div className="fixed bottom-6 right-6 z-50 w-[380px] max-w-[calc(100vw-3rem)]">
      <AnimatePresence mode="wait">
        {visible && (
          <motion.div
            key={feedback.id}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
          >
            <GlassCard
              hover={false}
              glow={config.glow}
              className="p-4 border-white/[0.08]"
            >
              <div className="flex items-start gap-3">
                <div
                  className={cn(
                    "flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center",
                    config.bg,
                    config.border,
                    "border"
                  )}
                >
                  <motion.div
                    animate={{ rotate: [0, -5, 5, 0] }}
                    transition={{
                      duration: 0.5,
                      delay: 0.2,
                    }}
                  >
                    <Icon className={cn("w-4 h-4", config.color)} />
                  </motion.div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <MessageCircle className="w-3 h-3 text-white/30" />
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-white/30">
                          AI Coach
                        </span>
                      </div>
                      <h4 className="text-sm font-semibold text-white/90">
                        {feedback.headline}
                      </h4>
                    </div>
                    <button
                      onClick={() => setVisible(false)}
                      className="flex-shrink-0 p-1 rounded-lg hover:bg-white/[0.06] transition-colors text-white/30 hover:text-white/60"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    {feedback.explanation}
                  </p>
                  {feedback.tip && (
                    <div className="mt-2 flex items-start gap-2 p-2 rounded-lg bg-white/[0.03] border border-white/[0.04]">
                      <Lightbulb className="w-3 h-3 text-[#F59E0B] mt-0.5 flex-shrink-0" />
                      <p className="text-[11px] text-white/40 leading-relaxed">
                        {feedback.tip}
                      </p>
                    </div>
                  )}
                </div>
              </div>
              {/* Progress bar */}
              <div className="mt-3 h-0.5 bg-white/[0.04] rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: "0%" }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 6, ease: "linear" }}
                  className={cn(
                    "h-full rounded-full",
                    feedback.severity === "info" && "bg-[#4F8CFF]/40",
                    feedback.severity === "warning" && "bg-[#F59E0B]/40",
                    feedback.severity === "critical" && "bg-[#EF4444]/40"
                  )}
                />
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
