"use client";

import { motion } from "framer-motion";
import Badge from "@/components/ui/Badge";
import { ScanLine, Video } from "lucide-react";

export default function CameraFeed() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="relative w-full aspect-video rounded-2xl overflow-hidden border border-white/[0.08] bg-black/40"
    >
      {/* Neon border glow */}
      <div className="absolute inset-0 rounded-2xl shadow-[inset_0_0_30px_rgba(79,140,255,0.08)]" />

      {/* Scanning grid overlay */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(79,140,255,0.5) 1px, transparent 1px),
            linear-gradient(90deg, rgba(79,140,255,0.5) 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px",
        }}
      />

      {/* Scanning line animation */}
      <motion.div
        className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#4F8CFF]/40 to-transparent"
        animate={{ top: ["0%", "100%", "0%"] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* Camera placeholder */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="relative">
          <motion.div
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="w-20 h-20 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center justify-center"
          >
            <Video className="w-8 h-8 text-white/20" />
          </motion.div>
          <motion.div
            animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.05, 0.2] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 rounded-full border border-[#4F8CFF]/30"
          />
        </div>
        <p className="text-white/20 text-sm mt-4">Camera feed will appear here</p>
      </div>

      {/* Top badges */}
      <div className="absolute top-4 left-4 flex items-center gap-2">
        <Badge variant="critical" pulse>
          LIVE ANALYSIS
        </Badge>
      </div>

      <div className="absolute top-4 right-4">
        <Badge variant="blue">
          <ScanLine className="w-3 h-3" />
          AI Tracking Active
        </Badge>
      </div>

      {/* Bottom status bar */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <motion.div
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-2 h-2 bg-[#EF4444] rounded-full"
            />
            <span className="text-xs text-white/50">Recording</span>
          </div>
          <span className="text-xs text-white/30 font-mono">1080p • 30fps</span>
        </div>
      </div>

      {/* Corner markers */}
      <svg className="absolute top-3 left-3 w-6 h-6 text-[#4F8CFF]/30">
        <path d="M0 8V2a2 2 0 012-2h6" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      <svg className="absolute top-3 right-3 w-6 h-6 text-[#4F8CFF]/30 rotate-90">
        <path d="M0 8V2a2 2 0 012-2h6" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      <svg className="absolute bottom-3 left-3 w-6 h-6 text-[#4F8CFF]/30 -rotate-90">
        <path d="M0 8V2a2 2 0 012-2h6" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      <svg className="absolute bottom-3 right-3 w-6 h-6 text-[#4F8CFF]/30 rotate-180">
        <path d="M0 8V2a2 2 0 012-2h6" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </motion.div>
  );
}
